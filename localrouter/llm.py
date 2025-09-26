from typing import (
    List,
    Callable,
    Type,
    Union,
    Optional,
    Any,
    Dict,
    AsyncIterator,
    Pattern,
)
import os
import re
import anthropic
import openai
from google import genai
from google.genai import types as genai_types
import json
import backoff
from typing import Dict, Any
from uuid import uuid4
from pydantic import BaseModel
from cache_on_disk import DCache
from .dtypes import (
    ChatMessage,
    MessageRole,
    anthropic_format,
    openai_format,
    genai_format,
    TextBlock,
    ToolDefinition,
    ReasoningConfig,
    ThinkingBlock,
)

from dotenv import load_dotenv

load_dotenv()


class Provider:
    def __init__(
        self,
        get_response: Callable[..., Any],
        models: List[Union[str, Pattern]],
        priority: int = 100,
    ) -> None:
        self.models = models
        self.get_response = get_response
        self.priority = priority

    def supports_model(self, model: str) -> bool:
        """Check if this provider supports the given model."""
        for model_pattern in self.models:
            if isinstance(model_pattern, str):
                if model == model_pattern:
                    return True
            elif isinstance(model_pattern, Pattern):
                if model_pattern.match(model):
                    return True
        return False


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------

anthr = anthropic.AsyncAnthropic()


async def get_response_anthropic(
    messages: List[ChatMessage],
    tools: Optional[List[ToolDefinition]],
    response_format: Optional[Dict[str, Any]] = None,
    reasoning: Optional[Union[ReasoningConfig, Dict[str, Any]]] = None,
    **kwargs: Any,
) -> ChatMessage:
    if response_format is not None:
        raise NotImplementedError(
            "Structured output is not supported for Anthropic models"
        )

    kwargs = anthropic_format(messages, tools, reasoning=reasoning, **kwargs)
    kwargs["timeout"] = 599
    resp = await anthr.messages.create(**kwargs)

    return ChatMessage.from_anthropic(resp.content)


# ---------------------------------------------------------------------------
# OpenAI provider factory
# ---------------------------------------------------------------------------


def get_response_factory(oai: openai.AsyncOpenAI) -> Callable[..., Any]:
    async def get_response_openai(
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]],
        response_format: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None,
        reasoning: Optional[Union[ReasoningConfig, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> ChatMessage:
        kwargs = openai_format(messages, tools, reasoning=reasoning, **kwargs)

        if "model" not in kwargs:
            raise ValueError("'model' is required for OpenAI completions")

        # Handle structured output
        if (
            response_format is not None
            and isinstance(response_format, type)
            and issubclass(response_format, BaseModel)
        ):
            resp = await oai.chat.completions.parse(
                response_format=response_format, **kwargs
            )
            response = ChatMessage.from_openai(resp.choices[0].message)
            response.parsed = resp.choices[0].message.parsed
            return response

        # Regular completion
        if response_format is not None:
            kwargs["response_format"] = response_format

        resp = await oai.chat.completions.create(**kwargs)
        return ChatMessage.from_openai(resp.choices[0].message)

    return get_response_openai


# ---------------------------------------------------------------------------
# Google GenAI provider
# ---------------------------------------------------------------------------


async def get_response_genai(
    messages: List[ChatMessage],
    tools: Optional[List[ToolDefinition]],
    response_format: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None,
    reasoning: Optional[Union[ReasoningConfig, Dict[str, Any]]] = None,
    **kwargs: Any,
) -> ChatMessage:

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY or GOOGLE_API_KEY environment variable is required"
        )

    client = genai.Client(api_key=api_key)
    request_kwargs = genai_format(messages, tools, reasoning=reasoning)

    # Build config
    config_params: Dict[str, Any] = {}
    for k, v in kwargs.items():
        if k not in [
            "contents",
            "tools",
            "system_instruction",
            "model",
            "thinking_budget",
        ]:
            if k == "max_tokens":
                config_params["max_output_tokens"] = v
            elif k in ["temperature", "top_p"]:
                config_params[k] = v

    # Handle structured output
    if (
        response_format is not None
        and isinstance(response_format, type)
        and issubclass(response_format, BaseModel)
    ):
        config_params.update(
            {
                "response_mime_type": "application/json",
                "response_schema": response_format,
            }
        )

    # Add tools and system instruction from request
    if "tools" in request_kwargs and request_kwargs["tools"]:
        config_params["tools"] = request_kwargs["tools"]
    if "system_instruction" in request_kwargs and request_kwargs["system_instruction"]:
        config_params["system_instruction"] = request_kwargs["system_instruction"]

    # Handle thinking budget
    if "thinking_budget" in request_kwargs:
        config_params["thinking_config"] = genai_types.ThinkingConfig(
            thinking_budget=request_kwargs["thinking_budget"],
            include_thoughts=True,  # Include thought summaries
        )

    config = (
        genai_types.GenerateContentConfig(**config_params) if config_params else None
    )
    # Make request
    response = await client.aio.models.generate_content(
        model=kwargs.get("model", "gemini-2.5-pro"),
        contents=request_kwargs["contents"],
        config=config,
    )
    # Convert response
    chat_response = ChatMessage.from_genai(response)

    # Handle structured output parsing
    if (
        response_format is not None
        and isinstance(response_format, type)
        and issubclass(response_format, BaseModel)
    ):
        if hasattr(response, "parsed") and response.parsed:
            chat_response.parsed = response.parsed
        elif response.text:
            try:
                parsed_data = json.loads(response.text)
                chat_response.parsed = response_format(**parsed_data)
            except Exception:
                pass

    return chat_response


# ---------------------------------------------------------------------------
# Provider registration
# ---------------------------------------------------------------------------

providers: List[Provider] = []

# Anthropic (priority 10 - higher priority than OpenRouter)
try:
    _available_anthropic_models = [
        m.id for m in anthropic.Anthropic().models.list(limit=1000).data
    ]
    providers.append(
        Provider(
            get_response_anthropic, models=_available_anthropic_models, priority=10
        )
    )
except Exception:
    pass

# OpenAI (priority 10 - higher priority than OpenRouter)
if "OPENAI_API_KEY" in os.environ:
    try:
        _available_openai_models = [
            m.id
            for m in openai.OpenAI().models.list().data
            if m.id.startswith("gpt") or m.id.startswith("o")
        ]
        providers.append(
            Provider(
                get_response_factory(openai.AsyncOpenAI()),
                models=_available_openai_models,
                priority=10,
            )
        )
    except Exception:
        pass

# Google GenAI (priority 10 - higher priority than OpenRouter)
if "GEMINI_API_KEY" in os.environ or "GOOGLE_API_KEY" in os.environ:
    providers.append(
        Provider(
            get_response_genai,
            models=["gemini-2.5-pro", "gemini-2.5-flash"],
            priority=10,
        )
    )

# OpenRouter (priority 1000 - lowest priority, fallback for models with "/")
if "OPENROUTER_API_KEY" in os.environ:
    providers.append(
        Provider(
            get_response_factory(
                openai.AsyncOpenAI(
                    api_key=os.environ["OPENROUTER_API_KEY"],
                    base_url="https://openrouter.ai/api/v1",
                )
            ),
            models=[re.compile(r".+/.+")],  # Matches any model with "/" in the name
            priority=1000,
        )
    )

# ---------------------------------------------------------------------------
# vLLM XML provider (if base URL is configured)
# ---------------------------------------------------------------------------
try:
    from .vllm_xml import (
        build_vllm_chat_payload,
        parse_vllm_xml_response,
        get_allowed_model_patterns,
    )

    if os.environ.get("VLLM_BASE_URL"):
        async def get_response_vllm(
            messages: List[ChatMessage],
            tools: Optional[List[ToolDefinition]],
            response_format: Optional[Dict[str, Any]] = None,
            reasoning: Optional[Union[ReasoningConfig, Dict[str, Any]]] = None,
            **kwargs: Any,
        ) -> ChatMessage:
            if response_format is not None:
                raise NotImplementedError(
                    "Structured output is not yet supported via vLLM provider"
                )

            payload = build_vllm_chat_payload(messages, tools, **kwargs)

            base_url = os.environ["VLLM_BASE_URL"].rstrip("/")
            url = f"{base_url}/chat/completions"

            # Use aiohttp if available, otherwise fallback to requests
            data = None
            try:
                import aiohttp  # type: ignore

                timeout = aiohttp.ClientTimeout(total=600)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, json=payload) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
            except Exception:
                import requests

                r = requests.post(url, json=payload, timeout=600)
                r.raise_for_status()
                data = r.json()

            # vLLM OAI-compatible response
            content = data.get("choices", [{}])[0].get("message", {}).get(
                "content", ""
            )
            # Parse XML back to ChatMessage with tool calls if any
            if content and "<tool_use>" in content:
                return parse_vllm_xml_response(content)
            else:
                return ChatMessage(
                    role=MessageRole.assistant,
                    content=[TextBlock(text=content or "")],
                )

        providers.append(
            Provider(
                get_response_vllm,
                models=get_allowed_model_patterns(),
                priority=20,  # prefer above OpenRouter fallback but below direct SDKs
            )
        )
except Exception:
    pass


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------
async def get_response(
    model: str,
    messages: List[ChatMessage],
    tools: Optional[List[ToolDefinition]] = None,
    response_format: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None,
    reasoning: Optional[Union[ReasoningConfig, Dict[str, Any]]] = None,
    **kwargs: Any,
) -> ChatMessage:
    # Convert dict to ReasoningConfig if needed (centralized conversion)
    if reasoning and isinstance(reasoning, dict):
        reasoning = ReasoningConfig(**reasoning)

    # Find all providers that support this model and sort by priority
    supporting_providers = []
    for provider in providers:
        if provider.supports_model(model):
            supporting_providers.append(provider)

    # Sort by priority (lower number = higher priority)
    supporting_providers.sort(key=lambda p: p.priority)

    if not supporting_providers:
        # Collect all available models for error message
        all_models = []
        for provider in providers:
            for model_pattern in provider.models:
                if isinstance(model_pattern, str):
                    all_models.append(model_pattern)
                else:
                    all_models.append(f"<regex: {model_pattern.pattern}>")

        raise ValueError(
            f"Model '{model}' not supported by any provider. Supported models: {all_models}"
        )

    # Use the highest priority provider
    provider = supporting_providers[0]
    response = await provider.get_response(
        model=model,
        messages=messages,
        tools=tools,
        response_format=response_format,
        reasoning=reasoning,
        **kwargs,
    )

    # Handle empty responses
    if len(response.content) == 0:
        messages = messages + [
            ChatMessage(
                role=MessageRole.user,
                content=[TextBlock(text="Please continue.")],
            )
        ]
        response = await provider.get_response(
            model=model,
            messages=messages,
            tools=tools,
            response_format=response_format,
            reasoning=reasoning,
            **kwargs,
        )

    assert len(response.content) > 0, "Response content is empty"
    return response


@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.InternalServerError,
        anthropic.APIConnectionError,
        anthropic.RateLimitError,
        anthropic.APIStatusError,
        genai.errors.ClientError,
        genai.errors.ServerError,
        TypeError,
        AssertionError,
    ),
    max_value=60,
    factor=1.5,
    on_backoff=lambda details: print(f"Retrying... {details['exception']}"),
)
async def get_response_with_backoff(
    model: str,
    messages: List[ChatMessage],
    tools: Optional[List[ToolDefinition]] = None,
    response_format: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None,
    reasoning: Optional[Union[ReasoningConfig, Dict[str, Any]]] = None,
    **kwargs: Any,
) -> ChatMessage:
    """Get a response from an LLM with exponential backoff retry logic for various API errors.

    Args:
        model (str): The name of the model to use
        messages (List[ChatMessage]): The conversation history
        tools (Optional[List[ToolDefinition]]): Optional list of tools/functions the model can use
        response_format (Optional[Dict]): Optional response format specification
        reasoning (Optional[Union[ReasoningConfig, Dict]]): Optional reasoning/thinking configuration
        **kwargs: Additional keyword arguments passed to the underlying API

    Returns:
        ChatResponse: The model's response

    Raises:
        ValueError: If the specified model is not supported by any provider
    """
    # Convert dict to ReasoningConfig if needed
    if reasoning and isinstance(reasoning, dict):
        reasoning = ReasoningConfig(**reasoning)

    return await get_response(
        model,
        messages,
        tools=tools,
        response_format=response_format,
        reasoning=reasoning,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Custom Provider API
# ---------------------------------------------------------------------------
def add_provider(
    get_response_func: Callable[..., Any],
    models: List[Union[str, Pattern]],
    priority: int = 100,
) -> None:
    """Add a custom provider to the router.

    Args:
        get_response_func: Async function that implements the provider's get_response interface
        models: List of model IDs (strings) or regex patterns to match against
        priority: Priority level (lower = higher priority, default 100)
    """
    providers.append(Provider(get_response_func, models, priority))


dcache = DCache(cache_dir=os.path.expanduser("~/.cache/localrouter"))


@dcache(required_kwargs=["cache_seed"])
async def get_response_cached(
    model: str,
    messages: List[ChatMessage],
    tools: Optional[List[ToolDefinition]] = None,
    response_format: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None,
    reasoning: Optional[Union[ReasoningConfig, Dict[str, Any]]] = None,
    cache_seed: Optional[int] = None,
    **kwargs: Any,
) -> ChatMessage:
    """Get a response from an LLM. Cache results on disk if called with same arguments and seed.

    Args:
        model (str): The name of the model to use
        messages (List[ChatMessage]): The conversation history
        tools (Optional[List[ToolDefinition]]): Optional list of tools/functions the model can use
        response_format (Optional[Dict]): Optional response format specification
        reasoning (Optional[Union[ReasoningConfig, Dict]]): Optional reasoning/thinking configuration
        cache_seed (int): if set, use cached responses
        **kwargs: Additional keyword arguments passed to the underlying API

    Returns:
        ChatResponse: The model's response, either from cache or freshly generated

    Raises:
        ValueError: If the specified model is not supported by any provider
    """
    # Convert dict to ReasoningConfig if needed
    if reasoning and isinstance(reasoning, dict):
        reasoning = ReasoningConfig(**reasoning)

    return await get_response(
        model,
        messages,
        tools=tools,
        response_format=response_format,
        reasoning=reasoning,
        **kwargs,
    )


@dcache(required_kwargs=["cache_seed"])
async def get_response_cached_with_backoff(
    model: str,
    messages: List[ChatMessage],
    tools: Optional[List[ToolDefinition]] = None,
    response_format: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None,
    reasoning: Optional[Union[ReasoningConfig, Dict[str, Any]]] = None,
    cache_seed: Optional[int] = None,
    **kwargs: Any,
) -> ChatMessage:
    """Get a response from an LLM. Cache results on disk if called with same arguments and seed. When no cached result is found, use backoff.

    Args:
        model (str): The name of the model to use
        messages (List[ChatMessage]): The conversation history
        tools (Optional[List[ToolDefinition]]): Optional list of tools/functions the model can use
        response_format (Optional[Dict]): Optional response format specification
        reasoning (Optional[Union[ReasoningConfig, Dict]]): Optional reasoning/thinking configuration
        cache_seed (int): if set, use cached responses
        **kwargs: Additional keyword arguments passed to the underlying API

    Returns:
        ChatResponse: The model's response, either from cache or freshly generated

    Raises:
        ValueError: If the specified model is not supported by any provider
    """
    # Convert dict to ReasoningConfig if needed
    if reasoning and isinstance(reasoning, dict):
        reasoning = ReasoningConfig(**reasoning)

    return await get_response_with_backoff(
        model,
        messages,
        tools=tools,
        response_format=response_format,
        reasoning=reasoning,
        **kwargs,
    )
