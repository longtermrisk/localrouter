import os
import json
import base64
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union, Annotated
from uuid import uuid4
from pydantic import BaseModel, Field, field_validator, Discriminator
import yaml
import logging

from .utils import dict_recursive
from .xml_utils import dump_xml, get_first_element, parse_xml


class ReasoningConfig(BaseModel):
    """Configuration for reasoning/thinking in LLMs.

    Can be specified either as:
    - effort: "minimal"/"low"/"medium"/"high" (OpenAI-style)
    - budget_tokens: int (Anthropic/Gemini-style explicit token count)
    - dynamic: bool (Gemini-style, let model decide)
    """

    effort: Optional[str] = None  # "minimal", "low", "medium", "high"
    budget_tokens: Optional[int] = None
    dynamic: Optional[bool] = None

    @field_validator("effort")
    @classmethod
    def validate_effort(cls, v):
        if v is not None and v not in ["minimal", "low", "medium", "high"]:
            raise ValueError(
                f"Invalid effort level: {v}. Must be one of: minimal, low, medium, high"
            )
        return v

    def to_openai_format(self, model: str) -> Optional[Dict[str, Any]]:
        """Convert to OpenAI reasoning format if applicable."""
        # GPT-5 uses reasoning.effort
        if model and model.startswith("gpt-5"):
            if self.effort:
                return {"effort": self.effort}
            elif self.budget_tokens:
                # Convert token budget to effort level
                if self.budget_tokens <= 2000:
                    return {"effort": "minimal"}
                elif self.budget_tokens <= 8000:
                    return {"effort": "medium"}
                else:
                    return {"effort": "high"}
            elif self.dynamic:
                return {"effort": "medium"}  # Default for dynamic
        # Other OpenAI models don't support reasoning
        return None

    def to_anthropic_format(self, model: str) -> Optional[Dict[str, Any]]:
        """Convert to Anthropic thinking format if applicable."""
        # Support for extended thinking models: Opus 4.1, Opus 4, Sonnet 4, Sonnet 3.7
        if model and (
            "claude-opus-4" in model
            or "claude-sonnet-4" in model
            or "claude-3-7-sonnet" in model
            or "claude-sonnet-3-7" in model
        ):
            if self.budget_tokens:
                return {"type": "enabled", "budget_tokens": self.budget_tokens}
            elif self.effort:
                # Convert effort to token budget
                budget_map = {
                    "minimal": 1024,  # Anthropic minimum
                    "low": 2000,
                    "medium": 8000,
                    "high": 16000,
                }
                return {
                    "type": "enabled",
                    "budget_tokens": budget_map.get(self.effort, 8000),
                }
            elif self.dynamic:
                # Use a reasonable default for dynamic
                return {"type": "enabled", "budget_tokens": 8000}
        return None

    def to_gemini_format(self, model: str) -> Optional[Dict[str, Any]]:
        """Convert to Gemini thinking format if applicable."""
        # Gemini 2.5 models support thinking
        if model and ("gemini-2.5" in model or "gemini-2-5" in model):
            if self.dynamic:
                return {"thinking_budget": -1}
            elif self.budget_tokens:
                return {"thinking_budget": self.budget_tokens}
            elif self.effort:
                # Convert effort to token budget
                budget_map = {
                    "minimal": 1024,  # Anthropic minimum
                    "low": 2000,
                    "medium": 8000,
                    "high": 16000,
                }
                return {"thinking_budget": budget_map.get(self.effort, 8000)}
        return None


class ThinkingBlock(BaseModel):
    """Represents a thinking/reasoning block from the model's response."""

    thinking: str
    type: str = "thinking"
    meta: Dict[str, Any] = Field(default_factory=dict)
    signature: Optional[str] = None  # For encrypted thinking blocks

    def anthropic_format(self) -> Dict[str, Any]:
        # Return proper thinking format for Anthropic
        result = {
            "type": "thinking",
            "thinking": self.thinking,
        }
        if self.signature:
            result["signature"] = self.signature
        return result

    def openai_format(self) -> Dict[str, Any]:
        # OpenAI doesn't expose thinking blocks in responses
        return {"type": "text", "text": ""}

    def xml_format(self) -> Dict[str, Any]:
        return {"type": "text", "text": f"<thinking>{self.thinking}</thinking>"}


class ToolDefinition(BaseModel):
    """Class for structured JSON response format that can be used as a tool for judges

    Arguments:
        definition: JSON schema for the structured response format in anthropic format
    """

    name: str
    description: str
    input_schema: Dict[str, Any]

    @property
    def anthropic_format(self):
        """Return the definition in Anthropic format"""
        return self.model_dump()

    @property
    def openai_format(self):
        """Convert the definition to OpenAI format"""
        oai_definition = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }
        return oai_definition

    @property
    def xml_format(self):
        """Convert the definition to XML format"""
        return dump_xml(tool=self.model_dump())


class SubagentToolDefinition(ToolDefinition):
    input_schema: Dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The message to the agent. ",
                },
            },
        }
    )


class MessageRole(str, Enum):
    user = "user"
    system = "system"
    assistant = "assistant"


class TextBlock(BaseModel):
    text: str
    type: str = "text"
    meta: Dict[str, Any] = Field(default_factory=dict)

    def anthropic_format(self) -> Dict[str, Any]:
        return {
            "type": "text",
            "text": self.text,
        }

    def openai_format(self) -> Dict[str, Any]:
        return {
            "type": "text",
            "text": self.text,
        }

    def xml_format(self) -> str:
        return {
            "type": "text",
            "text": self.text,
        }


class Base64ImageSource(BaseModel):
    data: str
    media_type: str
    type: str = "base64"


class ImageBlock(BaseModel):
    source: Base64ImageSource
    type: str = "image"
    meta: Dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def from_base64(
        data: str, media_type: str = "image/png", meta=None
    ) -> "ImageBlock":
        return ImageBlock(
            source=Base64ImageSource(data=data, media_type=media_type, meta=meta)
        )

    @staticmethod
    def from_file(
        file_path: str, media_type: str = "image/png", meta=None
    ) -> "ImageBlock":
        with open(file_path, "rb") as f:
            data = f.read()
        base64_data = base64.b64encode(data).decode("utf-8")
        return ImageBlock(
            source=Base64ImageSource(data=base64_data, media_type=media_type, meta=meta)
        )

    def anthropic_format(self) -> Dict[str, Any]:
        return {"type": "image", "source": self.source.model_dump()}

    def openai_format(self) -> Dict[str, Any]:
        data_url = f"data:{self.source.media_type};base64,{self.source.data}"
        return {"type": "image_url", "image_url": {"url": data_url}}

    def xml_format(self) -> str:
        return {"type": "text", "text": "(image)"}


class ToolUseBlock(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    input: Optional[Dict[str, Any]] = None
    name: str
    type: str = "tool_use"
    meta: Dict[str, Any] = Field(default_factory=dict)

    def anthropic_format(self) -> Dict[str, Any]:
        return {
            "type": "tool_use",
            "id": self.id,
            "name": self.name,
            "input": self.input,
        }

    def openai_format(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.input or {}),
            },
        }

    def xml_format(self) -> str:
        return dict(
            type="text", text=dump_xml(tool_use=dict(name=self.name, input=self.input))
        )


class ToolResultBlock(BaseModel):
    tool_use_id: str
    content: List[Union[TextBlock, ImageBlock]]
    type: str = "tool_result"
    meta: Dict[str, Any] = Field(default_factory=dict)

    def anthropic_format(self) -> Dict[str, Any]:
        return {
            "type": "tool_result",
            "tool_use_id": self.tool_use_id,
            "content": [block.anthropic_format() for block in self.content],
        }

    def openai_format(self) -> Dict[str, Any]:
        # OpenAI tool results only support text content, so we need to handle images differently
        text_parts = [b.text for b in self.content if isinstance(b, TextBlock)]

        # For images in tool results, we'll add a reference text since OpenAI doesn't support images in tool results
        image_count = len([b for b in self.content if isinstance(b, ImageBlock)])
        if image_count > 0:
            text_parts.append(
                f"[Tool returned {image_count} image(s) - images will be included in next user message]"
            )

        content_str = "\n".join(text_parts) if text_parts else ""
        return {
            "role": "tool",
            "tool_call_id": self.tool_use_id,
            "content": content_str,
        }

    def xml_format(self) -> str:
        output = "\n".join(
            [
                (
                    block.xml_format()["text"]
                    if hasattr(block, "xml_format")
                    else f"Skipped: {block.type}"
                )
                for block in self.content
            ]
        )
        return dict(type="text", text=dump_xml(tool_result=dict(output=output)))


ContentBlock = Union[
    TextBlock, ImageBlock, ToolUseBlock, ToolResultBlock, ThinkingBlock
]


class ChatMessage(BaseModel):
    role: MessageRole
    content: List[ContentBlock]
    meta: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        # Allow additional attributes for structured output
        extra = "allow"

    @field_validator("role", mode="before")
    @classmethod
    def convert_role_to_enum(cls, v):
        if isinstance(v, str):
            try:
                return MessageRole[v]
            except KeyError:
                raise ValueError(f"Invalid role: {v}")
        return v

    @field_validator("content", mode="before")
    @classmethod
    def convert_content_blocks(cls, v):
        """Convert raw dicts to proper ContentBlock types based on 'type' field"""
        if not isinstance(v, list):
            return v
        
        result = []
        for item in v:
            # If it's already a proper block type, keep it
            if isinstance(item, (TextBlock, ImageBlock, ToolUseBlock, ToolResultBlock, ThinkingBlock)):
                result.append(item)
                continue
            
            # If it's a dict, reconstruct the proper type
            if isinstance(item, dict):
                block_type = item.get('type', 'text')
                if block_type == 'thinking':
                    result.append(ThinkingBlock(**item))
                elif block_type == 'text':
                    result.append(TextBlock(**item))
                elif block_type == 'image':
                    result.append(ImageBlock(**item))
                elif block_type == 'tool_use':
                    result.append(ToolUseBlock(**item))
                elif block_type == 'tool_result':
                    result.append(ToolResultBlock(**item))
                else:
                    # Fallback to text block for unknown types
                    result.append(TextBlock(text=str(item), type='text'))
            else:
                # If it's something else, convert to text
                result.append(TextBlock(text=str(item), type='text'))
        
        return result

    def anthropic_format(self):  # -> dict[str, Any]
        if isinstance(self.content, str):
            return {"role": self.role.value, "content": self.content}

        # For Anthropic, include thinking blocks when they exist (needed for multi-turn conversations)
        content = [c.anthropic_format() for c in self.content]
        return {"role": self.role.value, "content": content}

    def openai_format(self) -> Dict[str, Any]:
        """Convert message to OpenAI format, handling tool calls and results properly"""
        if isinstance(self.content, str):
            return {"role": self.role.value, "content": self.content}

        # Handle tool results in user messages - they need to be separate messages
        tool_results = []
        image_parts = []
        non_tool_content = []

        for block in self.content:
            if isinstance(block, ThinkingBlock):
                continue
            if isinstance(block, ToolResultBlock):
                tool_results.append(block.openai_format())
                # Extract images from tool results for separate user message
                for content_block in block.content:
                    if isinstance(content_block, ImageBlock):
                        image_parts.append(content_block)
            else:
                non_tool_content.append(block)

        messages = []

        # Add tool result messages first
        messages.extend(tool_results)

        # Add image parts as separate user message if any
        if image_parts:
            messages.append(
                {
                    "role": "user",
                    "content": [block.openai_format() for block in image_parts],
                }
            )

        # Handle main message content
        if non_tool_content or (not tool_results and not image_parts):
            role = self.role.value
            tool_calls = []
            text_and_images = []

            for block in non_tool_content:
                if isinstance(block, ToolUseBlock):
                    tool_calls.append(block.openai_format())
                else:
                    text_and_images.append(block)

            oai_msg = {"role": role}

            if text_and_images:
                # Convert to OpenAI multimodal format
                oai_parts = []
                for block in text_and_images:
                    if isinstance(block, TextBlock):
                        oai_parts.append({"type": "text", "text": block.text})
                    elif isinstance(block, ImageBlock):
                        oai_parts.append(block.openai_format())

                # If only one text block, return as string
                if len(oai_parts) == 1 and oai_parts[0]["type"] == "text":
                    oai_msg["content"] = oai_parts[0]["text"]
                else:
                    oai_msg["content"] = oai_parts
            else:
                oai_msg["content"] = None

            if tool_calls:
                oai_msg["tool_calls"] = tool_calls

            if oai_msg["content"] is not None or tool_calls:
                messages.append(oai_msg)

        return (
            messages
            if len(messages) > 1
            else (messages[0] if messages else {"role": self.role.value, "content": ""})
        )

    def xml_format(self, keep_images: bool = True) -> str:
        """Convert message content to XML format"""
        if isinstance(self.content, str):
            return self.content

        content_parts = []
        for block in self.content:
            if isinstance(block, ImageBlock) and not keep_images:
                continue
            if hasattr(block, "xml_format"):
                content_parts.append(block.xml_format())
            else:
                content_parts.append(str(block))

        return dict(type="text", content=content_parts)

    @staticmethod
    def from_openai(message, **kwargs) -> "ChatMessage":
        """Convert OpenAI message response to ChatMessage"""
        blocks = []

        # Handle content
        if message.content:
            if isinstance(message.content, str):
                if message.content.strip():
                    blocks.append(TextBlock(text=message.content))
            else:
                # Multi-modal response
                for part in message.content:
                    if part["type"] == "text":
                        blocks.append(TextBlock(text=part["text"]))
                    elif part["type"] == "image_url":
                        url = part["image_url"]["url"]
                        if url.startswith("data:"):
                            header, b64_data = url.split(",", 1)
                            media_type = header[len("data:") : header.index(";")]
                            blocks.append(
                                ImageBlock.from_base64(
                                    data=b64_data, media_type=media_type
                                )
                            )

        # Handle tool calls
        if hasattr(message, "tool_calls") and message.tool_calls:
            for call in message.tool_calls:
                import json

                blocks.append(
                    ToolUseBlock(
                        id=call.id,
                        name=call.function.name,
                        input=json.loads(call.function.arguments),
                    )
                )

        return ChatMessage(role=MessageRole.assistant, content=blocks)

    @staticmethod
    def from_anthropic(response_content, **kwargs) -> "ChatMessage":
        """Convert Anthropic response content to ChatMessage"""
        blocks = []
        for item in response_content:
            if item.type == "text":
                blocks.append(TextBlock(text=item.text))
            elif item.type == "tool_use":
                blocks.append(
                    ToolUseBlock(id=item.id, name=item.name, input=item.input)
                )
            elif item.type == "thinking":
                # Properly handle thinking blocks with signature
                signature = getattr(item, "signature", None)
                blocks.append(
                    ThinkingBlock(
                        thinking=item.thinking,
                        signature=signature,
                        meta={"display_html": f"<i>{item.thinking}</i>"},
                    )
                )
            elif item.type == "redacted_thinking":
                # Handle redacted thinking blocks
                data = getattr(item, "data", "")
                blocks.append(
                    ThinkingBlock(
                        thinking="[REDACTED]",
                        meta={"redacted_data": data, "is_redacted": True},
                    )
                )
            else:
                raise ValueError(f"Unknown block type: {item.type}")

        return ChatMessage(role=MessageRole.assistant, content=blocks)

    @staticmethod
    def from_genai(response, **kwargs) -> "ChatMessage":
        """Convert Google GenAI response to ChatMessage"""
        blocks = []

        # Check for thought summaries in candidates
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                for part in candidate.content.parts:
                    if hasattr(part, "thought") and part.thought:
                        # Use ThinkingBlock for thoughts
                        blocks.append(ThinkingBlock(thinking=part.text))
                    elif hasattr(part, "text") and part.text:
                        blocks.append(TextBlock(text=part.text))

        # Fallback to simple text if no parts found
        if not blocks and response.text:
            blocks.append(TextBlock(text=response.text))

        # Handle function calls
        if hasattr(response, "function_calls") and response.function_calls:
            for func_call in response.function_calls:
                func_id = getattr(func_call, "id", None) or str(uuid4())
                func_args = getattr(func_call, "args", {})
                blocks.append(
                    ToolUseBlock(id=func_id, name=func_call.name, input=func_args)
                )

        return ChatMessage(role=MessageRole.assistant, content=blocks)

    @staticmethod
    def from_xml_format(message_dict: Dict[str, Any]) -> "ChatMessage":
        """Parse XML format back to ChatMessage"""
        content_blocks = []
        role = MessageRole[message_dict["role"]]
        content = message_dict["content"]

        if isinstance(content, str):
            # Parse XML content for tool use blocks
            while content:
                before, tool_use_text, content = get_first_element(content, "tool_use")

                if before.strip():
                    content_blocks.append(TextBlock(text=before))

                if tool_use_text:
                    try:
                        tool_call = parse_xml(
                            tool_use_text,
                            {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "input": {
                                        "type": "string"
                                    },  # Parse as string first, then convert
                                    "id": {
                                        "type": "string",
                                        "default": f"toolu-{uuid4().hex[:8]}",
                                    },
                                },
                            },
                        )
                        # Try to parse input as JSON if it's a string
                        if isinstance(tool_call.get("input"), str):
                            try:
                                import json

                                tool_call["input"] = json.loads(tool_call["input"])
                            except:
                                # If JSON parsing fails, try XML parsing
                                input_text = tool_call["input"]
                                input_dict = {}

                                # Extract all XML elements from input
                                remaining_text = input_text
                                while remaining_text:
                                    found_element = False
                                    # Try common parameter names
                                    for param in [
                                        "expression",
                                        "query",
                                        "param",
                                        "value",
                                        "text",
                                        "message",
                                        "content",
                                        "data",
                                    ]:
                                        before, element_text, after = get_first_element(
                                            remaining_text, param
                                        )
                                        if element_text is not None:
                                            input_dict[param] = element_text.strip()
                                            remaining_text = before + after
                                            found_element = True
                                            break

                                    if not found_element:
                                        break

                                if input_dict:
                                    tool_call["input"] = input_dict
                                else:
                                    # If no XML elements found, keep as string
                                    tool_call["input"] = {"content": input_text}
                        content_blocks.append(
                            ToolUseBlock(
                                id=tool_call.get("id", f"toolu-{uuid4().hex[:8]}"),
                                input=tool_call.get("input"),
                                name=tool_call["name"],
                            )
                        )
                    except Exception as e:
                        # If parsing fails, treat as text
                        content_blocks.append(
                            TextBlock(text=f"<tool_use>{tool_use_text}</tool_use>")
                        )

                if not tool_use_text and not before:
                    break

            # If there's remaining content, add as text block
            if content.strip():
                content_blocks.append(TextBlock(text=content))

        else:
            # Handle list of content blocks
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        content_blocks.append(TextBlock(**block))
                    elif block.get("type") == "tool_use":
                        content_blocks.append(ToolUseBlock(**block))
                    elif block.get("type") == "tool_result":
                        tool_content = []
                        for content_item in block.get("content", []):
                            if isinstance(content_item, str):
                                tool_content.append(TextBlock(text=content_item))
                            elif isinstance(content_item, dict):
                                if content_item.get("type") == "text":
                                    tool_content.append(TextBlock(**content_item))
                                elif content_item.get("type") == "image":
                                    tool_content.append(ImageBlock(**content_item))
                        content_blocks.append(
                            ToolResultBlock(
                                tool_use_id=block["tool_use_id"], content=tool_content
                            )
                        )
                    elif block.get("type") == "image":
                        content_blocks.append(ImageBlock(**block))

        return ChatMessage(
            role=role, content=content_blocks, meta=message_dict.get("meta", {})
        )


class PromptTemplate(BaseModel):
    messages: Sequence[ChatMessage]

    @staticmethod
    def from_yaml(file: str) -> "PromptTemplate":
        if not os.path.exists(file) and os.path.exists(
            os.path.join(os.path.expanduser("~/.automator/prompts"), file)
        ):
            file = os.path.join(os.path.expanduser("~/.automator/prompts"), file)
        if not os.path.exists(file):
            raise FileNotFoundError(f"Prompt template file not found: {file}")
        with open(file) as f:
            messages = yaml.load(f, Loader=yaml.FullLoader)["messages"]
        for message in messages:
            if isinstance(message["content"], str):
                message["content"] = [TextBlock(text=message["content"])]
        return PromptTemplate(messages=messages)

    def apply(self, params: Dict[str, Any]) -> List[ChatMessage]:
        @dict_recursive()
        def apply_params(template, params):
            if template is None:
                return None
            for key, value in params.items():
                if value is None:
                    continue
                template = template.replace(f"${key}", value)
            return template

        # ChatMessage -> json -> apply_params -> ChatMessage
        messages = [msg.model_dump() for msg in self.messages]
        messages = apply_params(messages, params)
        return [ChatMessage(**msg) for msg in messages]


# ---------------------------------------------------------------------------
# Format conversion functions for different providers
# ---------------------------------------------------------------------------


def anthropic_format(messages, tools, reasoning=None, **kwargs) -> Dict[str, Any]:
    """Convert our internal chat representation into a payload suitable for Anthropic."""
    system_message = None
    chat_messages = []

    if len(messages) > 0 and messages[0].role == MessageRole.system:
        system_message = "\n".join(
            [
                block.text
                for block in messages[0].content
                if isinstance(block, TextBlock)
            ]
        )
        chat_messages = [msg.anthropic_format() for msg in messages[1:]]
    else:
        chat_messages = [msg.anthropic_format() for msg in messages]

    kwargs["messages"] = chat_messages
    if tools:
        kwargs["tools"] = [tool.anthropic_format for tool in tools]
    if system_message:
        kwargs["system"] = system_message

    if not "max_tokens" in kwargs:
        kwargs["max_tokens"] = 32000

    # Handle reasoning/thinking configuration
    if reasoning and isinstance(reasoning, ReasoningConfig):
        model = kwargs.get("model", "")
        thinking_config = reasoning.to_anthropic_format(model)
        if thinking_config:
            kwargs["thinking"] = thinking_config
        if kwargs.get("temperature", 1) != 1:
            logging.warning(
                f"Anthropic models require temperature to be 1 when thinking is enabled. Overwriting given temperature."
            )
            kwargs["temperature"] = 1

    return kwargs


def openai_format(messages, tools, reasoning=None, **kwargs) -> Dict[str, Any]:
    """Convert our internal chat representation into a payload suitable for OpenAI."""
    oai_messages = []

    for msg in messages:
        formatted = msg.openai_format()
        if isinstance(formatted, list):
            oai_messages.extend(formatted)
        else:
            oai_messages.append(formatted)

    kwargs["messages"] = oai_messages

    model = kwargs.get("model", "")

    # Strip tools for models that don't support them (o1-preview, o1-mini)
    if model in ["o1-preview", "o1-mini"]:
        tools = None

    if tools:
        kwargs["tools"] = [t.openai_format for t in tools]

    if "max_tokens" in kwargs:
        kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")

    if model.startswith("o") or model.startswith("gpt-5"):
        kwargs.pop("temperature", None)

    # Handle reasoning configuration for GPT-5
    if reasoning and isinstance(reasoning, ReasoningConfig):
        reasoning_config = reasoning.to_openai_format(model)
        if reasoning_config:
            kwargs["reasoning_effort"] = reasoning_config["effort"]

    return kwargs


def genai_format(messages, tools, reasoning=None, **kwargs) -> Dict[str, Any]:
    """Convert our internal chat representation into a payload suitable for Google GenAI."""
    try:
        from google.genai import types as genai_types
    except ImportError:
        raise ImportError(
            "google-genai package is required for Google GenAI support. Install with: pip install google-genai"
        )

    # Convert messages to GenAI format
    genai_contents = []
    system_instruction = None

    for msg in messages:
        if msg.role == MessageRole.system:
            # System messages become system_instruction
            system_instruction = "\n".join(
                [block.text for block in msg.content if isinstance(block, TextBlock)]
            )
            continue

        # Convert role
        if msg.role == MessageRole.user:
            role = "user"
        elif msg.role == MessageRole.assistant:
            role = "model"
        else:
            role = "user"  # fallback

        # Convert content blocks
        parts = []
        for block in msg.content:
            if isinstance(block, ThinkingBlock):
                continue
            if isinstance(block, TextBlock):
                parts.append(genai_types.Part.from_text(text=block.text))
            elif isinstance(block, ImageBlock):
                # Convert base64 image to GenAI format using from_bytes
                import base64

                image_data = base64.b64decode(block.source.data)
                parts.append(
                    genai_types.Part.from_bytes(
                        data=image_data, mime_type=block.source.media_type
                    )
                )
            elif isinstance(block, ToolUseBlock):
                parts.append(
                    genai_types.Part.from_function_call(
                        name=block.name, args=block.input or {}
                    )
                )
            elif isinstance(block, ToolResultBlock):
                # For tool results, handle both text and images
                for content_block in block.content:
                    if isinstance(content_block, TextBlock):
                        parts.append(
                            genai_types.Part.from_text(text=content_block.text)
                        )
                    elif isinstance(content_block, ImageBlock):
                        # Handle images in tool results
                        import base64

                        image_data = base64.b64decode(content_block.source.data)
                        parts.append(
                            genai_types.Part.from_bytes(
                                data=image_data,
                                mime_type=content_block.source.media_type,
                            )
                        )

        if parts:
            if role == "user":
                genai_contents.append(genai_types.UserContent(parts=parts))
            else:
                genai_contents.append(genai_types.ModelContent(parts=parts))

    # Build the request
    request = {"contents": genai_contents, **kwargs}

    if system_instruction:
        request["system_instruction"] = system_instruction

    if tools:
        # Convert tools to GenAI format
        genai_tools = []
        for tool in tools:
            func_decl = genai_types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters_json_schema=tool.input_schema,
            )
            genai_tools.append(genai_types.Tool(function_declarations=[func_decl]))
        request["tools"] = genai_tools

    # Handle reasoning/thinking configuration
    if reasoning and isinstance(reasoning, ReasoningConfig):
        model = kwargs.get("model", "gemini-2.5-pro")
        thinking_config = reasoning.to_gemini_format(model)
        if thinking_config:
            request["thinking_budget"] = thinking_config["thinking_budget"]

    return request


def xml_format(messages, tools, keep_images: bool = True, **kwargs) -> Dict[str, Any]:
    """Convert our internal chat representation into XML format."""
    if tools:
        # Add system message with tool definitions if tools are provided
        system_msg = ""
        if messages and messages[0].role == MessageRole.system:
            system_msg = messages[0].xml_format(keep_images=keep_images)
            messages = messages[1:]

        definitions = "\n".join([tool.xml_format for tool in tools])

        system_msg += f"\n\nUse one of the available tools to choose an action.\n{definitions}\n\n"
        system_msg += "Respond in valid XML in order to call a tool, for example:\n\n<tool_use>\n  <name>tool_name</name>\n  <input>\n    <key>value</key>\n  </input>\n</tool_use>\n\n"

        system_message = ChatMessage(
            role=MessageRole.system, content=[TextBlock(text=system_msg)]
        )
        messages = [system_message] + messages

    return {
        "messages": [
            {
                "role": (
                    msg.role.value
                    if msg.role.value in ["system", "user", "assistant"]
                    else "user"
                ),
                "content": msg.xml_format(keep_images=keep_images),
            }
            for msg in messages
        ]
    }


def messages_to_content_blocks(messages, role=MessageRole.user):
    """Turn a list of messages into a content blocks with xml syntax for tool use and results
    Useful for asking a model something about an entire conversation, where a full conversation
    should appear in a single user message.
    """
    contents = []
    for message in messages:
        contents.append(TextBlock(text=f"<{message.role}>"))
        for block in message.content:
            if (
                block.type == "text"
                and message.role == "assistant"
                and any([isinstance(block, ToolUseBlock)])
            ):
                continue
            elif isinstance(block, ImageBlock):
                contents.append(block)
            else:
                contents.append(TextBlock(**block.xml_format()))
        contents.append(TextBlock(text=f"</{message.role}>"))
    return contents