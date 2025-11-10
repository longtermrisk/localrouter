"""LocalRouter: Multi-provider LLM client with unified message format and tool support."""

import re

from .dtypes import (
    ChatMessage,
    MessageRole,
    TextBlock,
    ImageBlock,
    ToolUseBlock,
    ToolResultBlock,
    ThinkingBlock,
    ToolDefinition,
    SubagentToolDefinition,
    PromptTemplate,
    ContentBlock,
    Base64ImageSource,
    ReasoningConfig,
)

from .llm import (
    get_response,
    get_response_with_backoff,
    get_response_cached,
    get_response_cached_with_backoff,
    providers,
    add_provider,
    register_router,
    print_available_models,
)
from .dtypes import (
    openai_format,
    anthropic_format,
    genai_format,
    xml_format,
    messages_to_content_blocks,
)

__version__ = "0.2.5"

__all__ = [
    "ChatMessage",
    "MessageRole",
    "TextBlock",
    "ImageBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "ThinkingBlock",
    "ToolDefinition",
    "SubagentToolDefinition",
    "PromptTemplate",
    "ContentBlock",
    "Base64ImageSource",
    "ReasoningConfig",
    "get_response",
    "get_response_with_backoff",
    "get_response_cached",
    "get_response_cached_with_backoff",
    "providers",
    "add_provider",
    "register_router",
    "print_available_models",
    "re",
    "openai_format",
    "anthropic_format",
    "genai_format",
    "xml_format",
    "messages_to_content_blocks"
]
