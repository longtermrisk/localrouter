"""LocalRouter: Multi-provider LLM client with unified message format and tool support."""

from .dtypes import (
    ChatMessage,
    MessageRole,
    TextBlock,
    ImageBlock,
    ToolUseBlock,
    ToolResultBlock,
    ToolDefinition,
    SubagentToolDefinition,
    PromptTemplate,
    ContentBlock,
    Base64ImageSource,
)

from .llm import (
    get_response,
    get_response_with_backoff,
    get_response_cached,
    get_response_cached_with_backoff,
    providers,
)
from .dtypes import (
    openai_format,
    anthropic_format,
    genai_format,
    xml_format,
    messages_to_content_blocks,
)

__version__ = "0.1.0"

__all__ = [
    "ChatMessage",
    "MessageRole",
    "TextBlock",
    "ImageBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "ToolDefinition",
    "SubagentToolDefinition",
    "PromptTemplate",
    "ContentBlock",
    "Base64ImageSource",
    "get_response",
    "get_response_with_backoff",
    "get_response_cached",
    "get_response_cached_with_backoff",
    "providers",
    "openai_format",
    "anthropic_format",
    "genai_format",
    "xml_format",
    "messages_to_content_blocks",
]
