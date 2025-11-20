import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from localrouter import (
    get_response,
    ChatMessage,
    MessageRole,
    TextBlock,
    ThinkingBlock,
    ReasoningConfig,
    ToolDefinition,
)
from localrouter.dtypes import openai_format, genai_format


@pytest.mark.asyncio
async def test_illegal_tool_use_stripped():
    """Test that tools are stripped for unsupported models like o1-preview."""
    
    tools = [ToolDefinition(name="test", description="test", input_schema={})]
    messages = [ChatMessage(role=MessageRole.user, content=[TextBlock(text="Hi")])]
    
    # Test openai_format directly
    formatted = openai_format(messages, tools, model="o1-preview")
    assert "tools" not in formatted
    
    formatted_ok = openai_format(messages, tools, model="gpt-4")
    assert "tools" in formatted_ok

@pytest.mark.asyncio
async def test_gemini_thinking_parsing():
    """Test that Gemini thinking blocks are parsed correctly."""
    
    from localrouter.dtypes import ChatMessage
    
    # Create simple objects instead of MagicMocks
    class Part:
        def __init__(self, thought=False, text=""):
            self.thought = thought
            self.text = text
            self.thought_signature = b"signature_bytes" if thought else None
    
    class Content:
        def __init__(self, parts):
            self.parts = parts
    
    class Candidate:
        def __init__(self, content):
            self.content = content
    
    class Response:
        def __init__(self, candidates):
            self.candidates = candidates
    
    mock_response = Response([
        Candidate(Content([
            Part(thought=True, text="I am thinking..."),
            Part(thought=False, text="Here is the answer.")
        ]))
    ])
    
    msg = ChatMessage.from_genai(mock_response)
    
    assert len(msg.content) == 2
    assert isinstance(msg.content[0], ThinkingBlock)
    assert msg.content[0].thinking == "I am thinking..."
    assert isinstance(msg.content[1], TextBlock)
    assert msg.content[1].text == "Here is the answer."

@pytest.mark.asyncio
async def test_anthropic_thinking_parsing():
    """Test that Anthropic thinking blocks are parsed correctly."""
    
    from localrouter.dtypes import ChatMessage
    
    mock_item_thinking = MagicMock()
    mock_item_thinking.type = "thinking"
    mock_item_thinking.thinking = "I am thinking..."
    mock_item_thinking.signature = "sig123"
    
    mock_item_text = MagicMock()
    mock_item_text.type = "text"
    mock_item_text.text = "Response"
    
    mock_content = [mock_item_thinking, mock_item_text]
    
    msg = ChatMessage.from_anthropic(mock_content)
    
    assert len(msg.content) == 2
    assert isinstance(msg.content[0], ThinkingBlock)
    assert msg.content[0].thinking == "I am thinking..."
    assert msg.content[0].signature == "sig123"