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
async def test_openai_completion_model():
    """Test that completion models use the completion API."""
    with patch("openai.AsyncOpenAI") as mock_openai_cls:
        mock_oai = AsyncMock()
        mock_openai_cls.return_value = mock_oai
        
        # Mock completions.create
        mock_oai.completions.create.return_value = MagicMock(
            choices=[MagicMock(text="Completion response")]
        )
        
        # We need to ensure the provider is registered with the mock
        # Since we patch AsyncOpenAI, we need to re-register providers or patch get_response_factory's internal usage
        
        # Easier way: Import llm module and patch the client in the provider? 
        # Or just call get_response_factory directly.
        
        from localrouter.llm import get_response_factory
        get_response_openai = get_response_factory(mock_oai)
        
        messages = [ChatMessage(role=MessageRole.user, content=[TextBlock(text="Hello")])]
        
        resp = await get_response_openai(
            messages=messages,
            tools=None,
            model="gpt-3.5-turbo-instruct"
        )
        
        assert resp.content[0].text == "Completion response"
        mock_oai.completions.create.assert_called_once()
        mock_oai.chat.completions.create.assert_not_called()
        
        # Verify prompt construction
        call_kwargs = mock_oai.completions.create.call_args.kwargs
        assert "User: Hello" in call_kwargs["prompt"]

@pytest.mark.asyncio
async def test_illegal_tool_use_stripped():
    """Test that tools are stripped for unsupported models like o1-preview."""
    
    tools = [ToolDefinition(name="test", description="test", input_schema={})]
    messages = [ChatMessage(role=MessageRole.user, content="Hi")]
    
    # Test openai_format directly
    formatted = openai_format(messages, tools, model="o1-preview")
    assert "tools" not in formatted
    
    formatted_ok = openai_format(messages, tools, model="gpt-4")
    assert "tools" in formatted_ok

@pytest.mark.asyncio
async def test_gemini_thinking_parsing():
    """Test that Gemini thinking blocks are parsed correctly."""
    
    from localrouter.dtypes import ChatMessage
    
    mock_part_thought = MagicMock()
    mock_part_thought.thought = True
    mock_part_thought.text = "I am thinking..."
    
    mock_part_text = MagicMock()
    mock_part_text.thought = False
    mock_part_text.text = "Here is the answer."
    
    mock_candidate = MagicMock()
    mock_candidate.content.parts = [mock_part_thought, mock_part_text]
    mock_response = MagicMock()
    mock_response.candidates = [mock_candidate]
    
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