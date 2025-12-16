"""Tests for Anthropic prompt caching support."""
import pytest
from localrouter import (
    ChatMessage,
    MessageRole,
    TextBlock,
    ImageBlock,
    ToolUseBlock,
    ToolResultBlock,
    ToolDefinition,
    CacheControl,
    anthropic_format,
)


class TestCacheControlModel:
    """Tests for the CacheControl model."""
    
    def test_cache_control_default(self):
        """Test that CacheControl defaults to ephemeral type."""
        cc = CacheControl()
        assert cc.type == "ephemeral"
    
    def test_cache_control_model_dump(self):
        """Test CacheControl serialization."""
        cc = CacheControl(type="ephemeral")
        assert cc.model_dump() == {"type": "ephemeral"}


class TestContentBlockCaching:
    """Tests for cache_control on content blocks."""
    
    def test_text_block_with_cache_control(self):
        """Test TextBlock with cache_control in anthropic_format."""
        block = TextBlock(
            text="Hello world",
            cache_control=CacheControl(type="ephemeral")
        )
        result = block.anthropic_format()
        assert result == {
            "type": "text",
            "text": "Hello world",
            "cache_control": {"type": "ephemeral"}
        }
    
    def test_text_block_without_cache_control(self):
        """Test TextBlock without cache_control."""
        block = TextBlock(text="Hello world")
        result = block.anthropic_format()
        assert result == {"type": "text", "text": "Hello world"}
        assert "cache_control" not in result
    
    def test_tool_use_block_with_cache_control(self):
        """Test ToolUseBlock with cache_control."""
        block = ToolUseBlock(
            id="test-123",
            name="test_tool",
            input={"query": "test"},
            cache_control=CacheControl(type="ephemeral")
        )
        result = block.anthropic_format()
        assert result["cache_control"] == {"type": "ephemeral"}
    
    def test_tool_result_block_with_cache_control(self):
        """Test ToolResultBlock with cache_control."""
        block = ToolResultBlock(
            tool_use_id="test-123",
            content=[TextBlock(text="Result")],
            cache_control=CacheControl(type="ephemeral")
        )
        result = block.anthropic_format()
        assert result["cache_control"] == {"type": "ephemeral"}


class TestToolDefinitionCaching:
    """Tests for cache_control on tool definitions."""
    
    def test_tool_definition_with_cache_control(self):
        """Test ToolDefinition with cache_control in anthropic_format."""
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {}},
            cache_control=CacheControl(type="ephemeral")
        )
        result = tool.anthropic_format
        assert result["cache_control"] == {"type": "ephemeral"}
    
    def test_tool_definition_without_cache_control(self):
        """Test ToolDefinition without cache_control."""
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {}}
        )
        result = tool.anthropic_format
        assert "cache_control" not in result


class TestAnthropicFormatCaching:
    """Tests for anthropic_format function with caching."""
    
    def test_system_message_with_cache_control_uses_structured_format(self):
        """Test that system messages with cache_control use structured format."""
        system_msg = ChatMessage(
            role=MessageRole.system,
            content=[
                TextBlock(text="You are a helpful assistant."),
                TextBlock(
                    text="<context>Important context here</context>",
                    cache_control=CacheControl(type="ephemeral")
                )
            ]
        )
        user_msg = ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text="Hello")]
        )
        
        result = anthropic_format([system_msg, user_msg], tools=None)
        
        # Should use structured system format (list of dicts)
        assert isinstance(result["system"], list)
        assert len(result["system"]) == 2
        
        # First block should not have cache_control
        assert result["system"][0]["type"] == "text"
        assert "cache_control" not in result["system"][0]
        
        # Second block should have cache_control
        assert result["system"][1]["type"] == "text"
        assert result["system"][1]["cache_control"] == {"type": "ephemeral"}
    
    def test_system_message_without_cache_control_uses_string_format(self):
        """Test that system messages without cache_control use string format."""
        system_msg = ChatMessage(
            role=MessageRole.system,
            content=[TextBlock(text="You are a helpful assistant.")]
        )
        user_msg = ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text="Hello")]
        )
        
        result = anthropic_format([system_msg, user_msg], tools=None)
        
        # Should use simple string format
        assert isinstance(result["system"], str)
        assert result["system"] == "You are a helpful assistant."
    
    def test_tools_with_cache_control(self):
        """Test that tools with cache_control are formatted correctly."""
        tools = [
            ToolDefinition(
                name="tool1",
                description="First tool",
                input_schema={"type": "object", "properties": {}}
            ),
            ToolDefinition(
                name="tool2",
                description="Second tool",
                input_schema={"type": "object", "properties": {}},
                cache_control=CacheControl(type="ephemeral")
            )
        ]
        
        messages = [
            ChatMessage(
                role=MessageRole.user,
                content=[TextBlock(text="Hello")]
            )
        ]
        
        result = anthropic_format(messages, tools=tools)
        
        # First tool should not have cache_control
        assert "cache_control" not in result["tools"][0]
        
        # Last tool should have cache_control
        assert result["tools"][1]["cache_control"] == {"type": "ephemeral"}
    
    def test_message_content_with_cache_control(self):
        """Test that message content with cache_control is preserved."""
        messages = [
            ChatMessage(
                role=MessageRole.user,
                content=[
                    TextBlock(text="First part"),
                    TextBlock(
                        text="Cached part",
                        cache_control=CacheControl(type="ephemeral")
                    )
                ]
            )
        ]
        
        result = anthropic_format(messages, tools=None)
        
        # Message content should have cache_control on last block
        msg_content = result["messages"][0]["content"]
        assert len(msg_content) == 2
        assert "cache_control" not in msg_content[0]
        assert msg_content[1]["cache_control"] == {"type": "ephemeral"}


class TestCacheControlSerialization:
    """Tests for cache_control serialization and deserialization."""
    
    def test_text_block_roundtrip(self):
        """Test that TextBlock with cache_control survives model_dump/reconstruct."""
        original = TextBlock(
            text="Hello",
            cache_control=CacheControl(type="ephemeral")
        )
        
        dumped = original.model_dump()
        reconstructed = TextBlock(**dumped)
        
        assert reconstructed.cache_control is not None
        assert reconstructed.cache_control.type == "ephemeral"
    
    def test_chat_message_roundtrip(self):
        """Test that ChatMessage with cached blocks survives roundtrip."""
        original = ChatMessage(
            role=MessageRole.user,
            content=[
                TextBlock(
                    text="Cached content",
                    cache_control=CacheControl(type="ephemeral")
                )
            ]
        )
        
        dumped = original.model_dump()
        reconstructed = ChatMessage(**dumped)
        
        assert len(reconstructed.content) == 1
        assert reconstructed.content[0].cache_control is not None
        assert reconstructed.content[0].cache_control.type == "ephemeral"