"""Tests for agent-level prompt caching in Thread._apply_prompt_caching."""
import pytest
from localrouter import (
    ChatMessage,
    MessageRole,
    TextBlock,
    ToolDefinition,
    CacheControl,
)

# Import Thread from automator
import sys
sys.path.insert(0, 'packages/automator/src')
from automator.agent import Thread


class TestThreadPromptCaching:
    """Tests for Thread._apply_prompt_caching method."""
    
    def test_caching_not_applied_to_non_claude_models(self):
        """Test that caching is not applied to non-Claude models."""
        thread = Thread(llm={"model": "gpt-4"}, messages=[], tools=[])
        
        messages = [
            ChatMessage(
                role=MessageRole.system,
                content=[TextBlock(text="You are helpful")]
            ),
            ChatMessage(
                role=MessageRole.user,
                content=[TextBlock(text="Hello")]
            )
        ]
        tools = [
            ToolDefinition(
                name="test_tool",
                description="A test tool",
                input_schema={"type": "object", "properties": {}}
            )
        ]
        
        cached_msgs, cached_tools = thread._apply_prompt_caching(messages, tools, "gpt-4")
        
        # No cache_control should be added
        assert cached_msgs[0].content[0].cache_control is None
        assert cached_msgs[1].content[0].cache_control is None
        assert cached_tools[0].cache_control is None
    
    def test_caching_applied_to_claude_models(self):
        """Test that caching is applied to Claude models."""
        thread = Thread(llm={"model": "claude-sonnet-4-5-20250929"}, messages=[], tools=[])
        
        messages = [
            ChatMessage(
                role=MessageRole.system,
                content=[TextBlock(text="You are helpful")]
            ),
            ChatMessage(
                role=MessageRole.user,
                content=[TextBlock(text="Hello")]
            )
        ]
        tools = [
            ToolDefinition(
                name="test_tool",
                description="A test tool",
                input_schema={"type": "object", "properties": {}}
            )
        ]
        
        cached_msgs, cached_tools = thread._apply_prompt_caching(messages, tools, "claude-sonnet-4-5-20250929")
        
        # System message last block should have cache_control
        assert cached_msgs[0].content[-1].cache_control is not None
        assert cached_msgs[0].content[-1].cache_control.type == "ephemeral"
        
        # Last tool should have cache_control
        assert cached_tools[-1].cache_control is not None
        assert cached_tools[-1].cache_control.type == "ephemeral"
        
        # Last message last block should have cache_control
        assert cached_msgs[-1].content[-1].cache_control is not None
        assert cached_msgs[-1].content[-1].cache_control.type == "ephemeral"
    
    def test_caching_does_not_mutate_originals(self):
        """Test that caching creates copies and doesn't mutate original messages."""
        thread = Thread(llm={"model": "claude-sonnet-4-5-20250929"}, messages=[], tools=[])
        
        original_text_block = TextBlock(text="Hello")
        messages = [
            ChatMessage(
                role=MessageRole.user,
                content=[original_text_block]
            )
        ]
        original_tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {}}
        )
        tools = [original_tool]
        
        cached_msgs, cached_tools = thread._apply_prompt_caching(messages, tools, "claude-sonnet-4-5-20250929")
        
        # Original should not be modified
        assert original_text_block.cache_control is None
        assert original_tool.cache_control is None
        
        # Cached versions should have cache_control
        assert cached_msgs[0].content[0].cache_control is not None
        assert cached_tools[0].cache_control is not None
    
    def test_caching_with_multiple_system_blocks(self):
        """Test caching when system message has multiple blocks."""
        thread = Thread(llm={"model": "claude-sonnet-4-5-20250929"}, messages=[], tools=[])
        
        messages = [
            ChatMessage(
                role=MessageRole.system,
                content=[
                    TextBlock(text="You are helpful."),
                    TextBlock(text="<context>Important info</context>"),
                    TextBlock(text="Follow these rules.")
                ]
            ),
            ChatMessage(
                role=MessageRole.user,
                content=[TextBlock(text="Hello")]
            )
        ]
        
        cached_msgs, _ = thread._apply_prompt_caching(messages, [], "claude-sonnet-4-5-20250929")
        
        # Only the LAST block of system message should have cache_control
        assert cached_msgs[0].content[0].cache_control is None
        assert cached_msgs[0].content[1].cache_control is None
        assert cached_msgs[0].content[2].cache_control is not None
        assert cached_msgs[0].content[2].cache_control.type == "ephemeral"
    
    def test_caching_with_multiple_tools(self):
        """Test caching when there are multiple tools."""
        thread = Thread(llm={"model": "claude-sonnet-4-5-20250929"}, messages=[], tools=[])
        
        messages = [
            ChatMessage(
                role=MessageRole.user,
                content=[TextBlock(text="Hello")]
            )
        ]
        tools = [
            ToolDefinition(
                name="tool1",
                description="First tool",
                input_schema={"type": "object", "properties": {}}
            ),
            ToolDefinition(
                name="tool2",
                description="Second tool",
                input_schema={"type": "object", "properties": {}}
            ),
            ToolDefinition(
                name="tool3",
                description="Third tool",
                input_schema={"type": "object", "properties": {}}
            )
        ]
        
        _, cached_tools = thread._apply_prompt_caching(messages, tools, "claude-sonnet-4-5-20250929")
        
        # Only the LAST tool should have cache_control
        assert cached_tools[0].cache_control is None
        assert cached_tools[1].cache_control is None
        assert cached_tools[2].cache_control is not None
        assert cached_tools[2].cache_control.type == "ephemeral"
    
    def test_caching_with_no_system_message(self):
        """Test caching when there's no system message."""
        thread = Thread(llm={"model": "claude-sonnet-4-5-20250929"}, messages=[], tools=[])
        
        messages = [
            ChatMessage(
                role=MessageRole.user,
                content=[TextBlock(text="Hello")]
            ),
            ChatMessage(
                role=MessageRole.assistant,
                content=[TextBlock(text="Hi there!")]
            ),
            ChatMessage(
                role=MessageRole.user,
                content=[TextBlock(text="How are you?")]
            )
        ]
        
        cached_msgs, _ = thread._apply_prompt_caching(messages, [], "claude-sonnet-4-5-20250929")
        
        # Last message's last block should have cache_control
        assert cached_msgs[-1].content[-1].cache_control is not None
        
        # Other messages should not have cache_control
        assert cached_msgs[0].content[0].cache_control is None
        assert cached_msgs[1].content[0].cache_control is None
    
    def test_caching_with_empty_tools(self):
        """Test caching when tools list is empty."""
        thread = Thread(llm={"model": "claude-sonnet-4-5-20250929"}, messages=[], tools=[])
        
        messages = [
            ChatMessage(
                role=MessageRole.user,
                content=[TextBlock(text="Hello")]
            )
        ]
        
        cached_msgs, cached_tools = thread._apply_prompt_caching(messages, [], "claude-sonnet-4-5-20250929")
        
        # Should not error and should still cache messages
        assert cached_tools == []
        assert cached_msgs[-1].content[-1].cache_control is not None