"""Test that thinking blocks survive serialization/deserialization."""

import json
import pytest
from localrouter.dtypes import ChatMessage, MessageRole, TextBlock, ThinkingBlock, ToolUseBlock, ToolResultBlock


def test_thinking_block_serialization():
    """Test that ThinkingBlock survives model_dump() and reconstruction."""
    original = ChatMessage(
        role=MessageRole.assistant,
        content=[
            ThinkingBlock(thinking="Let me think about this problem..."),
            TextBlock(text="Here is my answer.")
        ]
    )
    
    # Serialize and deserialize
    serialized = original.model_dump()
    deserialized = ChatMessage(**serialized)
    
    # Check that thinking block is preserved
    assert len(deserialized.content) == 2
    assert isinstance(deserialized.content[0], ThinkingBlock)
    assert deserialized.content[0].thinking == "Let me think about this problem..."
    assert isinstance(deserialized.content[1], TextBlock)
    assert deserialized.content[1].text == "Here is my answer."


def test_mixed_content_serialization():
    """Test serialization of messages with multiple content block types."""
    original = ChatMessage(
        role=MessageRole.assistant,
        content=[
            ThinkingBlock(thinking="First, I need to analyze..."),
            TextBlock(text="Based on my analysis:"),
            ToolUseBlock(id="test-id", name="calculator", input={"operation": "add", "a": 1, "b": 2})
        ]
    )
    
    # Serialize and deserialize
    serialized = original.model_dump()
    deserialized = ChatMessage(**serialized)
    
    # Check all types are preserved
    assert len(deserialized.content) == 3
    assert isinstance(deserialized.content[0], ThinkingBlock)
    assert isinstance(deserialized.content[1], TextBlock)
    assert isinstance(deserialized.content[2], ToolUseBlock)
    assert deserialized.content[2].name == "calculator"


def test_tool_result_with_thinking():
    """Test that tool results with text work correctly after thinking."""
    original = ChatMessage(
        role=MessageRole.user,
        content=[
            ToolResultBlock(
                tool_use_id="test-id",
                content=[TextBlock(text="Result: 3")]
            )
        ]
    )
    
    serialized = original.model_dump()
    deserialized = ChatMessage(**serialized)
    
    assert len(deserialized.content) == 1
    assert isinstance(deserialized.content[0], ToolResultBlock)
    assert len(deserialized.content[0].content) == 1
    assert isinstance(deserialized.content[0].content[0], TextBlock)


def test_thinking_with_signature():
    """Test that thinking blocks with signatures are preserved."""
    original = ChatMessage(
        role=MessageRole.assistant,
        content=[
            ThinkingBlock(
                thinking="Secret thoughts...",
                signature="abc123"
            ),
            TextBlock(text="Public response")
        ]
    )
    
    serialized = original.model_dump()
    deserialized = ChatMessage(**serialized)
    
    assert isinstance(deserialized.content[0], ThinkingBlock)
    assert deserialized.content[0].signature == "abc123"
    assert deserialized.content[0].thinking == "Secret thoughts..."


def test_json_round_trip():
    """Test full JSON serialization round trip (simulating file save/load)."""
    original = ChatMessage(
        role=MessageRole.assistant,
        content=[
            ThinkingBlock(thinking="Deep thought..."),
            TextBlock(text="My conclusion")
        ]
    )
    
    # Simulate saving to JSON file
    json_string = json.dumps(original.model_dump())
    
    # Simulate loading from JSON file
    loaded_dict = json.loads(json_string)
    reconstructed = ChatMessage(**loaded_dict)
    
    # Verify everything is preserved
    assert len(reconstructed.content) == 2
    assert isinstance(reconstructed.content[0], ThinkingBlock)
    assert reconstructed.content[0].thinking == "Deep thought..."
    assert isinstance(reconstructed.content[1], TextBlock)
    assert reconstructed.content[1].text == "My conclusion"


if __name__ == "__main__":
    # Run tests
    test_thinking_block_serialization()
    test_mixed_content_serialization()
    test_tool_result_with_thinking()
    test_thinking_with_signature()
    test_json_round_trip()
    print("✅ All serialization tests passed!")