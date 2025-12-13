"""Test script for Anthropic structured outputs support in localrouter."""

import asyncio
from pydantic import BaseModel
from localrouter.llm import get_response
from localrouter.dtypes import ChatMessage, MessageRole, TextBlock, ToolDefinition, ToolUseBlock


# Test 1: JSON outputs with Pydantic model
class ContactInfo(BaseModel):
    name: str
    email: str
    plan_interest: str
    demo_requested: bool


async def test_json_output_pydantic():
    """Test JSON outputs using a Pydantic model."""
    print("\n=== Test 1: JSON outputs with Pydantic model ===")

    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[
                TextBlock(
                    text="Extract the key information from this email: John Smith (john@example.com) is interested in our Enterprise plan and wants to schedule a demo for next Tuesday at 2pm."
                )
            ],
        )
    ]

    response = await get_response(
        model="claude-sonnet-4-5-20250929",
        messages=messages,
        response_format=ContactInfo,
        max_tokens=1024,
    )

    print("Response content:", response.content)
    if hasattr(response, "parsed"):
        print("Parsed output:", response.parsed)
        print("  - Name:", response.parsed.name)
        print("  - Email:", response.parsed.email)
        print("  - Plan interest:", response.parsed.plan_interest)
        print("  - Demo requested:", response.parsed.demo_requested)


# Test 2: JSON outputs with dict schema
async def test_json_output_dict():
    """Test JSON outputs using a dict schema."""
    print("\n=== Test 2: JSON outputs with dict schema ===")

    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[
                TextBlock(
                    text="Extract the key information from this email: John Smith (john@example.com) is interested in our Enterprise plan and wants to schedule a demo for next Tuesday at 2pm."
                )
            ],
        )
    ]

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "email": {"type": "string"},
            "plan_interest": {"type": "string"},
            "demo_requested": {"type": "boolean"}
        },
        "required": ["name", "email", "plan_interest", "demo_requested"],
        "additionalProperties": False
    }

    response = await get_response(
        model="claude-sonnet-4-5-20250929",
        messages=messages,
        response_format=schema,
        max_tokens=1024,
    )

    print("Response content:", response.content)
    # The response should be valid JSON in the text block
    if response.content and len(response.content) > 0:
        text_blocks = [b for b in response.content if isinstance(b, TextBlock)]
        if text_blocks:
            print("JSON output:", text_blocks[0].text)


# Test 3: Strict tool use
async def test_strict_tool_use():
    """Test strict tool use with validated parameters."""
    print("\n=== Test 3: Strict tool use ===")

    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text="What's the weather like in San Francisco?")],
        )
    ]

    tools = [
        ToolDefinition(
            name="get_weather",
            description="Get the current weather in a given location",
            strict=True,
            input_schema={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature"
                    }
                },
                "required": ["location"],
                "additionalProperties": False
            }
        )
    ]

    response = await get_response(
        model="claude-sonnet-4-5-20250929",
        messages=messages,
        tools=tools,
        max_tokens=1024,
    )

    print("Response content:", response.content)
    # Check if any tool use blocks are present
    tool_uses = [b for b in response.content if isinstance(b, ToolUseBlock)]
    if tool_uses:
        for tool_use in tool_uses:
            print(f"Tool used: {tool_use.name}")
            print(f"Tool input: {tool_use.input}")


async def main():
    """Run all tests."""
    try:
        await test_json_output_pydantic()
    except Exception as e:
        print(f"Test 1 failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        await test_json_output_dict()
    except Exception as e:
        print(f"Test 2 failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        await test_strict_tool_use()
    except Exception as e:
        print(f"Test 3 failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
