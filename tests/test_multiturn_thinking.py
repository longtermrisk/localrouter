"""
Test multi-turn conversations with extended thinking and tool use.
This test specifically addresses the original error about thinking blocks in tool use.
"""

import asyncio
import os
from localrouter.dtypes import (
    ChatMessage,
    MessageRole,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ToolResultBlock,
    ToolDefinition,
    ReasoningConfig,
)
from localrouter.llm import get_response


# Simple calculator tool
calculator_tool = ToolDefinition(
    name="calculator",
    description="Perform mathematical calculations",
    input_schema={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate",
            }
        },
        "required": ["expression"],
    },
)


async def test_anthropic_multiturn_with_tools():
    """Test the original error scenario: multi-turn with extended thinking and tools"""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("No ANTHROPIC_API_KEY found, skipping test")
        return

    print("=== Testing Anthropic Multi-turn with Extended Thinking and Tools ===")

    # Initial conversation
    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[
                TextBlock(
                    text="I need to calculate 25 * 4. Can you use the calculator tool? Think about this step by step."
                )
            ],
        )
    ]

    print("Step 1: Making initial request with thinking and tools...")
    response1 = await get_response(
        model="claude-3-7-sonnet-20250219",
        messages=messages,
        tools=[calculator_tool],
        reasoning=ReasoningConfig(budget_tokens=1024),
        max_tokens=3000,
    )

    thinking_blocks = [b for b in response1.content if isinstance(b, ThinkingBlock)]
    tool_use_blocks = [b for b in response1.content if isinstance(b, ToolUseBlock)]
    text_blocks = [b for b in response1.content if isinstance(b, TextBlock)]

    print(
        f"Response 1: {len(thinking_blocks)} thinking, {len(tool_use_blocks)} tool_use, {len(text_blocks)} text"
    )

    if thinking_blocks:
        print(f"Thinking content: {thinking_blocks[0].thinking[:100]}...")

    if tool_use_blocks:
        print(f"Tool call: {tool_use_blocks[0].name} with {tool_use_blocks[0].input}")

        # Simulate tool execution
        expression = tool_use_blocks[0].input.get("expression", "")
        try:
            result = eval(expression)  # Simple eval for testing
            tool_result = f"The result is: {result}"
        except:
            tool_result = "Error in calculation"

        print(f"Tool result: {tool_result}")

        # Add assistant response and tool result to conversation
        messages.append(response1)
        messages.append(
            ChatMessage(
                role=MessageRole.user,
                content=[
                    ToolResultBlock(
                        tool_use_id=tool_use_blocks[0].id,
                        content=[TextBlock(text=tool_result)],
                    )
                ],
            )
        )

        print("\nStep 2: Continuing conversation with tool result...")
        # This is where the original error would occur if thinking blocks weren't properly handled
        response2 = await get_response(
            model="claude-3-7-sonnet-20250219",
            messages=messages,
            tools=[calculator_tool],
            reasoning=ReasoningConfig(budget_tokens=1024),
            max_tokens=3000,
        )

        thinking_blocks2 = [
            b for b in response2.content if isinstance(b, ThinkingBlock)
        ]
        text_blocks2 = [b for b in response2.content if isinstance(b, TextBlock)]

        print(f"Response 2: {len(thinking_blocks2)} thinking, {len(text_blocks2)} text")

        if text_blocks2:
            print(f"Final response: {text_blocks2[0].text}")

        # Test one more turn to ensure thinking blocks are preserved properly
        messages.append(response2)
        messages.append(
            ChatMessage(
                role=MessageRole.user,
                content=[TextBlock(text="Great! Now can you also calculate 100 / 4?")],
            )
        )

        print("\nStep 3: Third turn to test continued thinking preservation...")
        response3 = await get_response(
            model="claude-3-7-sonnet-20250219",
            messages=messages,
            tools=[calculator_tool],
            reasoning=ReasoningConfig(budget_tokens=1024),
            max_tokens=3000,
        )

        tool_use_blocks3 = [b for b in response3.content if isinstance(b, ToolUseBlock)]
        thinking_blocks3 = [
            b for b in response3.content if isinstance(b, ThinkingBlock)
        ]

        print(
            f"Response 3: {len(thinking_blocks3)} thinking, {len(tool_use_blocks3)} tool_use"
        )

        if tool_use_blocks3:
            print(
                f"Third tool call: {tool_use_blocks3[0].name} with {tool_use_blocks3[0].input}"
            )

        print("✓ Multi-turn thinking with tools test successful!")
    else:
        print("No tool use in first response")


async def test_thinking_block_format_preservation():
    """Test that thinking blocks maintain proper format across API calls"""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("No ANTHROPIC_API_KEY found, skipping test")
        return

    print("\n=== Testing Thinking Block Format Preservation ===")

    # Create a message with thinking
    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text="What is 7 * 8? Think about it.")],
        )
    ]

    response = await get_response(
        model="claude-3-7-sonnet-20250219",
        messages=messages,
        reasoning=ReasoningConfig(budget_tokens=1024),
        max_tokens=2000,
    )

    thinking_blocks = [b for b in response.content if isinstance(b, ThinkingBlock)]

    if thinking_blocks:
        thinking_block = thinking_blocks[0]

        print(f"Original thinking block:")
        print(f"  Type: {thinking_block.type}")
        print(f"  Has signature: {thinking_block.signature is not None}")
        print(f"  Thinking length: {len(thinking_block.thinking)}")

        # Test anthropic_format method
        anthropic_format = thinking_block.anthropic_format()
        print(f"Anthropic format:")
        print(f"  Type: {anthropic_format.get('type')}")
        print(f"  Has thinking field: {'thinking' in anthropic_format}")
        print(f"  Has signature field: {'signature' in anthropic_format}")

        # Test that the message's anthropic_format includes thinking blocks
        msg_format = response.anthropic_format()
        thinking_in_content = [
            c for c in msg_format["content"] if c.get("type") == "thinking"
        ]
        print(f"Message format includes {len(thinking_in_content)} thinking blocks")

        if thinking_in_content:
            print(f"  First thinking block type: {thinking_in_content[0].get('type')}")
            print(f"  Has thinking field: {'thinking' in thinking_in_content[0]}")
            print(f"  Has signature field: {'signature' in thinking_in_content[0]}")

        print("✓ Thinking block format preservation test successful!")


async def test_different_provider_exclusion():
    """Test that thinking blocks are properly excluded for non-Anthropic providers"""
    print("\n=== Testing Thinking Block Exclusion for Other Providers ===")

    # Create a mock message with thinking blocks
    mock_message = ChatMessage(
        role=MessageRole.assistant,
        content=[
            ThinkingBlock(thinking="Let me think about this..."),
            TextBlock(text="The answer is 42."),
        ],
    )

    # Test OpenAI format
    if os.environ.get("OPENAI_API_KEY"):
        print("Testing OpenAI format exclusion:")
        openai_format = mock_message.openai_format()

        if isinstance(openai_format, dict):
            content = openai_format.get("content", "")
            has_thinking_text = "Let me think about this" in str(content)
            print(
                f"  Thinking text excluded from OpenAI format: {not has_thinking_text}"
            )
            print(f"  OpenAI content: {content}")

        # Test in actual conversation
        messages = [
            ChatMessage(
                role=MessageRole.user, content=[TextBlock(text="What is 2+2?")]
            ),
            mock_message,
        ]

        try:
            response = await get_response(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=100,
            )
            print("  ✓ OpenAI successfully processed conversation with thinking blocks")
        except Exception as e:
            print(f"  ✗ OpenAI error: {e}")

    # Test Google format
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        print("Testing Google GenAI exclusion:")
        messages = [
            ChatMessage(
                role=MessageRole.user, content=[TextBlock(text="What is 2+2?")]
            ),
            mock_message,
        ]

        try:
            response = await get_response(
                model="gemini-2.5-flash",  # Available model
                messages=messages,
                max_tokens=100,
            )
            print(
                "  ✓ Google GenAI successfully processed conversation with thinking blocks"
            )
        except Exception as e:
            print(f"  ✗ Google GenAI error: {e}")


async def main():
    """Run multi-turn thinking tests"""
    await test_anthropic_multiturn_with_tools()
    await test_thinking_block_format_preservation()
    await test_different_provider_exclusion()
    print("\n=== All Multi-turn Thinking Tests Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
