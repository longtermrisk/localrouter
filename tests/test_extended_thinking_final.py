"""
Final comprehensive test to verify extended thinking functionality is working correctly.
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


async def test_original_error_scenario():
    """Test the exact scenario that was causing the original error"""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Skipping Anthropic test - no API key")
        return False

    print("=== Testing Original Error Scenario ===")
    print("This test reproduces the original error scenario that should now work")

    # Define a tool
    calculator = ToolDefinition(
        name="calculator",
        description="Perform mathematical calculations",
        input_schema={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression"}
            },
            "required": ["expression"],
        },
    )

    # Start conversation
    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[
                TextBlock(
                    text="Calculate 15 * 8 using the calculator. Think step by step."
                )
            ],
        )
    ]

    try:
        print("Step 1: Initial request with thinking and tools...")
        response1 = await get_response(
            model="claude-3-7-sonnet-20250219",
            messages=messages,
            tools=[calculator],
            reasoning=ReasoningConfig(budget_tokens=1500),
            max_tokens=4000,
        )

        # Check response structure
        thinking_blocks = [b for b in response1.content if isinstance(b, ThinkingBlock)]
        tool_blocks = [b for b in response1.content if isinstance(b, ToolUseBlock)]

        print(f"  Response has {len(thinking_blocks)} thinking blocks")
        print(f"  Response has {len(tool_blocks)} tool use blocks")

        if not tool_blocks:
            print("  No tool use in response - test inconclusive")
            return False

        # Add response to conversation
        messages.append(response1)

        # Add tool result
        messages.append(
            ChatMessage(
                role=MessageRole.user,
                content=[
                    ToolResultBlock(
                        tool_use_id=tool_blocks[0].id, content=[TextBlock(text="120")]
                    )
                ],
            )
        )

        print("Step 2: Continue conversation with tool result...")
        print("  (This step would fail with the original error)")

        # This is the step that was failing before our fix
        response2 = await get_response(
            model="claude-3-7-sonnet-20250219",
            messages=messages,
            tools=[calculator],
            reasoning=ReasoningConfig(budget_tokens=1500),
            max_tokens=4000,
        )

        print("  ✅ Successfully continued conversation!")
        print(f"  Response 2 has {len(response2.content)} content blocks")

        return True

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


async def test_thinking_preservation():
    """Test that thinking blocks are properly preserved in Anthropic format"""
    print("\n=== Testing Thinking Block Preservation ===")

    # Create a response with thinking
    response = ChatMessage(
        role=MessageRole.assistant,
        content=[
            ThinkingBlock(
                thinking="Let me think about this problem...", signature="abc123"
            ),
            TextBlock(text="Here's my answer."),
        ],
    )

    # Test anthropic_format method
    anthropic_format = response.anthropic_format()
    content = anthropic_format["content"]

    thinking_blocks = [c for c in content if c.get("type") == "thinking"]
    text_blocks = [c for c in content if c.get("type") == "text"]

    print(f"Anthropic format contains {len(thinking_blocks)} thinking blocks")
    print(f"Anthropic format contains {len(text_blocks)} text blocks")

    if thinking_blocks:
        tb = thinking_blocks[0]
        print(f"  Thinking block has 'thinking' field: {'thinking' in tb}")
        print(f"  Thinking block has 'signature' field: {'signature' in tb}")
        print(f"  Content: {tb.get('thinking', '')[:50]}...")

    return len(thinking_blocks) == 1 and len(text_blocks) == 1


async def test_openai_exclusion():
    """Test that thinking blocks are excluded from OpenAI format"""
    print("\n=== Testing OpenAI Thinking Block Exclusion ===")

    response = ChatMessage(
        role=MessageRole.assistant,
        content=[
            ThinkingBlock(thinking="This should be excluded"),
            TextBlock(text="This should be included"),
        ],
    )

    openai_format = response.openai_format()

    if isinstance(openai_format, dict):
        content = openai_format.get("content", "")
        has_thinking = "This should be excluded" in str(content)
        has_text = "This should be included" in str(content)

        print(f"OpenAI format excludes thinking: {not has_thinking}")
        print(f"OpenAI format includes text: {has_text}")

        return not has_thinking and has_text
    else:
        print("Unexpected OpenAI format structure")
        return False


async def test_reasoning_config():
    """Test ReasoningConfig format conversion"""
    print("\n=== Testing ReasoningConfig Conversion ===")

    config = ReasoningConfig(budget_tokens=2000)

    # Test Anthropic conversion
    anthropic_result = config.to_anthropic_format("claude-3-7-sonnet-20250219")
    print(f"Anthropic format: {anthropic_result}")

    # Test OpenAI conversion
    openai_result = config.to_openai_format("gpt-5")
    print(f"OpenAI format (gpt-5): {openai_result}")

    # Test Gemini conversion
    gemini_result = config.to_gemini_format("gemini-2.5-pro")
    print(f"Gemini format: {gemini_result}")

    # Test non-supporting models
    no_support = config.to_anthropic_format("claude-3-haiku")
    print(f"Non-supporting model: {no_support}")

    expected_anthropic = (
        anthropic_result and anthropic_result.get("budget_tokens") == 2000
    )
    expected_openai = openai_result and openai_result.get("effort") == "minimal"
    expected_gemini = gemini_result and gemini_result.get("thinking_budget") == 2000

    return (
        expected_anthropic
        and expected_openai
        and expected_gemini
        and no_support is None
    )


async def main():
    """Run final comprehensive test"""
    print("🧠 Extended Thinking - Final Comprehensive Test")
    print("=" * 50)

    results = []

    # Test 1: Original error scenario
    results.append(await test_original_error_scenario())

    # Test 2: Thinking preservation
    results.append(await test_thinking_preservation())

    # Test 3: OpenAI exclusion
    results.append(await test_openai_exclusion())

    # Test 4: ReasoningConfig
    results.append(await test_reasoning_config())

    print(f"\n{'='*50}")
    print(f"📊 Test Results: {sum(results)}/{len(results)} passed")

    if all(results):
        print("🎉 All tests passed! Extended thinking is working correctly.")
        print("\n✅ Key Fixes Confirmed:")
        print("   • ThinkingBlock.anthropic_format() returns proper thinking format")
        print("   • ChatMessage.anthropic_format() includes thinking blocks")
        print("   • Multi-turn conversations with tools work correctly")
        print("   • Thinking blocks are excluded from OpenAI/Google formats")
        print("   • ReasoningConfig conversion works for all providers")
    else:
        print("❌ Some tests failed. Check the output above for details.")

    return all(results)


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
