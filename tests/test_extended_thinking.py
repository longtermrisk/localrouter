"""
Comprehensive tests for extended thinking/reasoning across different providers.

Tests multi-turn conversations, tool use, and proper handling of thinking blocks.
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


def get_available_extended_thinking_models():
    """Get models that support extended thinking"""
    models = []

    # Check for OpenAI GPT models (GPT-5 when available, o1/o3 models)
    if os.environ.get("OPENAI_API_KEY"):
        # o1-mini supports reasoning but doesn't expose thinking blocks
        models.append(
            ("o1-mini", "OpenAI", True, False)
        )  # supports reasoning, no thinking blocks visible
        # GPT-5 will support reasoning with thinking blocks when available
        # models.append(("gpt-5", "OpenAI", True, True))  # would support reasoning with thinking blocks

        # Regular models don't support reasoning
        models.append(("gpt-4o-mini", "OpenAI", False, False))

    # Check for Anthropic Claude models with thinking
    if os.environ.get("ANTHROPIC_API_KEY"):
        # Extended thinking models
        models.append(
            ("claude-3-7-sonnet-20250219", "Anthropic", True, True)
        )  # Sonnet 3.7 with thinking
        # models.append(("claude-sonnet-4-20250514", "Anthropic", True, True))  # Sonnet 4 with thinking
        # models.append(("claude-opus-4-1-20250805", "Anthropic", True, True))  # Opus 4.1 with thinking

        # Regular models don't support thinking
        models.append(("claude-3-haiku-20240307", "Anthropic", False, False))

    # Check for Google Gemini models
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        # Gemini 2.5 models support thinking
        models.append(("gemini-2.5-flash", "Google GenAI", True, True))
        models.append(("gemini-2.5-pro", "Google GenAI", True, True))

        # Regular models don't support thinking
        models.append(("gemini-1.5-flash", "Google GenAI", False, False))

    return models


# Simple calculator tool for testing
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


async def simulate_tool_call(tool_use: ToolUseBlock) -> str:
    """Simulate a tool call for testing"""
    if tool_use.name == "calculator" and tool_use.input:
        expression = tool_use.input.get("expression", "")
        try:
            # Simple eval for testing (don't use in production!)
            result = eval(expression)
            return f"The result is: {result}"
        except:
            return f"Error evaluating expression: {expression}"
    return "Tool not implemented"


async def test_basic_extended_thinking():
    """Test basic extended thinking across all models"""
    print("=== Testing Basic Extended Thinking ===")

    models = get_available_extended_thinking_models()
    if not models:
        print("No API keys found. Skipping tests.")
        return

    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text="What is 23 + 47? Think step by step.")],
        )
    ]

    reasoning_config = ReasoningConfig(budget_tokens=1500)  # Minimum is 1024

    for model_name, provider, supports_thinking, has_visible_thinking in models:
        print(f"\n{provider} ({model_name}):")

        try:
            if supports_thinking:
                # For Anthropic thinking models, max_tokens must be > budget_tokens
                max_tokens = 6000 if provider == "Anthropic" else 4000
                response = await get_response(
                    model=model_name,
                    messages=messages,
                    reasoning=reasoning_config,
                    max_tokens=max_tokens,
                )
            else:
                # For non-thinking models, use appropriate limits
                if "haiku" in model_name:
                    max_tokens = 4000  # Haiku has 4096 token limit
                else:
                    max_tokens = 4000
                response = await get_response(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                )

            # Check for thinking blocks
            thinking_blocks = [
                b for b in response.content if isinstance(b, ThinkingBlock)
            ]
            text_blocks = [b for b in response.content if isinstance(b, TextBlock)]

            print(f"  Supports thinking: {supports_thinking}")
            print(f"  Has visible thinking blocks: {len(thinking_blocks) > 0}")
            print(f"  Number of thinking blocks: {len(thinking_blocks)}")
            print(
                f"  Response: {text_blocks[0].text if text_blocks else 'No text response'}"
            )

            if thinking_blocks:
                print(f"  Thinking content: {thinking_blocks[0].thinking[:100]}...")
                print(f"  Has signature: {thinking_blocks[0].signature is not None}")

        except Exception as e:
            print(f"  Error: {e}")


async def test_multiturn_extended_thinking():
    """Test multi-turn conversations with extended thinking"""
    print("\n=== Testing Multi-turn Extended Thinking ===")

    models = get_available_extended_thinking_models()
    if not models:
        print("No API keys found. Skipping tests.")
        return

    for model_name, provider, supports_thinking, has_visible_thinking in models:
        if not supports_thinking:
            print(f"\n{provider} ({model_name}): Skipping (no thinking support)")
            continue

        print(f"\n{provider} ({model_name}):")

        try:
            # First turn
            messages = [
                ChatMessage(
                    role=MessageRole.user,
                    content=[
                        TextBlock(
                            text="I'm thinking of a number between 1 and 10. Can you guess what it is? Think about the probabilities."
                        )
                    ],
                )
            ]

            reasoning_config = ReasoningConfig(budget_tokens=1500)

            response1 = await get_response(
                model=model_name,
                messages=messages,
                reasoning=reasoning_config,
                max_tokens=6000,  # Must be greater than budget_tokens
            )

            print(
                f"  Turn 1 - Response: {response1.content[-1].text if response1.content else 'No response'}"
            )

            # Add the assistant's response to conversation
            messages.append(response1)

            # Second turn - give feedback
            messages.append(
                ChatMessage(
                    role=MessageRole.user,
                    content=[
                        TextBlock(
                            text="Good guess! The number I was thinking of was 7. Now let me think of a number between 1 and 20. What's your guess this time?"
                        )
                    ],
                )
            )

            response2 = await get_response(
                model=model_name,
                messages=messages,
                reasoning=reasoning_config,
                max_tokens=6000,
            )

            print(
                f"  Turn 2 - Response: {response2.content[-1].text if response2.content else 'No response'}"
            )

            # Check that thinking blocks are properly handled
            thinking_blocks_turn1 = [
                b for b in response1.content if isinstance(b, ThinkingBlock)
            ]
            thinking_blocks_turn2 = [
                b for b in response2.content if isinstance(b, ThinkingBlock)
            ]

            print(f"  Turn 1 thinking blocks: {len(thinking_blocks_turn1)}")
            print(f"  Turn 2 thinking blocks: {len(thinking_blocks_turn2)}")

        except Exception as e:
            print(f"  Error: {e}")


async def test_extended_thinking_with_tools():
    """Test extended thinking with tool use"""
    print("\n=== Testing Extended Thinking with Tools ===")

    models = get_available_extended_thinking_models()
    if not models:
        print("No API keys found. Skipping tests.")
        return

    for model_name, provider, supports_thinking, has_visible_thinking in models:
        if not supports_thinking:
            print(f"\n{provider} ({model_name}): Skipping (no thinking support)")
            continue

        print(f"\n{provider} ({model_name}):")

        try:
            messages = [
                ChatMessage(
                    role=MessageRole.user,
                    content=[
                        TextBlock(
                            text="What is (15 * 23) + (47 - 12)? Use the calculator tool and think through this step by step."
                        )
                    ],
                )
            ]

            reasoning_config = ReasoningConfig(budget_tokens=1500)

            response = await get_response(
                model=model_name,
                messages=messages,
                tools=[calculator_tool],
                reasoning=reasoning_config,
                max_tokens=6000,  # Must be greater than budget_tokens
            )

            # Check for tool use
            tool_use_blocks = [
                b for b in response.content if isinstance(b, ToolUseBlock)
            ]
            thinking_blocks = [
                b for b in response.content if isinstance(b, ThinkingBlock)
            ]

            print(f"  Tool use blocks: {len(tool_use_blocks)}")
            print(f"  Thinking blocks: {len(thinking_blocks)}")

            if tool_use_blocks:
                print(
                    f"  First tool call: {tool_use_blocks[0].name} with {tool_use_blocks[0].input}"
                )

                # Simulate tool result
                tool_result = await simulate_tool_call(tool_use_blocks[0])
                print(f"  Tool result: {tool_result}")

                # Continue conversation with tool result
                messages.append(response)
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

                # Get final response
                final_response = await get_response(
                    model=model_name,
                    messages=messages,
                    tools=[calculator_tool],
                    reasoning=reasoning_config,
                    max_tokens=6000,
                )

                final_thinking = [
                    b for b in final_response.content if isinstance(b, ThinkingBlock)
                ]
                print(f"  Final response thinking blocks: {len(final_thinking)}")
                print(
                    f"  Final response: {final_response.content[-1].text if final_response.content else 'No final response'}"
                )

        except Exception as e:
            print(f"  Error: {e}")


async def test_thinking_block_preservation():
    """Test that thinking blocks are properly preserved in multi-turn conversations"""
    print("\n=== Testing Thinking Block Preservation ===")

    models = get_available_extended_thinking_models()
    anthropic_models = [(m, p, s, v) for m, p, s, v in models if p == "Anthropic" and s]

    if not anthropic_models:
        print("No Anthropic models with thinking support found.")
        return

    for (
        model_name,
        provider,
        supports_thinking,
        has_visible_thinking,
    ) in anthropic_models:
        print(f"\n{provider} ({model_name}):")

        try:
            # Test that thinking blocks from previous turns are included in Anthropic format
            messages = [
                ChatMessage(
                    role=MessageRole.user,
                    content=[TextBlock(text="What's 5 + 5? Think about it.")],
                )
            ]

            response1 = await get_response(
                model=model_name,
                messages=messages,
                reasoning=ReasoningConfig(budget_tokens=1000),
                max_tokens=2000,
            )

            thinking_blocks = [
                b for b in response1.content if isinstance(b, ThinkingBlock)
            ]
            print(f"  Response 1 thinking blocks: {len(thinking_blocks)}")

            if thinking_blocks:
                # Test the anthropic_format method
                anthropic_msg = response1.anthropic_format()
                anthropic_content = anthropic_msg["content"]

                thinking_in_format = [
                    c for c in anthropic_content if c.get("type") == "thinking"
                ]
                print(
                    f"  Thinking blocks in Anthropic format: {len(thinking_in_format)}"
                )

                if thinking_in_format:
                    print(
                        f"  First thinking block has thinking field: {'thinking' in thinking_in_format[0]}"
                    )
                    print(
                        f"  First thinking block has signature: {'signature' in thinking_in_format[0]}"
                    )

                # Test multi-turn with thinking preservation
                messages.append(response1)
                messages.append(
                    ChatMessage(
                        role=MessageRole.user,
                        content=[TextBlock(text="Now what's 10 + 10?")],
                    )
                )

                response2 = await get_response(
                    model=model_name,
                    messages=messages,
                    reasoning=ReasoningConfig(budget_tokens=1000),
                    max_tokens=2000,
                )

                print(f"  Response 2 successful with preserved thinking blocks")

        except Exception as e:
            print(f"  Error: {e}")


async def test_thinking_exclusion_for_other_providers():
    """Test that thinking blocks are excluded for OpenAI and Google"""
    print("\n=== Testing Thinking Block Exclusion for Other Providers ===")

    models = get_available_extended_thinking_models()
    non_anthropic_models = [
        (m, p, s, v) for m, p, s, v in models if p != "Anthropic" and s
    ]

    if not non_anthropic_models:
        print("No non-Anthropic models with thinking support found.")
        return

    for (
        model_name,
        provider,
        supports_thinking,
        has_visible_thinking,
    ) in non_anthropic_models:
        print(f"\n{provider} ({model_name}):")

        try:
            # Create a mock ChatMessage with thinking blocks
            mock_response = ChatMessage(
                role=MessageRole.assistant,
                content=[
                    ThinkingBlock(thinking="Let me think about this..."),
                    TextBlock(text="The answer is 42."),
                ],
            )

            if provider == "OpenAI":
                openai_format = mock_response.openai_format()
                print(
                    f"  OpenAI format excludes thinking: {isinstance(openai_format, dict)}"
                )
                if isinstance(openai_format, dict):
                    # Check that thinking blocks are not in the content
                    content = openai_format.get("content", "")
                    has_thinking_text = "Let me think about this" in str(content)
                    print(f"  Thinking text found in content: {has_thinking_text}")

            elif provider == "Google GenAI":
                # Test that thinking blocks are excluded during genai_format processing
                messages = [
                    ChatMessage(
                        role=MessageRole.user, content=[TextBlock(text="What is 2+2?")]
                    ),
                    mock_response,
                ]

                # This should not include thinking blocks in the GenAI request
                response = await get_response(
                    model=model_name,
                    messages=messages,
                    max_tokens=100,
                )
                print(
                    f"  Successfully processed message with thinking blocks (excluded)"
                )

        except Exception as e:
            print(f"  Error: {e}")


async def test_reasoning_config_format_conversion():
    """Test ReasoningConfig conversion to different provider formats"""
    print("\n=== Testing ReasoningConfig Format Conversion ===")

    test_configs = [
        ReasoningConfig(effort="low"),
        ReasoningConfig(budget_tokens=8000),
        ReasoningConfig(dynamic=True),
        ReasoningConfig(effort="high", budget_tokens=16000),  # Should prioritize effort
    ]

    test_models = [
        ("gpt-5", "OpenAI"),
        ("claude-sonnet-4-20250514", "Anthropic"),
        ("claude-3-7-sonnet-20250219", "Anthropic"),
        ("gemini-2.5-pro", "Google GenAI"),
        ("gpt-4o", "OpenAI"),  # Should return None
        ("claude-3-haiku", "Anthropic"),  # Should return None
        ("gemini-1.5-flash", "Google GenAI"),  # Should return None
    ]

    for config in test_configs:
        print(f"\nTesting config: {config}")

        for model, provider in test_models:
            if provider == "OpenAI":
                result = config.to_openai_format(model)
                print(f"  {provider} ({model}): {result}")
            elif provider == "Anthropic":
                result = config.to_anthropic_format(model)
                print(f"  {provider} ({model}): {result}")
            elif provider == "Google GenAI":
                result = config.to_gemini_format(model)
                print(f"  {provider} ({model}): {result}")


async def test_error_handling():
    """Test error handling with extended thinking"""
    print("\n=== Testing Error Handling ===")

    models = get_available_extended_thinking_models()
    if not models:
        print("No API keys found. Skipping tests.")
        return

    # Test invalid reasoning config
    print("\nTesting invalid reasoning config:")
    invalid_config = {"invalid_field": "value"}

    for model_name, provider, supports_thinking, has_visible_thinking in models[
        :1
    ]:  # Test with first model only
        try:
            response = await get_response(
                model=model_name,
                messages=[
                    ChatMessage(
                        role=MessageRole.user, content=[TextBlock(text="Hello")]
                    )
                ],
                reasoning=invalid_config,
                max_tokens=100,
            )
            print(f"  {provider}: Handled invalid config gracefully")
        except Exception as e:
            print(f"  {provider}: Error with invalid config - {type(e).__name__}")

    # Test invalid effort level
    print("\nTesting invalid effort level:")
    try:
        bad_config = ReasoningConfig(effort="ultra-high")
        print("  ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"  Correctly rejected invalid effort: {e}")


async def main():
    """Run all extended thinking tests"""
    models = get_available_extended_thinking_models()
    print(f"Found models:")
    for model, provider, supports_thinking, has_visible_thinking in models:
        print(
            f"  {provider} ({model}) - Thinking: {supports_thinking}, Visible: {has_visible_thinking}"
        )

    if not models:
        print(
            "\nNo API keys found. Please set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY/GOOGLE_API_KEY"
        )
        return

    # Run all tests
    await test_basic_extended_thinking()
    await test_multiturn_extended_thinking()
    await test_extended_thinking_with_tools()
    await test_thinking_block_preservation()
    await test_thinking_exclusion_for_other_providers()
    await test_reasoning_config_format_conversion()
    await test_error_handling()

    print("\n=== Extended Thinking Tests Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
