"""
Basic test for extended thinking functionality.
"""

import asyncio
import os
from localrouter.dtypes import (
    ChatMessage,
    MessageRole,
    TextBlock,
    ThinkingBlock,
    ReasoningConfig,
)
from localrouter.llm import get_response


async def test_anthropic_thinking():
    """Test Anthropic extended thinking"""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("No ANTHROPIC_API_KEY found, skipping Anthropic test")
        return

    print("=== Testing Anthropic Extended Thinking ===")

    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text="What is 15 + 27? Think step by step.")],
        )
    ]

    # Test with thinking enabled
    response = await get_response(
        model="claude-3-7-sonnet-20250219",
        messages=messages,
        reasoning=ReasoningConfig(budget_tokens=1024),  # Minimum is 1024
        max_tokens=3000,  # Must be > budget_tokens
    )

    thinking_blocks = [b for b in response.content if isinstance(b, ThinkingBlock)]
    text_blocks = [b for b in response.content if isinstance(b, TextBlock)]

    print(f"Response has {len(thinking_blocks)} thinking block(s)")
    print(f"Response has {len(text_blocks)} text block(s)")

    if thinking_blocks:
        print(f"Thinking content: {thinking_blocks[0].thinking[:200]}...")
        print(f"Has signature: {thinking_blocks[0].signature is not None}")

        # Test anthropic_format method
        anthropic_format = thinking_blocks[0].anthropic_format()
        print(f"Anthropic format type: {anthropic_format.get('type')}")
        print(f"Anthropic format has thinking: {'thinking' in anthropic_format}")
        print(f"Anthropic format has signature: {'signature' in anthropic_format}")

    if text_blocks:
        print(f"Text response: {text_blocks[0].text}")

    # Test multi-turn conversation
    print("\n--- Testing Multi-turn ---")
    messages.append(response)
    messages.append(
        ChatMessage(
            role=MessageRole.user, content=[TextBlock(text="Now what is 42 + 58?")]
        )
    )

    response2 = await get_response(
        model="claude-3-7-sonnet-20250219",
        messages=messages,
        reasoning=ReasoningConfig(budget_tokens=1024),  # Minimum is 1024
        max_tokens=3000,
    )

    thinking_blocks2 = [b for b in response2.content if isinstance(b, ThinkingBlock)]
    text_blocks2 = [b for b in response2.content if isinstance(b, TextBlock)]

    print(
        f"Turn 2: {len(thinking_blocks2)} thinking blocks, {len(text_blocks2)} text blocks"
    )
    if text_blocks2:
        print(f"Turn 2 response: {text_blocks2[0].text}")

    print("✓ Anthropic thinking test successful!")


async def test_openai_thinking():
    """Test OpenAI reasoning (o1 models)"""
    if not os.environ.get("OPENAI_API_KEY"):
        print("No OPENAI_API_KEY found, skipping OpenAI test")
        return

    print("\n=== Testing OpenAI Reasoning ===")

    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text="What is the sum of the first 5 prime numbers?")],
        )
    ]

    # Test with o1-mini (has reasoning but no visible thinking blocks)
    response = await get_response(
        model="o1-mini",
        messages=messages,
        reasoning=ReasoningConfig(effort="low"),  # o1 models use effort levels
        max_tokens=1000,
    )

    thinking_blocks = [b for b in response.content if isinstance(b, ThinkingBlock)]
    text_blocks = [b for b in response.content if isinstance(b, TextBlock)]

    print(f"o1-mini response has {len(thinking_blocks)} thinking blocks (expected: 0)")
    print(f"o1-mini response has {len(text_blocks)} text blocks")

    if text_blocks:
        print(f"Response: {text_blocks[0].text}")

    print("✓ OpenAI reasoning test successful!")


async def test_google_thinking():
    """Test Google Gemini thinking"""
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("No GEMINI_API_KEY or GOOGLE_API_KEY found, skipping Google test")
        return

    print("\n=== Testing Google Gemini Thinking ===")

    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text="What is 123 + 456? Think carefully.")],
        )
    ]

    # Test with Gemini 2.5 Flash
    try:
        response = await get_response(
            model="gemini-2.5-flash",
            messages=messages,
            reasoning=ReasoningConfig(budget_tokens=2000),
            max_tokens=3000,
        )

        thinking_blocks = [b for b in response.content if isinstance(b, ThinkingBlock)]
        text_blocks = [b for b in response.content if isinstance(b, TextBlock)]

        print(f"Gemini response has {len(thinking_blocks)} thinking blocks")
        print(f"Gemini response has {len(text_blocks)} text blocks")

        if text_blocks:
            print(f"Response: {text_blocks[0].text}")

        print("✓ Google thinking test successful!")

    except Exception as e:
        print(f"Google test failed: {e}")


async def test_reasoning_config():
    """Test ReasoningConfig functionality"""
    print("\n=== Testing ReasoningConfig ===")

    # Test different config types
    configs = [
        ReasoningConfig(effort="low"),
        ReasoningConfig(budget_tokens=1500),
        ReasoningConfig(dynamic=True),
    ]

    for i, config in enumerate(configs):
        print(f"\nConfig {i+1}: {config}")

        # Test format conversions
        print(f"  OpenAI format (gpt-5): {config.to_openai_format('gpt-5')}")
        print(
            f"  Anthropic format (claude-sonnet-4): {config.to_anthropic_format('claude-sonnet-4-20250514')}"
        )
        print(
            f"  Gemini format (gemini-2.5): {config.to_gemini_format('gemini-2.5-pro')}"
        )

    print("✓ ReasoningConfig test successful!")


async def main():
    """Run basic thinking tests"""
    await test_reasoning_config()
    await test_anthropic_thinking()
    await test_openai_thinking()
    await test_google_thinking()
    print("\n=== All Basic Thinking Tests Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
