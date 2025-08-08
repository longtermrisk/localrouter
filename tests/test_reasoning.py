import asyncio
import os
from localrouter.dtypes import (
    ChatMessage,
    MessageRole,
    TextBlock,
    ReasoningConfig,
)
from localrouter.llm import (
    get_response,
    get_response_with_backoff,
)


MAX_TOKENS = 4000


def get_available_reasoning_models():
    """Get models that support reasoning/thinking"""
    models = []

    # Check for OpenAI GPT-5 (when available)
    if os.environ.get("OPENAI_API_KEY"):
        # GPT-5 not yet available, but we can test the config generation
        models.append(
            ("gpt-4o-mini", "OpenAI", False)
        )  # Doesn't actually support reasoning
        # models.append(("gpt-5", "OpenAI", True))  # Will support reasoning when available

    # Check for Anthropic Claude with thinking
    if os.environ.get("ANTHROPIC_API_KEY"):
        models.append(
            ("claude-3-haiku-20240307", "Anthropic", False)
        )  # Doesn't support thinking
        # models.append(("claude-sonnet-4-20250514", "Anthropic", True))  # Would support thinking

    # Check for Google Gemini 2.5
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        models.append(("gemini-2.5-flash", "Google GenAI", True))  # Supports thinking
        models.append(("gemini-2.5-pro", "Google GenAI", True))  # Supports thinking

    return models


async def test_reasoning_config_creation():
    """Test creating ReasoningConfig with different parameters"""
    print("=== Testing ReasoningConfig Creation ===")

    # Test effort-based config
    config1 = ReasoningConfig(effort="low")
    print(f"Effort-based config: {config1}")

    # Test token-based config
    config2 = ReasoningConfig(budget_tokens=8000)
    print(f"Token-based config: {config2}")

    # Test dynamic config
    config3 = ReasoningConfig(dynamic=True)
    print(f"Dynamic config: {config3}")

    # Test conversion methods
    print("\nTesting format conversions:")

    # OpenAI format
    print(
        f"OpenAI format for GPT-5 with effort='medium': {config1.to_openai_format('gpt-5')}"
    )
    print(
        f"OpenAI format for GPT-4 (should be None): {config1.to_openai_format('gpt-4o')}"
    )

    # Anthropic format
    config4 = ReasoningConfig(effort="high")
    print(
        f"Anthropic format for claude-sonnet-4: {config4.to_anthropic_format('claude-sonnet-4-20250514')}"
    )
    print(
        f"Anthropic format for claude-3 (should be None): {config4.to_anthropic_format('claude-3-haiku')}"
    )

    # Gemini format
    print(
        f"Gemini format with budget_tokens=4000: {config2.to_gemini_format('gemini-2.5-pro')}"
    )
    print(
        f"Gemini format with dynamic=True: {config3.to_gemini_format('gemini-2.5-flash')}"
    )


async def test_reasoning_with_effort():
    """Test reasoning with effort levels"""
    print("\n=== Testing Reasoning with Effort Levels ===")

    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[
                TextBlock(
                    text="What is the sum of the first 10 prime numbers? Think step by step."
                )
            ],
        )
    ]

    models = get_available_reasoning_models()
    if not models:
        print("No API keys found. Skipping tests.")
        return

    for effort_level in ["low", "medium", "high"]:
        print(f"\nTesting with effort={effort_level}:")

        for model_name, provider, supports_reasoning in models:
            if not supports_reasoning:
                continue

            try:
                reasoning_config = ReasoningConfig(effort=effort_level)
                response = await get_response(
                    model=model_name,
                    messages=messages,
                    reasoning=reasoning_config,
                    max_tokens=MAX_TOKENS,
                )

                # Check for thinking metadata in response
                has_thinking = any(
                    block.meta.get("user_sees")
                    for block in response.content
                    if isinstance(block, TextBlock)
                )

                print(f"  {provider} ({model_name}):")
                print(f"    Response: {response.content[0].text[:100]}...")
                print(f"    Has thinking metadata: {has_thinking}")

            except Exception as e:
                print(f"  {provider} ({model_name}) error: {e}")


async def test_reasoning_with_token_budget():
    """Test reasoning with explicit token budgets"""
    print("\n=== Testing Reasoning with Token Budgets ===")

    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[
                TextBlock(
                    text="Explain quantum entanglement in simple terms, but think through the explanation carefully first."
                )
            ],
        )
    ]

    models = get_available_reasoning_models()
    if not models:
        print("No API keys found. Skipping tests.")
        return

    for budget in [1000, 4000, 8000]:
        print(f"\nTesting with budget_tokens={budget}:")

        for model_name, provider, supports_reasoning in models:
            if not supports_reasoning:
                continue

            try:
                reasoning_config = ReasoningConfig(budget_tokens=budget)
                response = await get_response(
                    model=model_name,
                    messages=messages,
                    reasoning=reasoning_config,
                    max_tokens=MAX_TOKENS,
                )

                print(f"  {provider} ({model_name}):")
                print(f"    Response length: {len(response.content[0].text)} chars")

            except Exception as e:
                print(f"  {provider} ({model_name}) error: {e}")


async def test_reasoning_with_dict_config():
    """Test reasoning with dict configuration (backward compatibility)"""
    print("\n=== Testing Reasoning with Dict Config ===")

    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text="What is 2+2? (This is just a simple test)")],
        )
    ]

    models = get_available_reasoning_models()
    if not models:
        print("No API keys found. Skipping tests.")
        return

    # Test various dict configs
    dict_configs = [
        {"effort": "low"},
        {"budget_tokens": 2000},
        {"dynamic": True},
    ]

    for config_dict in dict_configs:
        print(f"\nTesting dict config: {config_dict}")

        for model_name, provider, supports_reasoning in models:
            if not supports_reasoning:
                continue

            try:
                response = await get_response(
                    model=model_name,
                    messages=messages,
                    reasoning=config_dict,  # Pass dict instead of ReasoningConfig
                    max_tokens=MAX_TOKENS,
                )

                print(f"  {provider} ({model_name}): ✓ Success")

            except Exception as e:
                print(f"  {provider} ({model_name}): ✗ Error - {e}")


async def test_reasoning_with_dynamic():
    """Test dynamic reasoning (let model decide)"""
    print("\n=== Testing Dynamic Reasoning ===")

    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[
                TextBlock(
                    text="Is there an infinite number of prime numbers? Provide a brief proof."
                )
            ],
        )
    ]

    models = get_available_reasoning_models()
    if not models:
        print("No API keys found. Skipping tests.")
        return

    reasoning_config = ReasoningConfig(effort="medium")

    for model_name, provider, supports_reasoning in models:
        if not supports_reasoning:
            continue

        try:
            response = await get_response(
                model=model_name,
                messages=messages,
                reasoning=reasoning_config,
                max_tokens=MAX_TOKENS,
            )

            print(f"{provider} ({model_name}) with dynamic reasoning:")
            print(f"  Response: {response.content[0].text[:150]}...")

        except Exception as e:
            print(f"{provider} ({model_name}) dynamic reasoning error: {e}")


async def test_reasoning_disabled():
    """Test that reasoning can be disabled"""
    print("\n=== Testing Reasoning Disabled ===")

    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text="What is the capital of France?")],
        )
    ]

    models = get_available_reasoning_models()
    if not models:
        print("No API keys found. Skipping tests.")
        return

    for model_name, provider, supports_reasoning in models:
        try:
            # Call without reasoning config
            response = await get_response(
                model=model_name, messages=messages, max_tokens=MAX_TOKENS
            )

            print(f"{provider} ({model_name}) without reasoning:")
            print(f"  Response: {response.content[0].text}")

        except Exception as e:
            print(f"{provider} ({model_name}) no reasoning error: {e}")


async def test_reasoning_with_backoff():
    """Test reasoning with backoff functionality"""
    print("\n=== Testing Reasoning with Backoff ===")

    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text="What is the derivative of x^2?")],
        )
    ]

    models = get_available_reasoning_models()
    if not models:
        print("No API keys found. Skipping tests.")
        return

    reasoning_config = ReasoningConfig(effort="low")

    for model_name, provider, supports_reasoning in models:
        if not supports_reasoning:
            continue

        try:
            response = await get_response_with_backoff(
                model=model_name,
                messages=messages,
                reasoning=reasoning_config,
                max_tokens=MAX_TOKENS,
            )

            print(f"{provider} ({model_name}) with backoff and reasoning:")
            print(f"  Response: {response.content[0].text}")

        except Exception as e:
            print(f"{provider} ({model_name}) backoff+reasoning error: {e}")


async def test_effort_validation():
    """Test that invalid effort levels are rejected"""
    print("\n=== Testing Effort Validation ===")

    try:
        invalid_config = ReasoningConfig(effort="super-high")
        print("ERROR: Should have raised ValueError for invalid effort")
    except ValueError as e:
        print(f"Correctly rejected invalid effort: {e}")

    try:
        valid_config = ReasoningConfig(effort="high")
        print(f"Correctly accepted valid effort: {valid_config}")
    except Exception as e:
        print(f"ERROR: Should have accepted valid effort: {e}")


async def test_dict_to_config_conversion():
    """Test that dict configs are properly converted to ReasoningConfig"""
    print("\n=== Testing Dict to ReasoningConfig Conversion ===")

    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text="Hello")],
        )
    ]

    # Test that various dict formats work
    test_cases = [
        ({"effort": "low"}, "effort-based dict"),
        ({"budget_tokens": 1000}, "token-based dict"),
        ({"dynamic": True}, "dynamic dict"),
        (
            {"effort": "high", "budget_tokens": 5000},
            "mixed dict (should prioritize effort)",
        ),
    ]

    for config_dict, description in test_cases:
        print(f"\nTesting {description}: {config_dict}")
        try:
            # This should work without errors
            response = await get_response(
                model="gpt-4o-mini",
                messages=messages,
                reasoning=config_dict,
                max_tokens=10,
            )
            print(f"  ✓ Dict config accepted and processed")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    # Test invalid dict configs
    print("\nTesting invalid dict configs:")
    invalid_cases = [
        ({"effort": "ultra-max"}, "invalid effort level"),
        ({"budget_tokens": "not_a_number"}, "invalid type for budget_tokens"),
    ]

    for config_dict, description in invalid_cases:
        print(f"\n  Testing {description}: {config_dict}")
        try:
            response = await get_response(
                model="gpt-4o-mini",
                messages=messages,
                reasoning=config_dict,
                max_tokens=10,
            )
            print(f"    ✗ Should have raised error but didn't")
        except Exception as e:
            print(f"    ✓ Correctly raised: {type(e).__name__}")


async def main():
    """Run all reasoning tests"""
    models = get_available_reasoning_models()
    print(
        f"Found models: {', '.join([f'{provider} ({model})' for model, provider, _ in models])}"
    )

    if not models:
        print(
            "No API keys found. Please set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY/GOOGLE_API_KEY"
        )
        return

    # Run all tests
    await test_reasoning_config_creation()
    await test_effort_validation()
    await test_dict_to_config_conversion()
    await test_reasoning_with_effort()
    await test_reasoning_with_token_budget()
    await test_reasoning_with_dict_config()
    await test_reasoning_with_dynamic()
    await test_reasoning_disabled()
    await test_reasoning_with_backoff()


if __name__ == "__main__":
    asyncio.run(main())
