import pytest
import re
from unittest.mock import AsyncMock, patch
from localrouter import add_provider, ChatMessage, MessageRole, TextBlock, get_response
from localrouter.llm import providers, Provider


@pytest.mark.asyncio
async def test_provider_priority_routing():
    """Test that providers are selected based on priority"""

    # Mock response
    mock_response = ChatMessage(
        role=MessageRole.assistant, content=[TextBlock(text="Test response")]
    )

    # Create mock providers with different priorities
    high_priority_func = AsyncMock(return_value=mock_response)
    low_priority_func = AsyncMock(return_value=mock_response)

    # Save original providers to restore later
    original_providers = providers.copy()
    providers.clear()

    try:
        # Add providers with different priorities
        providers.append(
            Provider(low_priority_func, models=["test-model"], priority=100)
        )
        providers.append(
            Provider(high_priority_func, models=["test-model"], priority=10)
        )

        # Make request - should use high priority provider
        response = await get_response(
            model="test-model",
            messages=[
                ChatMessage(role=MessageRole.user, content=[TextBlock(text="test")])
            ],
        )

        # Verify high priority provider was called, low priority was not
        high_priority_func.assert_called_once()
        low_priority_func.assert_not_called()

    finally:
        # Restore original providers
        providers.clear()
        providers.extend(original_providers)


@pytest.mark.asyncio
async def test_regex_pattern_matching():
    """Test that regex patterns work for model matching"""

    mock_response = ChatMessage(
        role=MessageRole.assistant, content=[TextBlock(text="Test response")]
    )

    regex_func = AsyncMock(return_value=mock_response)
    exact_func = AsyncMock(return_value=mock_response)

    # Save original providers
    original_providers = providers.copy()
    providers.clear()

    try:
        # Add provider with regex pattern
        providers.append(
            Provider(regex_func, models=[re.compile(r"custom-.*")], priority=50)
        )
        # Add provider with exact match
        providers.append(Provider(exact_func, models=["custom-exact"], priority=10))

        # Test regex match
        await get_response(
            model="custom-test-123",
            messages=[
                ChatMessage(role=MessageRole.user, content=[TextBlock(text="test")])
            ],
        )
        regex_func.assert_called_once()

        # Test exact match takes priority over regex
        regex_func.reset_mock()
        exact_func.reset_mock()

        await get_response(
            model="custom-exact",
            messages=[
                ChatMessage(role=MessageRole.user, content=[TextBlock(text="test")])
            ],
        )

        # Exact match should be called (higher priority)
        exact_func.assert_called_once()
        regex_func.assert_not_called()

    finally:
        providers.clear()
        providers.extend(original_providers)


@pytest.mark.asyncio
async def test_openrouter_fallback():
    """Test that OpenRouter works as fallback for models with '/'"""

    # Find OpenRouter provider in current providers
    openrouter_provider = None
    for provider in providers:
        # Check if provider has regex pattern for models with "/"
        for model_pattern in provider.models:
            if hasattr(model_pattern, "pattern") and ".+/.+" in model_pattern.pattern:
                openrouter_provider = provider
                break

    if openrouter_provider is None:
        pytest.skip("OpenRouter provider not found (OPENROUTER_API_KEY not set)")

    # Verify OpenRouter has lowest priority
    assert openrouter_provider.priority == 1000

    # Verify it matches models with "/"
    assert openrouter_provider.supports_model("meta-llama/llama-3.3-70b")
    assert openrouter_provider.supports_model("qwen/qwen3-235b-a22b")
    assert not openrouter_provider.supports_model("gpt-4o-mini")


def test_add_provider_function():
    """Test the add_provider convenience function"""

    async def dummy_provider(**kwargs):
        return ChatMessage(
            role=MessageRole.assistant, content=[TextBlock(text="dummy")]
        )

    # Save original providers
    original_count = len(providers)

    try:
        # Add custom provider
        add_provider(
            dummy_provider, models=["dummy-model", re.compile(r"test-.*")], priority=75
        )

        # Verify provider was added
        assert len(providers) == original_count + 1

        # Find our provider
        added_provider = providers[-1]
        assert added_provider.get_response == dummy_provider
        assert added_provider.priority == 75
        assert added_provider.supports_model("dummy-model")
        assert added_provider.supports_model("test-123")
        assert not added_provider.supports_model("other-model")

    finally:
        # Remove our test provider
        if len(providers) > original_count:
            providers.pop()


@pytest.mark.asyncio
async def test_model_not_found_error():
    """Test error message when model is not supported"""

    with pytest.raises(ValueError) as exc_info:
        await get_response(
            model="definitely-not-a-real-model-12345",
            messages=[
                ChatMessage(role=MessageRole.user, content=[TextBlock(text="test")])
            ],
        )

    error_msg = str(exc_info.value)
    assert "definitely-not-a-real-model-12345" in error_msg
    assert "not supported by any provider" in error_msg


if __name__ == "__main__":
    import asyncio

    async def main():
        await test_provider_priority_routing()
        print("✓ Provider priority routing test passed")

        await test_regex_pattern_matching()
        print("✓ Regex pattern matching test passed")

        await test_openrouter_fallback()
        print("✓ OpenRouter fallback test passed")

        test_add_provider_function()
        print("✓ Add provider function test passed")

        await test_model_not_found_error()
        print("✓ Model not found error test passed")

        print("\nAll provider routing tests passed!")

    asyncio.run(main())
