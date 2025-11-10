import pytest
from unittest.mock import AsyncMock
from localrouter import register_router, get_response, ChatMessage, MessageRole, TextBlock
from localrouter.llm import routers, providers, Provider


@pytest.mark.asyncio
async def test_register_router_basic():
    """Test basic router registration and model aliasing"""

    # Mock response
    mock_response = ChatMessage(
        role=MessageRole.assistant, content=[TextBlock(text="Test response")]
    )

    # Create a mock provider that supports "gpt-5"
    mock_provider_func = AsyncMock(return_value=mock_response)

    # Save original state
    original_routers = routers.copy()
    original_providers = providers.copy()
    routers.clear()
    providers.clear()

    try:
        # Add a provider that supports gpt-5
        providers.append(Provider(mock_provider_func, models=["gpt-5"], priority=10))

        # Register a router that maps "default" to "gpt-5"
        def default_router(req):
            if req["model"] == "default":
                return "gpt-5"
            return None

        register_router(default_router)

        # Make request with "default" model
        response = await get_response(
            model="default",
            messages=[
                ChatMessage(role=MessageRole.user, content=[TextBlock(text="test")])
            ],
        )

        # Verify the provider was called with the routed model
        assert mock_provider_func.called
        call_kwargs = mock_provider_func.call_args[1]
        assert call_kwargs["model"] == "gpt-5"
        assert response.content[0].text == "Test response"

    finally:
        # Restore original state
        routers.clear()
        routers.extend(original_routers)
        providers.clear()
        providers.extend(original_providers)


@pytest.mark.asyncio
async def test_router_with_images():
    """Test router that routes based on message content (e.g., images)"""

    mock_response = ChatMessage(
        role=MessageRole.assistant, content=[TextBlock(text="Image response")]
    )

    mock_provider_func = AsyncMock(return_value=mock_response)

    # Save original state
    original_routers = routers.copy()
    original_providers = providers.copy()
    routers.clear()
    providers.clear()

    try:
        # Add providers
        providers.append(
            Provider(mock_provider_func, models=["vision-model", "text-model"], priority=10)
        )

        # Register a router that checks for images in messages
        def image_router(req):
            messages = req.get("messages", [])
            for msg in messages:
                for block in msg.content:
                    if hasattr(block, "__class__") and "ImageBlock" in block.__class__.__name__:
                        return "vision-model"
            return None

        register_router(image_router)

        # Test with text-only message
        response = await get_response(
            model="text-model",
            messages=[
                ChatMessage(role=MessageRole.user, content=[TextBlock(text="test")])
            ],
        )

        # Should use text-model (no routing)
        call_kwargs = mock_provider_func.call_args[1]
        assert call_kwargs["model"] == "text-model"

    finally:
        # Restore original state
        routers.clear()
        routers.extend(original_routers)
        providers.clear()
        providers.extend(original_providers)


@pytest.mark.asyncio
async def test_multiple_routers():
    """Test that multiple routers can be registered and applied in order"""

    mock_response = ChatMessage(
        role=MessageRole.assistant, content=[TextBlock(text="Test response")]
    )

    mock_provider_func = AsyncMock(return_value=mock_response)

    # Save original state
    original_routers = routers.copy()
    original_providers = providers.copy()
    routers.clear()
    providers.clear()

    try:
        # Add provider
        providers.append(
            Provider(mock_provider_func, models=["final-model"], priority=10)
        )

        # Register first router
        def router1(req):
            if req["model"] == "alias1":
                return "alias2"
            return None

        # Register second router
        def router2(req):
            if req["model"] == "alias2":
                return "final-model"
            return None

        register_router(router1)
        register_router(router2)

        # Make request with "alias1"
        response = await get_response(
            model="alias1",
            messages=[
                ChatMessage(role=MessageRole.user, content=[TextBlock(text="test")])
            ],
        )

        # Should be routed through both routers to "final-model"
        call_kwargs = mock_provider_func.call_args[1]
        assert call_kwargs["model"] == "final-model"

    finally:
        # Restore original state
        routers.clear()
        routers.extend(original_routers)
        providers.clear()
        providers.extend(original_providers)


@pytest.mark.asyncio
async def test_router_with_parameters():
    """Test router that has access to all request parameters"""

    mock_response = ChatMessage(
        role=MessageRole.assistant, content=[TextBlock(text="Test response")]
    )

    mock_provider_func = AsyncMock(return_value=mock_response)

    # Save original state
    original_routers = routers.copy()
    original_providers = providers.copy()
    routers.clear()
    providers.clear()

    try:
        # Add providers
        providers.append(
            Provider(
                mock_provider_func,
                models=["cheap-model", "expensive-model"],
                priority=10,
            )
        )

        # Register router that checks temperature parameter
        def temperature_router(req):
            temperature = req.get("temperature", 0)
            if temperature > 0.8:
                return "expensive-model"
            return "cheap-model"

        register_router(temperature_router)

        # Test with high temperature
        response = await get_response(
            model="auto",
            messages=[
                ChatMessage(role=MessageRole.user, content=[TextBlock(text="test")])
            ],
            temperature=0.9,
        )

        call_kwargs = mock_provider_func.call_args[1]
        assert call_kwargs["model"] == "expensive-model"

        # Test with low temperature
        mock_provider_func.reset_mock()
        response = await get_response(
            model="auto",
            messages=[
                ChatMessage(role=MessageRole.user, content=[TextBlock(text="test")])
            ],
            temperature=0.3,
        )

        call_kwargs = mock_provider_func.call_args[1]
        assert call_kwargs["model"] == "cheap-model"

    finally:
        # Restore original state
        routers.clear()
        routers.extend(original_routers)
        providers.clear()
        providers.extend(original_providers)


def test_register_router_function():
    """Test that register_router adds routers to the list"""

    # Save original routers
    original_count = len(routers)

    try:
        # Define a simple router
        def my_router(req):
            return None

        # Register it
        register_router(my_router)

        # Verify it was added
        assert len(routers) == original_count + 1
        assert routers[-1] == my_router

    finally:
        # Remove our test router
        if len(routers) > original_count:
            routers.pop()


if __name__ == "__main__":
    import asyncio

    async def main():
        await test_register_router_basic()
        print("✓ Basic router registration test passed")

        await test_router_with_images()
        print("✓ Router with images test passed")

        await test_multiple_routers()
        print("✓ Multiple routers test passed")

        await test_router_with_parameters()
        print("✓ Router with parameters test passed")

        test_register_router_function()
        print("✓ Register router function test passed")

        print("\nAll router registration tests passed!")

    asyncio.run(main())
