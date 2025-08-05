import asyncio
import os
import base64
import time
from pydantic import BaseModel
from typing import List, Optional, Tuple
from localrouter.dtypes import (
    ChatMessage,
    MessageRole,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    ImageBlock,
    ToolDefinition,
)
from localrouter.llm import (
    get_response,
    get_response_with_backoff,
    get_response_cached,
    get_response_cached_with_backoff,
)


os.environ.pop("GOOGLE_API_KEY", None)
MAX_TOKENS = 4000


# Test Pydantic model for structured output
class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: List[str]


class WeatherInfo(BaseModel):
    location: str
    temperature: int
    condition: str
    humidity: Optional[int] = None


# Simple tool definition for testing
def get_weather(location: str) -> str:
    """Get weather information for a location"""
    return f"The weather in {location} is sunny and 72°F"


weather_tool = ToolDefinition(
    name="get_weather",
    description="Get current weather for a location",
    input_schema={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The location to get weather for",
            }
        },
        "required": ["location"],
    },
)


def get_available_models() -> List[Tuple[str, str]]:
    """Check which API keys exist and return list of (model_name, provider) tuples"""
    models = []

    if os.environ.get("OPENAI_API_KEY"):
        models.append(("gpt-4o-mini", "OpenAI"))

    if os.environ.get("ANTHROPIC_API_KEY"):
        models.append(("claude-3-haiku-20240307", "Anthropic"))

    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        models.append(("gemini-2.5-flash", "Google GenAI"))

    return models


def create_test_image() -> ImageBlock:
    """Create a simple test image (1x1 red pixel PNG)"""
    # This is a base64-encoded 1x1 red pixel PNG
    red_pixel_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    return ImageBlock.from_base64(data=red_pixel_png, media_type="image/png")


async def test_basic_functionality():
    """Test basic text generation with all available models"""
    print("=== Testing Basic Functionality ===")

    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text="How do I ask where is the library in spanish?")],
        )
    ]

    models = get_available_models()
    if not models:
        print("No API keys found. Skipping tests.")
        return

    for model_name, provider in models:
        try:
            response = await get_response(
                model=model_name, messages=messages, tools=None, max_tokens=MAX_TOKENS
            )
            print(
                f"{provider} ({model_name}) basic response: {response.content[0].text}"
            )
        except Exception as e:
            print(f"{provider} ({model_name}) basic error: {e}")


async def test_structured_output():
    """Test structured output with Pydantic models for all available models"""
    print("\n=== Testing Structured Output ===")

    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[
                TextBlock(text="Alice and Bob are going to a science fair on Friday.")
            ],
        )
    ]

    models = get_available_models()
    if not models:
        print("No API keys found. Skipping tests.")
        return

    for model_name, provider in models:
        try:
            response = await get_response(
                model=model_name,
                messages=messages,
                tools=None,
                response_format=CalendarEvent,
                max_tokens=MAX_TOKENS,
            )
            print(f"{provider} ({model_name}) structured response: {response}")
            if hasattr(response, "parsed") and response.parsed:
                print(f"  Parsed event: {response.parsed}")
        except NotImplementedError as e:
            print(f"{provider} ({model_name}) structured (expected): {e}")
        except Exception as e:
            print(f"{provider} ({model_name}) structured error: {e}")


async def test_tool_usage():
    """Test tool usage with all available models"""
    print("\n=== Testing Tool Usage ===")

    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text="What's the weather like in San Francisco?")],
        )
    ]

    tools = [weather_tool]
    models = get_available_models()
    if not models:
        print("No API keys found. Skipping tests.")
        return

    for model_name, provider in models:
        try:
            response = await get_response(
                model=model_name, messages=messages, tools=tools, max_tokens=MAX_TOKENS
            )
            print(f"{provider} ({model_name}) tool response:")
            for block in response.content:
                if isinstance(block, TextBlock):
                    print(f"  Text: {block.text}")
                elif isinstance(block, ToolUseBlock):
                    print(f"  Tool call: {block.name}({block.input})")
        except Exception as e:
            print(f"{provider} ({model_name}) tool error: {e}")


async def test_image_input():
    """Test image input with all available models"""
    print("\n=== Testing Image Input ===")

    test_image = create_test_image()
    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[
                test_image,
                TextBlock(
                    text="What do you see in this image? Please describe it briefly."
                ),
            ],
        )
    ]

    models = get_available_models()
    if not models:
        print("No API keys found. Skipping tests.")
        return

    for model_name, provider in models:
        try:
            response = await get_response(
                model=model_name, messages=messages, tools=None, max_tokens=MAX_TOKENS
            )
            print(
                f"{provider} ({model_name}) image response: {response.content[0].text[:150]}..."
            )
        except Exception as e:
            print(f"{provider} ({model_name}) image error: {e}")


async def test_multimodal_with_tools():
    """Test combining images with tool usage"""
    print("\n=== Testing Multimodal with Tools ===")

    test_image = create_test_image()
    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[
                test_image,
                TextBlock(
                    text="Look at this image and then get the weather for New York."
                ),
            ],
        )
    ]

    tools = [weather_tool]
    models = get_available_models()
    if not models:
        print("No API keys found. Skipping tests.")
        return

    for model_name, provider in models:
        try:
            response = await get_response(
                model=model_name, messages=messages, tools=tools, max_tokens=MAX_TOKENS
            )
            print(f"{provider} ({model_name}) multimodal+tools response:")
            for block in response.content:
                if isinstance(block, TextBlock):
                    print(f"  Text: {block.text[:100]}...")
                elif isinstance(block, ToolUseBlock):
                    print(f"  Tool call: {block.name}({block.input})")
        except Exception as e:
            print(f"{provider} ({model_name}) multimodal+tools error: {e}")


async def test_conversation_flow():
    """Test a multi-turn conversation with tool usage"""
    print("\n=== Testing Conversation Flow ===")

    models = get_available_models()
    if not models:
        print("No API keys found. Skipping tests.")
        return

    for model_name, provider in models:
        try:
            # First message: user asks for weather
            messages = [
                ChatMessage(
                    role=MessageRole.user,
                    content=[TextBlock(text="What's the weather like in Boston?")],
                )
            ]

            # Get first response (should include tool call)
            response1 = await get_response(
                model=model_name,
                messages=messages,
                tools=[weather_tool],
                max_tokens=MAX_TOKENS,
            )

            # Add the assistant's response to conversation
            messages.append(response1)

            # Simulate tool execution and add result
            tool_calls = [
                block for block in response1.content if isinstance(block, ToolUseBlock)
            ]
            if tool_calls:
                tool_call = tool_calls[0]
                tool_result = ToolResultBlock(
                    tool_use_id=tool_call.id,
                    content=[
                        TextBlock(
                            text=get_weather(tool_call.input.get("location", "Boston"))
                        )
                    ],
                )
                messages.append(
                    ChatMessage(role=MessageRole.user, content=[tool_result])
                )

                # Get final response
                response2 = await get_response(
                    model=model_name,
                    messages=messages,
                    tools=[weather_tool],
                    max_tokens=MAX_TOKENS,
                )

                print(f"{provider} ({model_name}) conversation flow:")
                print(f"  Tool call: {tool_call.name}({tool_call.input})")
                print(f"  Final response: {response2.content[0].text[:100]}...")
            else:
                print(f"{provider} ({model_name}) conversation flow: No tool call made")

        except Exception as e:
            print(f"{provider} ({model_name}) conversation flow error: {e}")


async def test_backoff_functionality():
    """Test get_response_with_backoff functionality"""
    print("\n=== Testing Backoff Functionality ===")

    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text="What is 2+2?")],
        )
    ]

    models = get_available_models()
    if not models:
        print("No API keys found. Skipping tests.")
        return

    for model_name, provider in models:
        try:
            start_time = time.time()
            response = await get_response_with_backoff(
                model=model_name, messages=messages, tools=None, max_tokens=MAX_TOKENS
            )
            end_time = time.time()
            print(
                f"{provider} ({model_name}) backoff response: {response.content[0].text} (took {end_time - start_time:.2f}s)"
            )
        except Exception as e:
            print(f"{provider} ({model_name}) backoff error: {e}")


async def test_cached_functionality():
    """Test get_response_cached functionality"""
    print("\n=== Testing Cached Functionality ===")

    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text="What is the capital of France?")],
        )
    ]

    models = get_available_models()
    if not models:
        print("No API keys found. Skipping tests.")
        return

    # Test with a fixed cache seed
    cache_seed = 12345

    for model_name, provider in models:
        try:
            # First call - should make API request
            print(f"{provider} ({model_name}) - First cached call...")
            start_time = time.time()
            response1 = await get_response_cached(
                model=model_name,
                messages=messages,
                tools=None,
                cache_seed=cache_seed,
                max_tokens=MAX_TOKENS,
            )
            first_call_time = time.time() - start_time

            # Second call - should use cache
            print(f"{provider} ({model_name}) - Second cached call...")
            start_time = time.time()
            response2 = await get_response_cached(
                model=model_name,
                messages=messages,
                tools=None,
                cache_seed=cache_seed,
                max_tokens=MAX_TOKENS,
            )
            second_call_time = time.time() - start_time

            print(
                f"{provider} ({model_name}) cached response: {response1.content[0].text}"
            )
            print(
                f"  First call: {first_call_time:.2f}s, Second call: {second_call_time:.2f}s"
            )
            print(
                f"  Responses match: {response1.content[0].text == response2.content[0].text}"
            )

        except Exception as e:
            print(f"{provider} ({model_name}) cached error: {e}")


async def test_cached_with_backoff_functionality():
    """Test get_response_cached_with_backoff functionality"""
    print("\n=== Testing Cached with Backoff Functionality ===")

    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text="What is the largest planet in our solar system?")],
        )
    ]

    models = get_available_models()
    if not models:
        print("No API keys found. Skipping tests.")
        return

    # Test with a different cache seed
    cache_seed = 54321

    for model_name, provider in models:
        try:
            # First call - should make API request with backoff protection
            print(f"{provider} ({model_name}) - First cached+backoff call...")
            start_time = time.time()
            response1 = await get_response_cached_with_backoff(
                model=model_name,
                messages=messages,
                tools=None,
                cache_seed=cache_seed,
                max_tokens=MAX_TOKENS,
            )
            first_call_time = time.time() - start_time

            # Second call - should use cache
            print(f"{provider} ({model_name}) - Second cached+backoff call...")
            start_time = time.time()
            response2 = await get_response_cached_with_backoff(
                model=model_name,
                messages=messages,
                tools=None,
                cache_seed=cache_seed,
                max_tokens=MAX_TOKENS,
            )
            second_call_time = time.time() - start_time

            print(
                f"{provider} ({model_name}) cached+backoff response: {response1.content[0].text}"
            )
            print(
                f"  First call: {first_call_time:.2f}s, Second call: {second_call_time:.2f}s"
            )
            print(
                f"  Responses match: {response1.content[0].text == response2.content[0].text}"
            )

        except Exception as e:
            print(f"{provider} ({model_name}) cached+backoff error: {e}")


async def test_cached_with_structured_output():
    """Test cached functionality with structured output"""
    print("\n=== Testing Cached with Structured Output ===")

    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[
                TextBlock(
                    text="Create an event for a team meeting on Monday with John and Sarah."
                )
            ],
        )
    ]

    models = get_available_models()
    if not models:
        print("No API keys found. Skipping tests.")
        return

    cache_seed = 98765

    for model_name, provider in models:
        try:
            # First call
            response1 = await get_response_cached(
                model=model_name,
                messages=messages,
                tools=None,
                response_format=CalendarEvent,
                cache_seed=cache_seed,
                max_tokens=MAX_TOKENS,
            )

            # Second call - should use cache
            response2 = await get_response_cached(
                model=model_name,
                messages=messages,
                tools=None,
                response_format=CalendarEvent,
                cache_seed=cache_seed,
                max_tokens=MAX_TOKENS,
            )

            print(f"{provider} ({model_name}) cached structured response:")
            if hasattr(response1, "parsed") and response1.parsed:
                print(f"  Parsed event: {response1.parsed}")
            print(
                f"  Responses match: {str(response1.content) == str(response2.content)}"
            )

        except NotImplementedError as e:
            print(f"{provider} ({model_name}) cached structured (expected): {e}")
        except Exception as e:
            print(f"{provider} ({model_name}) cached structured error: {e}")


async def test_cached_with_tools():
    """Test cached functionality with tools"""
    print("\n=== Testing Cached with Tools ===")

    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text="What's the weather like in Miami?")],
        )
    ]

    tools = [weather_tool]
    models = get_available_models()
    if not models:
        print("No API keys found. Skipping tests.")
        return

    cache_seed = 13579

    for model_name, provider in models:
        try:
            # First call
            start_time = time.time()
            response1 = await get_response_cached_with_backoff(
                model=model_name,
                messages=messages,
                tools=tools,
                cache_seed=cache_seed,
                max_tokens=MAX_TOKENS,
            )
            first_call_time = time.time() - start_time

            # Second call - should use cache
            start_time = time.time()
            response2 = await get_response_cached_with_backoff(
                model=model_name,
                messages=messages,
                tools=tools,
                cache_seed=cache_seed,
                max_tokens=MAX_TOKENS,
            )
            second_call_time = time.time() - start_time

            print(f"{provider} ({model_name}) cached tools response:")
            for block in response1.content:
                if isinstance(block, TextBlock):
                    print(f"  Text: {block.text[:100]}...")
                elif isinstance(block, ToolUseBlock):
                    print(f"  Tool call: {block.name}({block.input})")

            print(
                f"  First call: {first_call_time:.2f}s, Second call: {second_call_time:.2f}s"
            )
            print(
                f"  Responses match: {str(response1.content) == str(response2.content)}"
            )

        except Exception as e:
            print(f"{provider} ({model_name}) cached tools error: {e}")


async def test_cache_seed_isolation():
    """Test that different cache seeds produce independent cached results"""
    print("\n=== Testing Cache Seed Isolation ===")

    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text="Tell me a random fact.")],
        )
    ]

    models = get_available_models()
    if not models:
        print("No API keys found. Skipping tests.")
        return

    # Use different cache seeds
    cache_seed_1 = 11111
    cache_seed_2 = 22222

    for model_name, provider in models:
        try:
            # Call with first cache seed
            response1 = await get_response_cached(
                model=model_name,
                messages=messages,
                tools=None,
                cache_seed=cache_seed_1,
                max_tokens=MAX_TOKENS,
            )

            # Call with second cache seed (should not use cache from first call)
            start_time = time.time()
            response2 = await get_response_cached(
                model=model_name,
                messages=messages,
                tools=None,
                cache_seed=cache_seed_2,
                max_tokens=MAX_TOKENS,
            )
            second_call_time = time.time() - start_time

            # Call again with first cache seed (should use cache)
            start_time = time.time()
            response3 = await get_response_cached(
                model=model_name,
                messages=messages,
                tools=None,
                cache_seed=cache_seed_1,
                max_tokens=MAX_TOKENS,
            )
            third_call_time = time.time() - start_time

            print(f"{provider} ({model_name}) cache seed isolation:")
            print(f"  Seed 1 response: {response1.content[0].text[:50]}...")
            print(f"  Seed 2 response: {response2.content[0].text[:50]}...")
            print(f"  Seed 1 again: {response3.content[0].text[:50]}...")
            print(f"  Second call (different seed) took: {second_call_time:.2f}s")
            print(f"  Third call (same seed as first) took: {third_call_time:.2f}s")
            print(
                f"  Seed 1 responses match: {response1.content[0].text == response3.content[0].text}"
            )

        except Exception as e:
            print(f"{provider} ({model_name}) cache seed isolation error: {e}")


async def test_cache_seed_optional_behavior():
    """Test that cached functions work without cache_seed (no caching occurs)"""
    print("\n=== Testing Cache Seed Optional Behavior ===")

    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text="What is 1+1?")],
        )
    ]

    models = get_available_models()
    if not models:
        print("No API keys found. Skipping tests.")
        return

    model_name, provider = models[0]  # Just test with first available model

    try:
        # This should work but not use caching (no cache_seed provided)
        start_time = time.time()
        response1 = await get_response_cached(
            model=model_name,
            messages=messages,
            tools=None,
            max_tokens=MAX_TOKENS,
        )
        first_call_time = time.time() - start_time

        # Second call should also work and not use cache
        start_time = time.time()
        response2 = await get_response_cached(
            model=model_name,
            messages=messages,
            tools=None,
            max_tokens=MAX_TOKENS,
        )
        second_call_time = time.time() - start_time

        print(f"{provider} ({model_name}) no cache_seed behavior:")
        print(f"  Response 1: {response1.content[0].text}")
        print(f"  Response 2: {response2.content[0].text}")
        print(
            f"  First call: {first_call_time:.2f}s, Second call: {second_call_time:.2f}s"
        )
        print(
            f"  Both calls took similar time (no caching): {abs(first_call_time - second_call_time) < 2.0}"
        )

    except Exception as e:
        print(f"{provider} ({model_name}) no cache_seed error: {e}")

    try:
        # Test the same with backoff version
        start_time = time.time()
        response3 = await get_response_cached_with_backoff(
            model=model_name,
            messages=messages,
            tools=None,
            max_tokens=MAX_TOKENS,
        )
        third_call_time = time.time() - start_time

        print(f"  Backoff version response: {response3.content[0].text}")
        print(f"  Third call (backoff): {third_call_time:.2f}s")

    except Exception as e:
        print(f"{provider} ({model_name}) no cache_seed backoff error: {e}")


async def main():
    """Run all tests"""
    models = get_available_models()
    print(
        f"Found API keys for: {', '.join([f'{provider} ({model})' for model, provider in models])}"
    )

    if not models:
        print(
            "No API keys found. Please set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY/GOOGLE_API_KEY"
        )
        return

    await test_basic_functionality()
    await test_structured_output()
    await test_tool_usage()
    await test_image_input()
    await test_multimodal_with_tools()
    await test_conversation_flow()

    # New tests for cached and backoff functionality
    await test_backoff_functionality()
    await test_cached_functionality()
    await test_cached_with_backoff_functionality()
    await test_cached_with_structured_output()
    await test_cached_with_tools()
    await test_cache_seed_isolation()
    await test_cache_seed_optional_behavior()


if __name__ == "__main__":
    asyncio.run(main())
