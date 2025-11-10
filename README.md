# LocalRouter

A unified multi-provider LLM client with consistent message formats and tool support across OpenAI, Anthropic, and Google GenAI.


## Quick Start

Install the package:
```bash
pip install localrouter
```

Set your API keys as environment variables:
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key" 
export GEMINI_API_KEY="your-gemini-key"  # or GOOGLE_API_KEY
```

Basic usage:
```python
import asyncio
from localrouter import get_response, ChatMessage, MessageRole, TextBlock

async def main():
    messages = [
        ChatMessage(
            role=MessageRole.user, 
            content=[TextBlock(text="Hello, how are you?")]
        )
    ]
    
    response = await get_response(
        model="gpt-4.1",  # or "o3", "claude-sonnet-4-20250514", "gemini-2.5-pro", etc
        messages=messages
    )
    
    print(response.content[0].text)

asyncio.run(main())
```

## Alternative Response Functions

LocalRouter provides several variants of `get_response` for different use cases:

### Caching
To use disk caching, `import get_response_cached as get_response`:
```python
# Import as get_response for consistent usage
from localrouter import get_response_cached as get_response

response = await get_response(
    model="gpt-4o-mini",
    messages=messages,
    cache_seed=12345  # Required for caching
)
```
This will return cached results whenever get_response is called with identical inputs and `cache_seed` is provided. If no `cache_seed` is provided, it will behave exactly like `localrouter.get_response`.

### Retry with Backoff
Automatically retry failed requests with exponential backoff:
```python
from localrouter import get_response_with_backoff as get_response

response = await get_response(
    model="gpt-4o-mini", 
    messages=messages
)
```

### Caching + Backoff
Combine caching with retry logic:
```python
from localrouter import get_response_cached_with_backoff as get_response

response = await get_response(
    model="gpt-4o-mini",
    messages=messages,
    cache_seed=12345  # Required for caching
)
```

**Note**: When using cached functions without `cache_seed`, they behave like non-cached versions (no caching occurs).

## Images

```python
from localrouter import ChatMessage, MessageRole, TextBlock, ImageBlock

# Text message
text_msg = ChatMessage(
    role=MessageRole.user,
    content=[TextBlock(text="Hello world")]
)
# Image message  
image_msg = ChatMessage(
    role=MessageRole.user,
    content=[
        ImageBlock.from_base64(base64_data, media_type="image/png"), # or: ImageBlock.from_file("image.png")
        TextBlock(text="What's in this image?")
    ]
)
```

## Tool Calling

Define tools and get structured function calls:

```python
from localrouter import ToolDefinition, get_response

# Define a tool
weather_tool = ToolDefinition(
    name="get_weather",
    description="Get current weather for a location",
    input_schema={
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"}
        },
        "required": ["location"]
    }
)

# Use the tool
response = await get_response(
    model="gpt-4.1-nano",
    messages=[ChatMessage(
        role=MessageRole.user,
        content=[TextBlock(text="What's the weather in Paris?")]
    )],
    tools=[weather_tool]
)

# Check for tool calls
for block in response.content:
    if isinstance(block, ToolUseBlock):
        print(f"Tool: {block.name}, Args: {block.input}")
```

## Structured Output

Get validated Pydantic models as responses:

```python
from pydantic import BaseModel
from typing import List

class Event(BaseModel):
    name: str
    date: str
    participants: List[str]

response = await get_response(
    model="gpt-4.1-mini",
    messages=[ChatMessage(
        role=MessageRole.user,
        content=[TextBlock(text="Alice and Bob meet for lunch Friday")]
    )],
    response_format=Event
)

event = response.parsed  # Validated Event instance
print(f"Event: {event.name} on {event.date}")
```

### Conversation Flow

Handle multi-turn conversations with tool results:

```python
from localrouter import ToolResultBlock

# Initial request
messages = [ChatMessage(
    role=MessageRole.user,
    content=[TextBlock(text="Get weather for Tokyo")]
)]

# Get response with tool call
response = await get_response(model="gpt-4o-mini", messages=messages, tools=[weather_tool])
messages.append(response)

# Execute tool and add result
tool_call = response.content[0]  # ToolUseBlock
tool_result = ToolResultBlock(
    tool_use_id=tool_call.id,
    content=[TextBlock(text="Tokyo: 22°C, sunny")] # Tool result may also contain ImageBlock parts
)
messages.append(ChatMessage(role=MessageRole.user, content=[tool_result]))

# Continue conversation
final_response = await get_response(model="gpt-4o-mini", messages=messages, tools=[weather_tool])
```

### Tool Definition

- `ToolDefinition(name, description, input_schema)` - Define available tools
- `SubagentToolDefinition()` - Predefined tool for sub-agents

## Reasoning/Thinking Support

Configure reasoning budgets for models that support explicit thinking (GPT-5, Claude Sonnet 4+, Gemini 2.5):

```python
from localrouter import ReasoningConfig

# Using effort levels (OpenAI-style)
response = await get_response(
    model="gpt-5",  # When available
    messages=messages,
    reasoning=ReasoningConfig(effort="high")  # "minimal", "low", "medium", "high"
)

# Using explicit token budget (Anthropic/Gemini-style)
response = await get_response(
    model="gemini-2.5-pro",
    messages=messages,
    reasoning=ReasoningConfig(budget_tokens=8000)
)

# Let model decide (Gemini dynamic thinking)
response = await get_response(
    model="gemini-2.5-flash",
    messages=messages,
    reasoning=ReasoningConfig(dynamic=True)
)

# Backward compatible dict config
response = await get_response(
    model="claude-sonnet-4-20250514",  # When available
    messages=messages,
    reasoning={"effort": "medium"}
)
```

The reasoning configuration automatically converts between provider formats:
- **OpenAI (GPT-5)**: Uses `effort` levels
- **Anthropic (Claude 4+)**: Uses `budget_tokens` 
- **Google (Gemini 2.5)**: Uses `thinking_budget` with dynamic option

Models that don't support reasoning will ignore the configuration.

## Custom Providers and Model Routing

LocalRouter supports regex patterns for model matching and prioritized provider selection. OpenRouter serves as a fallback for any model containing "/" (e.g., "meta-llama/llama-3.3-70b") with lowest priority.

```python
from localrouter import add_provider, re

# Add a custom provider with regex pattern support
async def custom_get_response(model, messages, **kwargs):
    # Your custom implementation
    pass

add_provider(
    custom_get_response,
    models=["custom-model-1", re.compile(r"custom-.*")],  # Exact match or regex
    priority=50  # Lower = higher priority (default: 100, OpenRouter: 1000)
)
```

## Request-Level Routing

LocalRouter allows you to register router functions that can dynamically modify model selection based on request parameters. This is useful for:
- Creating model aliases
- Routing requests with images to vision models
- Selecting models based on temperature, tools, or other parameters
- Implementing fallback strategies

```python
from localrouter import register_router

# Example 1: Simple alias
def alias_router(req):
    if req['model'] == 'default':
        return 'gpt-5'
    return None  # Keep original model

register_router(alias_router)

# Now you can use the alias
response = await get_response(
    model="default",  # Will be routed to gpt-5
    messages=messages
)
```

```python
# Example 2: Route based on message content
def vision_router(req):
    """Route requests with images to vision-capable models"""
    messages = req.get('messages', [])
    for msg in messages:
        for block in msg.content:
            if hasattr(block, '__class__') and 'ImageBlock' in block.__class__.__name__:
                return 'qwen/qwen3-vl-30b-a3b-instruct'
    return None  # Use original model for text-only requests

register_router(vision_router)
```

```python
# Example 3: Route based on parameters
def temperature_router(req):
    """Use different models based on temperature"""
    temperature = req.get('temperature', 0)
    if temperature > 0.8:
        return 'gpt-5'  # Creative tasks
    return 'gpt-4.1-mini'  # Deterministic tasks

register_router(temperature_router)
```

**Router Function Interface:**
- **Input**: Dictionary with keys: `model`, `messages`, `tools`, `response_format`, `reasoning`, and any other kwargs
- **Output**: String (new model name) or None (keep original model)
- **Execution**: Routers are applied in registration order, and each router sees the model name from the previous router
