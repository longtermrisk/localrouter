# LocalRouter

A unified multi-provider LLM client with consistent message formats and tool support across OpenAI, Anthropic, and Google GenAI.


## Quick Start

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
Cache responses to disk for faster repeated calls:
```python
# Import as get_response for consistent usage
from localrouter import get_response_cached as get_response

response = await get_response(
    model="gpt-4o-mini",
    messages=messages,
    cache_seed=12345  # Required for caching
)
```

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
