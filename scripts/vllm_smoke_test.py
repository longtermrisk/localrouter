import os
import asyncio
from localrouter.dtypes import ChatMessage, MessageRole, TextBlock, ToolDefinition
from localrouter.llm import get_response

async def main():
    model = os.environ.get("VLLM_MODEL", "unsloth/Qwen3-4B")
    os.environ.setdefault("VLLM_ALLOWED_MODELS", ".+/.*")

    tools = [
        ToolDefinition(
            name="echo",
            description="Echo the given text",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )
    ]

    msgs = [
        ChatMessage(role=MessageRole.user, content=[TextBlock(text="Call echo with text=hello")])
    ]

    resp = await get_response(model=model, messages=msgs, tools=tools, max_tokens=256)
    print("Response blocks:")
    for b in resp.content:
        print(b)

if __name__ == "__main__":
    asyncio.run(main())