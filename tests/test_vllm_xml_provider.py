import os
import re
import json
import importlib
from typing import List, Dict, Any
from pydantic import BaseModel

from localrouter.dtypes import (
    ChatMessage,
    MessageRole,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    ToolDefinition,
)

# New module we will implement
import localrouter.vllm_xml as vllm_xml


def make_weather_tool() -> ToolDefinition:
    return ToolDefinition(
        name="get_weather",
        description="Get current weather for a location",
        input_schema={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {
                    "type": "string",
                    "enum": ["c", "f"],
                    "description": "Temperature unit",
                },
            },
            "required": ["location"],
        },
    )


def test_build_system_message_xml():
    tool = make_weather_tool()
    sys_text = vllm_xml.build_system_message_xml([tool])
    assert "Use one of the available tools" in sys_text
    assert "<tool>" in sys_text and "<name> get_weather </name>" in sys_text
    assert "<input_schema>" in sys_text
    assert "Respond in valid XML" in sys_text


def test_render_message_to_xml_string_with_tool_result():
    # Simulate a user sending a tool result back
    tool_result = ToolResultBlock(
        tool_use_id="abc123",
        content=[TextBlock(text="It is sunny in Paris, 22C.")],
    )
    msg = ChatMessage(role=MessageRole.user, content=[tool_result])
    rendered = vllm_xml.render_message_to_xml_string(msg)
    assert "<tool_result>" in rendered
    assert "It is sunny in Paris" in rendered


def test_generate_tool_use_grammar_basic():
    tool = make_weather_tool()
    grammar = vllm_xml.generate_tool_use_grammar([tool])
    # Should restrict to known tool names and structure
    assert "tool_use" in grammar
    assert "get_weather" in grammar
    # Ensure input keys appear in grammar
    assert "location" in grammar and "unit" in grammar


def test_build_vllm_chat_payload():
    tool = make_weather_tool()
    messages = [
        ChatMessage(role=MessageRole.user, content=[TextBlock(text="Weather in SF?")])
    ]
    payload = vllm_xml.build_vllm_chat_payload(messages, [tool], model="unsloth/Qwen3-4B")

    # Verify OpenAI-compatible payload shape
    assert isinstance(payload["messages"], list)
    assert payload["messages"][0]["role"] == "system"
    assert any(m["role"] == "user" for m in payload["messages"])

    # Guided decoding present
    assert "guided_decoding" in payload
    gd = payload["guided_decoding"]
    assert gd["type"] == "grammar"
    assert isinstance(gd["grammar"], str) and len(gd["grammar"]) > 0


def test_parse_vllm_xml_response_to_chatmessage():
    xml = """
    <tool_use>
      <name>get_weather</name>
      <input>
        <location>Paris</location>
        <unit>c</unit>
      </input>
    </tool_use>
    """
    cm = vllm_xml.parse_vllm_xml_response(xml)
    assert isinstance(cm, ChatMessage)
    assert cm.role == MessageRole.assistant
    tool_calls = [b for b in cm.content if isinstance(b, ToolUseBlock)]
    assert tool_calls, "Expected a ToolUseBlock"
    assert tool_calls[0].name == "get_weather"
    assert tool_calls[0].input["location"] == "Paris"
    assert tool_calls[0].input["unit"] == "c"


def test_provider_registration_env_guard(monkeypatch):
    # Ensure provider registration helper produces model patterns safely
    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("VLLM_ALLOWED_MODELS", "unsloth/.+,qwen.*")

    # Reimport module to simulate registration time logic
    import localrouter.llm as llm
    importlib.reload(llm)

    # Expect at least one provider with regex patterns from env
    patterns: List[str] = []
    for p in llm.providers:
        for m in p.models:
            if hasattr(m, "pattern"):
                patterns.append(m.pattern)
    assert any("unsloth/.+" in pat for pat in patterns)
    assert any("qwen" in pat.lower() for pat in patterns)