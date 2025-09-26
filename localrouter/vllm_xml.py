import os
import re
import json
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel

from .dtypes import (
    ChatMessage,
    MessageRole,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    ToolDefinition,
)
from .xml_utils import dump_xml


# -----------------------------
# XML helper construction
# -----------------------------

def build_system_message_xml(tools: Optional[List[ToolDefinition]]) -> str:
    if not tools:
        return (
            "You can converse normally. If tools are provided later, you will be given XML usage instructions."
        )

    tool_defs = []
    for t in tools:
        tool_defs.append(
            {
                "tool": {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": json.dumps(t.input_schema),
                }
            }
        )

    header = (
        "Use one of the available tools to choose an action. "
        "Respond in valid XML to call a tool."
    )
    example = (
        "<tool_use>\n  <name>tool_name</name>\n  <input>\n    <key>value</key>\n  </input>\n</tool_use>\n"
    )
    return header + "\n\n" + dump_xml(*tool_defs).strip() + "\n\nRespond in valid XML in order to call a tool, for example:\n\n" + example


def render_message_to_xml_string(message: ChatMessage) -> str:
    parts: List[str] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            parts.append(block.text)
        elif isinstance(block, ToolResultBlock):
            # Serialize tool result content as simple string text aggregation
            inner_texts = []
            for c in block.content:
                if isinstance(c, TextBlock):
                    inner_texts.append(c.text)
            xml = dump_xml(tool_result={"tool_use_id": block.tool_use_id, "output": "\n".join(inner_texts)})
            parts.append(xml)
        elif isinstance(block, ToolUseBlock):
            # Generally model emits tool_use, but if present in history, include
            xml = dump_xml(tool_use={"name": block.name, "input": json.dumps(block.input or {})})
            parts.append(xml)
        else:
            # Other blocks (images, etc.) are skipped in XML history
            continue
    return "\n".join(parts)


# -----------------------------
# Grammar for tool use
# -----------------------------

def _escape_identifier(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)


def generate_tool_use_grammar(tools: Optional[List[ToolDefinition]]) -> str:
    """Generate a simple EBNF-like grammar string constraining the model to emit a single
    <tool_use>...</tool_use> block with constrained name and key/value string inputs.

    vLLM supports JSON-schema and regex-based grammars; to keep generic, we provide a PEG-like
    grammar string widely used by vLLM guided decoding plugins.
    """
    if not tools:
        # Allow any text (fallback) but still a minimal element
        return (
            "tool_use := '<tool_use>' name input '</tool_use>'\n"
            "name := '<name>' TEXT '</name>'\n"
            "input := '<input>' ANY '</input>'\n"
            "TEXT := /[^<][^<]*/\n"
            "ANY := /[\s\S]*/\n"
        )

    names = [t.name for t in tools]
    # Build alternation for names
    name_alt = " | ".join([f"'{n}'" for n in names])

    # Collect unique keys from schemas (best effort)
    keys = set()
    for t in tools:
        props = (t.input_schema or {}).get("properties", {})
        for k in props.keys():
            keys.add(k)
    key_alt = " | ".join([f"'{k}'" for k in sorted(keys)]) if keys else "'key'"

    grammar = [
        "tool_use := '<tool_use>' name input '</tool_use>'",
        "name := '<name>' (" + name_alt + ") '</name>'",
        "input := '<input>' (kv_pair)+ '</input>'",
        "kv_pair := '<' key '>' TEXT '</' key '>'",
        "key := " + key_alt,
        "TEXT := /[^<][^<]*/",
    ]
    return "\n".join(grammar) + "\n"


# -----------------------------
# Build vLLM chat payload
# -----------------------------

def build_vllm_chat_payload(
    messages: List[ChatMessage],
    tools: Optional[List[ToolDefinition]],
    **kwargs: Any,
) -> Dict[str, Any]:
    # vLLM OpenAI-compatible Chat Completions endpoint generally expects messages
    # We'll inject a system message with tool XML instructions and attach guided decoding

    sys_text = build_system_message_xml(tools)

    # Convert our messages to a single XML-bearing user/assistant history
    oai_msgs: List[Dict[str, Any]] = []

    # Always include the system message first
    oai_msgs.append({"role": "system", "content": sys_text})

    for m in messages:
        if m.role == MessageRole.system:
            # Merge any system content into the first system
            extra = []
            for b in m.content:
                if isinstance(b, TextBlock):
                    extra.append(b.text)
            if extra:
                oai_msgs[0]["content"] += "\n\n" + "\n".join(extra)
            continue
        # Render as plain text containing XML chunks where relevant
        oai_msgs.append({
            "role": m.role.value,
            "content": render_message_to_xml_string(m),
        })

    payload: Dict[str, Any] = {
        "messages": oai_msgs,
    }

    # Attach grammar-guided decoding config
    grammar = generate_tool_use_grammar(tools)
    payload["guided_decoding"] = {"type": "grammar", "grammar": grammar}

    # Pass through common kwargs (model, temperature, max_tokens, etc.)
    for k, v in kwargs.items():
        payload[k] = v

    return payload


# -----------------------------
# Parse model XML back into ChatMessage
# -----------------------------

def parse_vllm_xml_response(xml_text: str) -> ChatMessage:
    """Parse a vLLM XML tool_use response into a ChatMessage with ToolUseBlock."""
    # Very lightweight parse: find <tool_use> with <name> and <input> and parse simple kv pairs
    def _find(tag: str, s: str) -> Optional[str]:
        start = s.find(f"<{tag}>")
        if start == -1:
            return None
        start += len(tag) + 2
        end = s.find(f"</{tag}>", start)
        if end == -1:
            return None
        return s[start:end]

    tu_body = _find("tool_use", xml_text) or xml_text
    name = _find("name", tu_body) or ""

    input_body = _find("input", tu_body) or ""
    # Parse each immediate child tag as a key
    inputs: Dict[str, Any] = {}
    pos = 0
    while True:
        lt = input_body.find("<", pos)
        if lt == -1:
            break
        gt = input_body.find(">", lt + 1)
        if gt == -1:
            break
        key = input_body[lt + 1 : gt]
        if "/" in key or " " in key:
            pos = gt + 1
            continue
        close_tag = f"</{key}>"
        close = input_body.find(close_tag, gt + 1)
        if close == -1:
            pos = gt + 1
            continue
        value = input_body[gt + 1 : close].strip()
        inputs[key] = value
        pos = close + len(close_tag)

    tub = ToolUseBlock(name=name, input=inputs)
    return ChatMessage(role=MessageRole.assistant, content=[tub])


# -----------------------------
# Provider integration helper
# -----------------------------

def get_allowed_model_patterns() -> List[re.Pattern]:
    patterns: List[re.Pattern] = []
    allowed = os.environ.get("VLLM_ALLOWED_MODELS")
    if allowed:
        for part in re.split(r"[,\s]+", allowed.strip()):
            if not part:
                continue
            try:
                patterns.append(re.compile(part, re.IGNORECASE))
            except re.error:
                # Fallback to exact match by escaping
                patterns.append(re.compile(re.escape(part), re.IGNORECASE))
    else:
        # Default to models containing "/" so they won't collide with other providers
        patterns.append(re.compile(r".+/.+", re.IGNORECASE))
    return patterns