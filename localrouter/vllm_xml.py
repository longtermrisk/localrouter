import os
import re
import json
from typing import List, Dict, Any, Optional, Union
import os
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

def _schema_to_definition_xml(schema: Dict[str, Any]) -> Dict[str, Any]:
    t = schema.get("type")
    if t == "object":
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        out: Dict[str, Any] = {}
        for k, v in props.items():
            # Add type label with optional flag and description in value text
            vt = v.get("type", "string")
            opt = "optional " if k not in required else ""
            desc = v.get("description") or ""
            if vt == "object":
                out[k] = _schema_to_definition_xml(v)
            elif vt == "array":
                # Represent array as <k> array: description <item> ... </item> </k>
                item_schema = v.get("items", {"type": "string"})
                out[k] = {
                    "type": f"{opt}array: {desc}",
                    "item": _schema_to_definition_xml(item_schema).get(
                        "item", _schema_to_definition_xml(item_schema)
                    ),
                }
            else:
                out[k] = f"{opt}{vt}: {desc}".strip()
        return out
    elif t == "array":
        item_schema = schema.get("items", {"type": "string"})
        return {"item": _schema_to_definition_xml(item_schema)}
    else:
        desc = schema.get("description") or ""
        return {"value": f"{t or 'string'}: {desc}"}


def build_system_message_xml(tools: Optional[List[ToolDefinition]]) -> str:
    if not tools:
        return (
            "You can converse normally. If tools are provided later, you will be given XML usage instructions."
        )

    tool_defs = []
    for t in tools:
        definition = {
            "tool": {
                "name": t.name,
                "input": _schema_to_definition_xml(t.input_schema),
            }
        }
        tool_defs.append(definition)

    header = (
        "Use one of the available tools to choose an action.\n"
        "Definition schema mirrors the tool_use XML. Fill values in a tool_use block.\n"
        "Respond in valid XML to call a tool."
    )
    example = dump_xml(
        tool_use={
            "name": "tool_name",
            "input": {"key": "value"},
        }
    )
    return header + "\n\n" + dump_xml(*tool_defs) + "\n\nExample:\n\n" + example


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
    # Recursive grammar for arbitrarily nested objects/arrays
    # We derive key set from schemas but allow recursive nesting via VALUE := ELEMENT | ARRAY | STRING
    keys = set()
    def collect_keys(schema: Dict[str, Any]):
        t = schema.get('type')
        if t == 'object':
            props = schema.get('properties', {})
            for k, v in props.items():
                keys.add(k)
                collect_keys(v)
        elif t == 'array':
            collect_keys(schema.get('items', {'type': 'string'}))
    for t in tools:
        collect_keys(t.input_schema or {})

    key_alt = " | ".join([f"'{k}'" for k in sorted(keys)]) if keys else "'key'"

    grammar = [
        "tool_use := '<tool_use>' name input '</tool_use>'",
        "name := '<name>' (" + name_alt + ") '</name>'",
        "input := '<input>' (pair)+ '</input>'",
        "pair := '<' key '>' VALUE '</' key '>'",
        "key := " + key_alt,
        "VALUE := OBJECT | ARRAY | CDATA | TEXT",
        "OBJECT := '<input>' (pair)+ '</input>'",
        "ARRAY := '<list>' (VALUE)+ '</list>'",
        r"CDATA := '<![CDATA[' TEXT ']]>'",
        r"TEXT := /[^<][^<]*/",
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
    """Parse a vLLM XML tool_use response into a ChatMessage with ToolUseBlock.

    Supports nested objects via nested <input> blocks and arrays via <list> ... </list>
    """
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

    def _strip_cdata(val: str) -> str:
        val = val.strip()
        if val.startswith("<![CDATA[") and val.endswith("]]>"):
            return val[len("<![CDATA[") : -3]
        return val

    def parse_value(t: str):
        t = t.strip()
        if t.startswith("<input>"):
            inner = _find("input", t) or ""
            return parse_object(inner)
        if t.startswith("<list>"):
            items = _find("list", t) or ""
            arr = []
            p = 0
            while True:
                lt = items.find("<", p)
                if lt == -1:
                    break
                gt = items.find(">", lt + 1)
                if gt == -1:
                    break
                tag = items[lt + 1 : gt]
                if tag.startswith("/"):
                    p = gt + 1
                    continue
                end = items.find(f"</{tag}>", gt + 1)
                if end == -1:
                    break
                val = items[gt + 1 : end]
                arr.append(parse_value(val))
                p = end + len(f"</{tag}>")
            return arr
        return _strip_cdata(t)

    def parse_object(text: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        pos = 0
        while True:
            lt = text.find("<", pos)
            if lt == -1:
                break
            gt = text.find(">", lt + 1)
            if gt == -1:
                break
            key = text[lt + 1 : gt]
            if "/" in key or " " in key:
                pos = gt + 1
                continue
            close_tag = f"</{key}>"
            close = text.find(close_tag, gt + 1)
            if close == -1:
                pos = gt + 1
                continue
            value = text[gt + 1 : close]
            result[key] = parse_value(value)
            pos = close + len(close_tag)
        return result

    input_body = _find("input", tu_body) or ""
    inputs = parse_object(input_body)

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