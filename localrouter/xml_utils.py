def get_first_element(text, element):
    """Finds the first <element> in the text and returns the text before, the element, and the text after."""
    start = text.find(f"<{element}>")
    if start == -1:
        return text, None, ""
    start += len(f"<{element}>")
    tmp = start
    end = text.find(f"</{element}>", start)
    while f"<{element}>" in text[tmp:end]:
        tmp += text[tmp:].find(f"</{element}>") + len(f"</{element}>")
        end = text.find(f"</{element}>", tmp)
        if end == -1:
            return text, None, ""
    if end == -1:
        return text, None, ""
    return (
        text[: start - len(f"<{element}>")],
        text[start:end],
        text[end + len(f"</{element}>") :],
    )


def strip(text, maxleft=0, maxright=0, chars=["\n", " "]):
    """Strip characters from the left and right of a string."""
    left = 0
    while left < maxleft and text[left] in chars:
        left += 1
    right = len(text)
    while right > maxright and text[right - 1] in chars:
        right -= 1
    return text[left:right]


def parse_xml(text, schema):
    """Parse an XML string into a specified schema

    Args:
        text (str): The XML string to parse
        schema (dict): A pydantic_model.model_json_schema() like schema

    Returns:
        dict: The parsed XML string
    """
    if schema["type"] == "string":
        return strip(text, maxleft=1, maxright=1, chars=["\n", " "])
    elif schema["type"] == "number":
        return float(text)
    elif schema["type"] == "integer":
        return int(text)
    elif schema["type"] == "boolean":
        return text.lower().strip() == "true"
    elif schema["type"] == "array":
        parsed = []
        while True:
            _, item, text = get_first_element(text, "item")
            if item:
                parsed.append(parse_xml(item, schema["items"]))
            else:
                break
        return parsed
    elif schema["type"] == "object":
        parsed = {}
        for key, value in schema["properties"].items():
            before, item, after = get_first_element(text, key)
            if item:
                parsed[key] = parse_xml(item, value)
                text = before + after
            else:
                default = schema["properties"][key].get("default")
                if not default and schema["properties"][key].get("required"):
                    raise ValueError(f"Missing required key: {key}")
                else:
                    parsed[key] = default
        return parsed


def dump_xml(*args, indent=0, **data):
    """Pretty print XML from dicts/lists with indentation and CDATA for strings.

    Usage examples:
      dump_xml(tool={"name": "get_weather", "input_schema": {...}})
      dump_xml({"tool": {...}}, {"tool": {...}})
    """
    lines = []

    def cdata(text: str) -> str:
        return f"<![CDATA[{text}]]>"

    def serialize_key_value(key, value, level):
        ind = "  " * level
        if isinstance(value, dict):
            lines.append(f"{ind}<{key}>")
            for ck, cv in value.items():
                serialize_key_value(ck, cv, level + 1)
            lines.append(f"{ind}</{key}>")
        elif isinstance(value, list):
            lines.append(f"{ind}<{key}>")
            for item in value:
                serialize_key_value("item", item, level + 1)
            lines.append(f"{ind}</{key}>")
        elif value is None:
            lines.append(f"{ind}<{key}></{key}>")
        else:
            # Wrap string content in CDATA to allow XML-in-payload safely
            text = cdata(str(value))
            lines.append(f"{ind}<{key}>{text}</{key}>")

    # Positional args may contain dicts or lists
    for arg in args:
        if isinstance(arg, dict):
            for k, v in arg.items():
                serialize_key_value(k, v, indent)
        elif isinstance(arg, list):
            for item in arg:
                if isinstance(item, dict):
                    for k, v in item.items():
                        serialize_key_value(k, v, indent)
                else:
                    serialize_key_value("item", item, indent)
        else:
            # Raw strings: include as-is (already CDATA elsewhere when needed)
            lines.append(("  " * indent) + str(arg))

    # Keyword args
    for key, value in data.items():
        serialize_key_value(key, value, indent)

    return "\n".join(lines)
