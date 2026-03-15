from types import SimpleNamespace

from localrouter import ChatMessage, ImageBlock, MessageRole, TextBlock


def test_from_openai_parses_refusal_only_response():
    message = SimpleNamespace(
        content=None,
        refusal="I can’t help with that request.",
        tool_calls=None,
    )

    parsed = ChatMessage.from_openai(message)

    assert len(parsed.content) == 1
    assert isinstance(parsed.content[0], TextBlock)
    assert parsed.content[0].text == "I can’t help with that request."
    assert parsed.meta["refusal"] is True


def test_from_openai_parses_object_multimodal_parts():
    message = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="A small red pixel."),
            SimpleNamespace(
                type="image_url",
                image_url=SimpleNamespace(
                    url="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
                ),
            ),
        ],
        refusal=None,
        tool_calls=None,
    )

    parsed = ChatMessage.from_openai(message)

    assert len(parsed.content) == 2
    assert isinstance(parsed.content[0], TextBlock)
    assert parsed.content[0].text == "A small red pixel."
    assert isinstance(parsed.content[1], ImageBlock)


def test_image_block_openai_format_respects_detail_meta():
    image = ImageBlock.from_base64(
        data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
        media_type="image/png",
        meta={"openai_detail": "low"},
    )

    formatted = ChatMessage(
        role=MessageRole.user,
        content=[TextBlock(text="Describe the image"), image],
    ).openai_format()

    assert formatted["content"][1]["image_url"]["detail"] == "low"
