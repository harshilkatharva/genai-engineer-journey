from rag_app.services.redis_history import _message_text


def test_message_text_flattens_provider_content_blocks():
    content = [
        {"type": "text", "text": "First part. "},
        {"type": "image", "url": "ignored"},
        {"type": "text", "text": "Second part."},
    ]

    assert _message_text(content) == "First part. Second part."


def test_message_text_preserves_string_content():
    assert _message_text("A plain assistant response.") == "A plain assistant response."
