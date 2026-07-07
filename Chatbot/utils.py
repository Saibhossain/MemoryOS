"""
Chatbot/utils.py

Small helpers: turning an uploaded Streamlit image into the multimodal
message format LangChain/Ollama expect, and pulling display text back
out of a message whose content might be plain text or a content-block list.
"""
import base64
from langchain_core.messages import HumanMessage


def build_human_message(text: str, uploaded_image=None) -> HumanMessage:
    """
    uploaded_image: a Streamlit UploadedFile, or None.

    Returns a HumanMessage whose content is either:
      - a plain string (text-only turn), or
      - a list of content blocks [text, image_url] (multimodal turn).

    IMPORTANT: only vision-capable Ollama models (e.g. qwen2.5vl, llava,
    bakllava, minicpm-v) will actually *look at* the image. A text-only
    model like a plain qwen2.5:8b will typically just ignore the image
    block, or the call may error depending on the Ollama version. If you
    hit that, either `ollama pull qwen2.5vl` or treat image upload as
    "attach for display only" for now - the image will still be saved
    correctly into the Postgres checkpoint either way.
    """
    if uploaded_image is None:
        return HumanMessage(content=text)

    image_bytes = uploaded_image.getvalue()
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    mime = uploaded_image.type or "image/png"

    content = [
        {"type": "text", "text": text or "Describe this image."},
        {"type": "image_url", "image_url": f"data:{mime};base64,{b64}"},
    ]
    return HumanMessage(content=content)


def message_text(msg) -> str:
    """Extract a display-friendly string from a message, whether its
    content is a plain string or a multimodal content-block list."""
    if isinstance(msg.content, str):
        return msg.content
    parts = []
    for block in msg.content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block["text"])
    return "\n".join(parts) if parts else "[image]"


def message_has_image(msg) -> bool:
    if isinstance(msg.content, str):
        return False
    return any(
        isinstance(b, dict) and b.get("type") == "image_url" for b in msg.content
    )