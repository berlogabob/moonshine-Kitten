import ollama

from .ollama_models import OLLAMA_MODEL
from .ollama_settings import EMPTY_REPLY_FALLBACK, OLLAMA_OPTIONS
from .text_utils import clean_text


def generate_assistant_reply(chat_history, model: str = OLLAMA_MODEL) -> str:
    response = ollama.chat(
        model=model,
        messages=chat_history,
        options=OLLAMA_OPTIONS,
    )

    message = response.get("message") or {}
    content = message.get("content")
    if content is None:
        content = ""
    reply_text = content.strip()
    if not reply_text:
        print("AI returned empty content, using fallback reply.")
        reply_text = EMPTY_REPLY_FALLBACK

    clean_resp = clean_text(reply_text)
    if not clean_resp:
        clean_resp = EMPTY_REPLY_FALLBACK
    return clean_resp

