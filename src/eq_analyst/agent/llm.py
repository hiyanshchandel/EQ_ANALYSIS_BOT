"""Chat model factory. The only module that knows we use Gemini."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from eq_analyst.core.config import Settings


def build_chat_model(settings: Settings) -> BaseChatModel:
    if settings.google_api_key is None:
        raise RuntimeError("GOOGLE_API_KEY is not set. Add it to your .env file.")
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_seconds,
        max_retries=3,  # retries rate-limit (429) and transient server errors with backoff
    )
