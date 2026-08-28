"""
llm.py

Handles configuration and initialization of the Groq-hosted LLM used to
generate answers. Reads configuration from environment variables (.env)
and never hardcodes or prints the API key.
"""

from __future__ import annotations

import os
from typing import Optional

import httpx
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

DEFAULT_MODEL_NAME = "openai/gpt-oss-120b"


class LLMConfigError(Exception):
    """Raised when the LLM cannot be initialized due to missing/invalid config."""


def get_groq_api_key() -> Optional[str]:
    """Return the Groq API key from the environment, or None if unset."""
    return os.getenv("GROQ_API_KEY")


def get_configured_model_name() -> str:
    """Return the configured model name from the environment, or the default."""
    return os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)


def get_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: Optional[int] = 1024,
) -> ChatGroq:
    """
    Create a ChatGroq LLM client using configuration from the environment.

    Args:
        model_name: Optional override for the model name. If not provided,
            falls back to MODEL_NAME from .env, then the default model.
        temperature: Sampling temperature for the LLM.
        max_tokens: Maximum tokens to generate in the response.

    Returns:
        A configured ChatGroq instance.

    Raises:
        LLMConfigError: If the GROQ_API_KEY is missing.
    """
    api_key = get_groq_api_key()
    if not api_key:
        raise LLMConfigError(
            "Groq API key is not configured. Please add GROQ_API_KEY to your .env file."
        )

    resolved_model = model_name or get_configured_model_name()

    # Configure custom httpx client to bypass local Windows SSL CA inspection issues if needed
    verify_ssl = os.getenv("GROQ_SSL_VERIFY", "false").lower() not in ("false", "0", "no")
    http_client = httpx.Client(verify=verify_ssl)

    return ChatGroq(
        api_key=api_key,
        model=resolved_model,
        temperature=temperature,
        max_tokens=max_tokens,
        http_client=http_client,
    )

