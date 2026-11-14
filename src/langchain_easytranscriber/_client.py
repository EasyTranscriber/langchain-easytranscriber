"""Shared helpers: API key resolution, client construction and error mapping."""

from __future__ import annotations

import os

import httpx
from easytranscriber_sdk import APIError, APIResponseError, AsyncEasyTranscriber, EasyTranscriber

API_KEY_ENV = "EASYTRANSCRIBER_API_KEY"
DEFAULT_BASE_URL = "https://www.easytranscriber.com/api/v1"
# New transcriptions can take minutes (speech-to-text for videos without captions).
DEFAULT_TIMEOUT = 310.0
KEYS_URL = "https://www.easytranscriber.com/dashboard/api-keys"

# What one API call can raise: an error status, an unexpected response body, or a
# transport failure (connection, timeout) from HTTPX.
REQUEST_ERRORS: tuple[type[Exception], ...] = (APIError, APIResponseError, httpx.HTTPError)


def resolve_api_key(api_key: str | None) -> str:
    """Return the explicit key, else ``EASYTRANSCRIBER_API_KEY``; raise if neither is set."""
    key = api_key if api_key is not None else os.environ.get(API_KEY_ENV, "")
    if not key.strip():
        msg = (
            f"No EasyTranscriber API key. Pass api_key=... or set {API_KEY_ENV}. "
            f"Create a key at {KEYS_URL}"
        )
        raise ValueError(msg)
    return key


def sync_client(
    api_key: str, *, base_url: str, timeout: float, max_retries: int
) -> EasyTranscriber:
    """Build a synchronous SDK client."""
    return EasyTranscriber(api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)


def async_client(
    api_key: str, *, base_url: str, timeout: float, max_retries: int
) -> AsyncEasyTranscriber:
    """Build an asynchronous SDK client."""
    return AsyncEasyTranscriber(
        api_key, base_url=base_url, timeout=timeout, max_retries=max_retries
    )


def describe_error(error: Exception) -> str:
    """Turn a failed API call into a message an agent or developer can act on."""
    if isinstance(error, APIError):
        hint = ""
        if error.code == "insufficient_credits":
            hint = " Check the balance or plan at https://www.easytranscriber.com/pricing"
        elif error.code == "unauthorized":
            hint = f" Check the API key ({KEYS_URL})."
        elif error.code == "rate_limited":
            hint = " The per-key request limit was reached; retry after a short wait."
        message = error.message.rstrip(".")
        return f"EasyTranscriber API error {error.status} ({error.code}): {message}.{hint}"
    if isinstance(error, httpx.TimeoutException):
        return "EasyTranscriber API request timed out. Retry, or raise the timeout setting."
    return f"EasyTranscriber API request failed: {error}"


def watch_url(video_id: str) -> str:
    """Canonical YouTube watch URL for a video ID."""
    return f"https://www.youtube.com/watch?v={video_id}"
