"""Document loader for YouTube transcripts from EasyTranscriber."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from easytranscriber_sdk import AsyncEasyTranscriber, EasyTranscriber
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from langchain_easytranscriber._client import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT,
    REQUEST_ERRORS,
    async_client,
    describe_error,
    resolve_api_key,
    sync_client,
)

logger = logging.getLogger(__name__)


class EasyTranscriberLoader(BaseLoader):
    """Load YouTube transcripts as Documents through the EasyTranscriber API.

    Each URL becomes one ``Document``: ``page_content`` is the transcript text and
    ``metadata`` holds ``source``, ``video_id``, ``title``, ``method``, ``cached``
    and ``credits_used``. With ``include_summary=True`` the metadata also gets an
    AI ``summary`` (Markdown).

    Credits: a cached transcript costs 0, a new one 1 (2 for videos over 2 hours,
    4-hour maximum). Each summary costs 1 more credit.

    Setup:
        ```bash
        pip install -U langchain-easytranscriber
        export EASYTRANSCRIBER_API_KEY="et_..."
        ```

    Instantiate:
        ```python
        from langchain_easytranscriber import EasyTranscriberLoader

        loader = EasyTranscriberLoader(
            ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
            include_summary=True,
            summary_language="English",
        )
        ```

    Lazy load:
        ```python
        for doc in loader.lazy_load():
            print(doc.metadata["title"], len(doc.page_content))
        ```

    Async load:
        ```python
        docs = [doc async for doc in loader.alazy_load()]
        ```
    """

    def __init__(
        self,
        urls: str | Sequence[str],
        *,
        api_key: str | None = None,
        include_summary: bool = False,
        summary_language: str | None = None,
        continue_on_failure: bool = False,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = 2,
    ) -> None:
        """Create the loader.

        Args:
            urls: One YouTube URL or a list of them (watch, youtu.be, shorts or
                embed links; bare video IDs are not accepted by the API).
            api_key: EasyTranscriber API key. Defaults to ``EASYTRANSCRIBER_API_KEY``.
            include_summary: Also generate an AI summary per video (1 credit each).
            summary_language: Summary language display name, e.g. ``"English"``
                or ``"Spanish"``. ``None`` matches the video's language.
            continue_on_failure: Log and skip videos that fail instead of raising.
            base_url: API base URL.
            timeout: Per-request timeout in seconds.
            max_retries: Retries after HTTP 429 responses.
        """
        self.urls = [urls] if isinstance(urls, str) else list(urls)
        if not self.urls:
            msg = "urls must contain at least one YouTube URL"
            raise ValueError(msg)
        self._api_key = resolve_api_key(api_key)
        self.include_summary = include_summary
        self.summary_language = summary_language
        self.continue_on_failure = continue_on_failure
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

    def _client_kwargs(self) -> dict[str, Any]:
        return {"base_url": self.base_url, "timeout": self.timeout, "max_retries": self.max_retries}

    def _document(
        self, url: str, transcript: dict[str, Any], summary: dict[str, Any] | None
    ) -> Document:
        metadata: dict[str, Any] = {
            "source": url,
            "video_id": transcript["video_id"],
            "title": transcript.get("video_title"),
            "method": transcript.get("method"),
            "cached": transcript.get("cached"),
            "credits_used": transcript.get("credits_used", 0),
        }
        if summary is not None:
            metadata["summary"] = summary["summary"]
            metadata["credits_used"] += summary.get("credits_used", 0)
        return Document(page_content=transcript["transcript"], metadata=metadata)

    def _load_one(self, client: EasyTranscriber, url: str) -> Document:
        transcript: dict[str, Any] = dict(client.transcribe(url))
        summary: dict[str, Any] | None = None
        if self.include_summary:
            summary = dict(
                client.summarize(
                    transcript["transcript"],
                    video_id=transcript["video_id"],
                    language=self.summary_language,
                )
            )
        return self._document(url, transcript, summary)

    async def _aload_one(self, client: AsyncEasyTranscriber, url: str) -> Document:
        transcript: dict[str, Any] = dict(await client.transcribe(url))
        summary: dict[str, Any] | None = None
        if self.include_summary:
            summary = dict(
                await client.summarize(
                    transcript["transcript"],
                    video_id=transcript["video_id"],
                    language=self.summary_language,
                )
            )
        return self._document(url, transcript, summary)

    def lazy_load(self) -> Iterator[Document]:
        """Yield one Document per URL."""
        with sync_client(self._api_key, **self._client_kwargs()) as client:
            for url in self.urls:
                try:
                    yield self._load_one(client, url)
                except REQUEST_ERRORS as error:
                    if not self.continue_on_failure:
                        raise
                    logger.warning("Skipping %s: %s", url, describe_error(error))

    async def alazy_load(self) -> AsyncIterator[Document]:
        """Asynchronously yield one Document per URL."""
        async with async_client(self._api_key, **self._client_kwargs()) as client:
            for url in self.urls:
                try:
                    yield await self._aload_one(client, url)
                except REQUEST_ERRORS as error:
                    if not self.continue_on_failure:
                        raise
                    logger.warning("Skipping %s: %s", url, describe_error(error))
