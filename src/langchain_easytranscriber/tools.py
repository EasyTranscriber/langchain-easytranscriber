"""LangChain tools for YouTube transcripts, summaries and search via EasyTranscriber."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from langchain_core.utils import secret_from_env
from pydantic import BaseModel, Field, SecretStr, model_validator

from langchain_easytranscriber._client import (
    API_KEY_ENV,
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT,
    KEYS_URL,
    REQUEST_ERRORS,
    async_client,
    describe_error,
    sync_client,
    watch_url,
)


class _EasyTranscriberTool(BaseTool):
    """Shared configuration: API key, base URL, timeout and retries."""

    api_key: SecretStr = Field(default_factory=secret_from_env(API_KEY_ENV, default=""))
    """EasyTranscriber API key (starts with ``et_``). Defaults to ``EASYTRANSCRIBER_API_KEY``."""
    base_url: str = DEFAULT_BASE_URL
    timeout: float = DEFAULT_TIMEOUT
    max_retries: int = 2
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    @model_validator(mode="after")
    def _require_api_key(self) -> _EasyTranscriberTool:
        if not self.api_key.get_secret_value().strip():
            msg = (
                f"No EasyTranscriber API key. Pass api_key=... or set {API_KEY_ENV}. "
                f"Create a key at {KEYS_URL}"
            )
            raise ValueError(msg)
        return self

    def _settings(self) -> dict[str, Any]:
        return {"base_url": self.base_url, "timeout": self.timeout, "max_retries": self.max_retries}

    def _key(self) -> str:
        return self.api_key.get_secret_value()


# --------------------------------------------------------------------------- transcript


class TranscriptInput(BaseModel):
    """Input for EasyTranscriberTranscriptTool."""

    url: str = Field(description="YouTube video URL: a watch, youtu.be, shorts or embed link.")


def _transcript_content(result: dict[str, Any]) -> str:
    title = result.get("video_title") or "(untitled)"
    return (
        f"Title: {title}\n"
        f"URL: {watch_url(result['video_id'])}\n\n"
        f"Transcript:\n{result['transcript']}"
    )


class EasyTranscriberTranscriptTool(_EasyTranscriberTool):
    """Get the transcript of a YouTube video.

    Uses the video's captions, or speech-to-text when a video has none.
    Credits: 0 if cached, otherwise 1 (2 for videos over 2 hours; 4-hour maximum).

    Setup:
        ```bash
        pip install -U langchain-easytranscriber
        export EASYTRANSCRIBER_API_KEY="et_..."
        ```

    Instantiation:
        ```python
        from langchain_easytranscriber import EasyTranscriberTranscriptTool

        tool = EasyTranscriberTranscriptTool()
        ```

    Invocation with args:
        ```python
        tool.invoke({"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"})
        ```

    Invocation with ToolCall:
        ```python
        msg = tool.invoke(
            {
                "args": {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
                "id": "1",
                "name": tool.name,
                "type": "tool_call",
            }
        )
        msg.artifact["credits_used"]
        ```
    """

    name: str = "easytranscriber_youtube_transcript"
    description: str = (
        "Get the full transcript of a YouTube video. Input: a YouTube video URL. "
        "Returns the video title, URL and transcript text."
    )
    args_schema: type[BaseModel] = TranscriptInput

    def _run(
        self, url: str, run_manager: CallbackManagerForToolRun | None = None
    ) -> tuple[str, dict[str, Any]]:
        try:
            with sync_client(self._key(), **self._settings()) as client:
                result: dict[str, Any] = dict(client.transcribe(url))
        except REQUEST_ERRORS as error:
            raise ToolException(describe_error(error)) from error
        return _transcript_content(result), result

    async def _arun(
        self, url: str, run_manager: AsyncCallbackManagerForToolRun | None = None
    ) -> tuple[str, dict[str, Any]]:
        try:
            async with async_client(self._key(), **self._settings()) as client:
                result: dict[str, Any] = dict(await client.transcribe(url))
        except REQUEST_ERRORS as error:
            raise ToolException(describe_error(error)) from error
        return _transcript_content(result), result


# --------------------------------------------------------------------------- summary


class SummaryInput(BaseModel):
    """Input for EasyTranscriberSummaryTool."""

    url: str = Field(description="YouTube video URL: a watch, youtu.be, shorts or embed link.")
    language: str | None = Field(
        default=None,
        description=(
            "Summary language as an English display name, for example 'English', "
            "'Spanish' or 'Japanese'. Omit to use the video's language."
        ),
    )


def _summary_content(transcript: dict[str, Any], summary: dict[str, Any]) -> str:
    title = transcript.get("video_title") or "(untitled)"
    return (
        f"Title: {title}\n"
        f"URL: {watch_url(transcript['video_id'])}\n\n"
        f"Summary:\n{summary['summary']}"
    )


def _summary_artifact(transcript: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "video_id": transcript["video_id"],
        "video_title": transcript.get("video_title"),
        "summary": summary["summary"],
        "cached_transcript": transcript.get("cached"),
        "credits_used": transcript.get("credits_used", 0) + summary.get("credits_used", 0),
    }


class EasyTranscriberSummaryTool(_EasyTranscriberTool):
    """Summarize a YouTube video: fetch its transcript, then generate an AI summary.

    Summaries are Markdown, in 65 languages. Credits: the transcript (0 if cached,
    otherwise 1, or 2 over 2 hours) plus 1 for the summary.

    Setup:
        ```bash
        pip install -U langchain-easytranscriber
        export EASYTRANSCRIBER_API_KEY="et_..."
        ```

    Instantiation:
        ```python
        from langchain_easytranscriber import EasyTranscriberSummaryTool

        tool = EasyTranscriberSummaryTool()
        ```

    Invocation with args:
        ```python
        tool.invoke({"url": "https://youtu.be/dQw4w9WgXcQ", "language": "French"})
        ```

    Invocation with ToolCall:
        ```python
        tool.invoke(
            {
                "args": {"url": "https://youtu.be/dQw4w9WgXcQ"},
                "id": "1",
                "name": tool.name,
                "type": "tool_call",
            }
        )
        ```
    """

    name: str = "easytranscriber_youtube_summary"
    description: str = (
        "Summarize a YouTube video. Input: a YouTube video URL and an optional summary "
        "language (for example 'English'). Returns the title, URL and a Markdown summary."
    )
    args_schema: type[BaseModel] = SummaryInput

    def _run(
        self,
        url: str,
        language: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> tuple[str, dict[str, Any]]:
        try:
            with sync_client(self._key(), **self._settings()) as client:
                transcript: dict[str, Any] = dict(client.transcribe(url))
                summary: dict[str, Any] = dict(
                    client.summarize(
                        transcript["transcript"],
                        video_id=transcript["video_id"],
                        language=language,
                    )
                )
        except REQUEST_ERRORS as error:
            raise ToolException(describe_error(error)) from error
        return _summary_content(transcript, summary), _summary_artifact(transcript, summary)

    async def _arun(
        self,
        url: str,
        language: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> tuple[str, dict[str, Any]]:
        try:
            async with async_client(self._key(), **self._settings()) as client:
                transcript: dict[str, Any] = dict(await client.transcribe(url))
                summary: dict[str, Any] = dict(
                    await client.summarize(
                        transcript["transcript"],
                        video_id=transcript["video_id"],
                        language=language,
                    )
                )
        except REQUEST_ERRORS as error:
            raise ToolException(describe_error(error)) from error
        return _summary_content(transcript, summary), _summary_artifact(transcript, summary)


# --------------------------------------------------------------------------- search


class SearchInput(BaseModel):
    """Input for EasyTranscriberSearchTool."""

    query: str = Field(description="What to search for on YouTube.")


def _search_content(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No videos found."
    lines = []
    for index, item in enumerate(results, start=1):
        video_id = item.get("video_id")
        link = watch_url(video_id) if video_id else "(no video ID)"
        lines.append(
            f"{index}. {item.get('title')} | {item.get('channel')} | "
            f"{item.get('published_at')} | {link}"
        )
    return "\n".join(lines)


class EasyTranscriberSearchTool(_EasyTranscriberTool):
    """Search YouTube videos; returns up to 10 results. Credits: 1 per search.

    Setup:
        ```bash
        pip install -U langchain-easytranscriber
        export EASYTRANSCRIBER_API_KEY="et_..."
        ```

    Instantiation:
        ```python
        from langchain_easytranscriber import EasyTranscriberSearchTool

        tool = EasyTranscriberSearchTool()
        ```

    Invocation with args:
        ```python
        tool.invoke({"query": "python asyncio tutorial"})
        ```

    Invocation with ToolCall:
        ```python
        tool.invoke(
            {
                "args": {"query": "python asyncio tutorial"},
                "id": "1",
                "name": tool.name,
                "type": "tool_call",
            }
        )
        ```
    """

    name: str = "easytranscriber_youtube_search"
    description: str = (
        "Search YouTube for videos. Input: a search query. Returns up to 10 results "
        "with title, channel, publish date and video URL."
    )
    args_schema: type[BaseModel] = SearchInput

    def _run(
        self, query: str, run_manager: CallbackManagerForToolRun | None = None
    ) -> tuple[str, dict[str, Any]]:
        try:
            with sync_client(self._key(), **self._settings()) as client:
                result: dict[str, Any] = dict(client.search(query, type="video"))
        except REQUEST_ERRORS as error:
            raise ToolException(describe_error(error)) from error
        return _search_content(list(result["results"])), result

    async def _arun(
        self, query: str, run_manager: AsyncCallbackManagerForToolRun | None = None
    ) -> tuple[str, dict[str, Any]]:
        try:
            async with async_client(self._key(), **self._settings()) as client:
                result: dict[str, Any] = dict(await client.search(query, type="video"))
        except REQUEST_ERRORS as error:
            raise ToolException(describe_error(error)) from error
        return _search_content(list(result["results"])), result
