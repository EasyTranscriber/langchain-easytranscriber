from __future__ import annotations

from collections.abc import Iterator

import pytest
import respx

BASE = "https://www.easytranscriber.com/api/v1"
TEST_KEY = "et_" + "x" * 40
URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

TRANSCRIPT = {
    "video_id": "dQw4w9WgXcQ",
    "video_title": "Example video",
    "transcript": "Hello and welcome. Today we cover three steps.",
    "method": "yt-dlp",
    "cached": False,
    "credits_used": 1,
}
SUMMARY = {"summary": "## Summary\n\n- Three steps.", "video_id": "dQw4w9WgXcQ", "credits_used": 1}
SEARCH = {
    "results": [
        {
            "video_id": "abcdefghijk",
            "channel_id": "UC" + "a" * 22,
            "title": "Asyncio in 20 minutes",
            "channel": "Example channel",
            "published_at": "2026-01-15T12:00:00Z",
            "thumbnail": None,
        }
    ],
    "next_page_token": None,
    "credits_used": 1,
}


@pytest.fixture(autouse=True)
def _no_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EASYTRANSCRIBER_API_KEY", raising=False)


@pytest.fixture
def api() -> Iterator[respx.MockRouter]:
    with respx.mock(base_url=BASE, assert_all_called=False, assert_all_mocked=True) as router:
        yield router
