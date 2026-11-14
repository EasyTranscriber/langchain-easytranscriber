from __future__ import annotations

import json

import httpx
import pytest
import respx
from easytranscriber_sdk import APIError

from langchain_easytranscriber import EasyTranscriberLoader

from .conftest import SUMMARY, TEST_KEY, TRANSCRIPT, URL


def test_requires_api_key() -> None:
    with pytest.raises(ValueError, match="EASYTRANSCRIBER_API_KEY"):
        EasyTranscriberLoader(URL)


def test_reads_key_from_env(monkeypatch: pytest.MonkeyPatch, api: respx.MockRouter) -> None:
    monkeypatch.setenv("EASYTRANSCRIBER_API_KEY", TEST_KEY)
    route = api.post("/transcribe").mock(return_value=httpx.Response(200, json=TRANSCRIPT))
    EasyTranscriberLoader(URL).load()
    assert route.calls.last.request.headers["Authorization"] == f"Bearer {TEST_KEY}"


def test_rejects_empty_url_list() -> None:
    with pytest.raises(ValueError, match="at least one"):
        EasyTranscriberLoader([], api_key=TEST_KEY)


def test_load_transcript_document(api: respx.MockRouter) -> None:
    route = api.post("/transcribe").mock(return_value=httpx.Response(200, json=TRANSCRIPT))
    summarize = api.post("/summarize")

    docs = EasyTranscriberLoader(URL, api_key=TEST_KEY).load()

    assert len(docs) == 1
    assert docs[0].page_content == TRANSCRIPT["transcript"]
    assert docs[0].metadata == {
        "source": URL,
        "video_id": "dQw4w9WgXcQ",
        "title": "Example video",
        "method": "yt-dlp",
        "cached": False,
        "credits_used": 1,
    }
    assert json.loads(route.calls.last.request.content) == {"url": URL}
    assert route.calls.last.request.headers["Authorization"] == f"Bearer {TEST_KEY}"
    assert not summarize.called


def test_include_summary_adds_metadata_and_credits(api: respx.MockRouter) -> None:
    api.post("/transcribe").mock(return_value=httpx.Response(200, json=TRANSCRIPT))
    summarize = api.post("/summarize").mock(return_value=httpx.Response(200, json=SUMMARY))

    loader = EasyTranscriberLoader(
        URL, api_key=TEST_KEY, include_summary=True, summary_language="French"
    )
    doc = next(loader.lazy_load())

    assert doc.metadata["summary"] == SUMMARY["summary"]
    assert doc.metadata["credits_used"] == 2
    body = json.loads(summarize.calls.last.request.content)
    assert body == {
        "transcript": TRANSCRIPT["transcript"],
        "video_id": "dQw4w9WgXcQ",
        "language": "French",
    }


def test_multiple_urls_in_order(api: respx.MockRouter) -> None:
    second = dict(TRANSCRIPT, video_id="abcdefghijk", cached=True, credits_used=0)
    api.post("/transcribe").mock(
        side_effect=[httpx.Response(200, json=TRANSCRIPT), httpx.Response(200, json=second)]
    )
    urls = [URL, "https://youtu.be/abcdefghijk"]

    docs = EasyTranscriberLoader(urls, api_key=TEST_KEY).load()

    assert [d.metadata["source"] for d in docs] == urls
    assert [d.metadata["credits_used"] for d in docs] == [1, 0]


def test_api_error_raises_by_default(api: respx.MockRouter) -> None:
    api.post("/transcribe").mock(
        return_value=httpx.Response(
            402, json={"error": {"code": "insufficient_credits", "message": "Not enough credits"}}
        )
    )
    with pytest.raises(APIError) as excinfo:
        EasyTranscriberLoader(URL, api_key=TEST_KEY).load()
    assert excinfo.value.code == "insufficient_credits"


def test_continue_on_failure_skips_bad_video(
    api: respx.MockRouter, caplog: pytest.LogCaptureFixture
) -> None:
    api.post("/transcribe").mock(
        side_effect=[
            httpx.Response(
                422, json={"error": {"code": "transcription_unavailable", "message": "No audio"}}
            ),
            httpx.Response(200, json=TRANSCRIPT),
        ]
    )
    loader = EasyTranscriberLoader(
        ["https://youtu.be/broken00000", URL], api_key=TEST_KEY, continue_on_failure=True
    )

    docs = loader.load()

    assert [d.metadata["source"] for d in docs] == [URL]
    assert "transcription_unavailable" in caplog.text


def test_continue_on_failure_skips_transport_error(
    api: respx.MockRouter, caplog: pytest.LogCaptureFixture
) -> None:
    api.post("/transcribe").mock(
        side_effect=[httpx.ReadTimeout("timed out"), httpx.Response(200, json=TRANSCRIPT)]
    )
    loader = EasyTranscriberLoader(
        ["https://youtu.be/slow0000000", URL], api_key=TEST_KEY, continue_on_failure=True
    )

    docs = loader.load()

    assert [d.metadata["source"] for d in docs] == [URL]
    assert "timed out" in caplog.text


def test_transport_error_raises_by_default(api: respx.MockRouter) -> None:
    api.post("/transcribe").mock(side_effect=httpx.ConnectError("connection refused"))
    with pytest.raises(httpx.ConnectError):
        EasyTranscriberLoader(URL, api_key=TEST_KEY).load()


async def test_async_continue_on_failure_skips_transport_error(api: respx.MockRouter) -> None:
    api.post("/transcribe").mock(
        side_effect=[httpx.ConnectError("refused"), httpx.Response(200, json=TRANSCRIPT)]
    )
    loader = EasyTranscriberLoader(
        ["https://youtu.be/down0000000", URL], api_key=TEST_KEY, continue_on_failure=True
    )
    docs = [doc async for doc in loader.alazy_load()]
    assert [d.metadata["source"] for d in docs] == [URL]


async def test_alazy_load(api: respx.MockRouter) -> None:
    api.post("/transcribe").mock(return_value=httpx.Response(200, json=TRANSCRIPT))
    api.post("/summarize").mock(return_value=httpx.Response(200, json=SUMMARY))

    loader = EasyTranscriberLoader(URL, api_key=TEST_KEY, include_summary=True)
    docs = [doc async for doc in loader.alazy_load()]

    assert len(docs) == 1
    assert docs[0].metadata["summary"] == SUMMARY["summary"]
    assert docs[0].metadata["credits_used"] == 2


async def test_aload_uses_native_async(api: respx.MockRouter) -> None:
    api.post("/transcribe").mock(return_value=httpx.Response(200, json=TRANSCRIPT))
    docs = await EasyTranscriberLoader(URL, api_key=TEST_KEY).aload()
    assert docs[0].metadata["video_id"] == "dQw4w9WgXcQ"


def test_custom_base_url() -> None:
    with respx.mock(base_url="http://127.0.0.1:9/api/v1") as router:
        router.post("/transcribe").mock(return_value=httpx.Response(200, json=TRANSCRIPT))
        docs = EasyTranscriberLoader(
            URL, api_key=TEST_KEY, base_url="http://127.0.0.1:9/api/v1"
        ).load()
    assert docs[0].page_content == TRANSCRIPT["transcript"]
