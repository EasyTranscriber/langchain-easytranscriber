from __future__ import annotations

import json

import httpx
import pytest
import respx
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException

from langchain_easytranscriber import (
    EasyTranscriberSearchTool,
    EasyTranscriberSummaryTool,
    EasyTranscriberToolkit,
    EasyTranscriberTranscriptTool,
)

from .conftest import SEARCH, SUMMARY, TEST_KEY, TRANSCRIPT, URL


def tool_call(name: str, args: dict[str, object]) -> dict[str, object]:
    return {"args": args, "id": "call_1", "name": name, "type": "tool_call"}


def test_tools_require_api_key() -> None:
    tools = (EasyTranscriberTranscriptTool, EasyTranscriberSummaryTool, EasyTranscriberSearchTool)
    for cls in tools:
        with pytest.raises(ValueError, match="EASYTRANSCRIBER_API_KEY"):
            cls()


def test_api_key_is_secret() -> None:
    tool = EasyTranscriberTranscriptTool(api_key=TEST_KEY)
    assert TEST_KEY not in repr(tool)
    assert TEST_KEY not in str(tool.model_dump())


def test_transcript_tool_content(api: respx.MockRouter) -> None:
    route = api.post("/transcribe").mock(return_value=httpx.Response(200, json=TRANSCRIPT))
    tool = EasyTranscriberTranscriptTool(api_key=TEST_KEY)

    content = tool.invoke({"url": URL})

    assert content.startswith(
        "Title: Example video\nURL: https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    )
    assert content.endswith(TRANSCRIPT["transcript"])
    assert json.loads(route.calls.last.request.content) == {"url": URL}
    assert route.calls.last.request.headers["Authorization"] == f"Bearer {TEST_KEY}"


def test_transcript_tool_call_returns_artifact(api: respx.MockRouter) -> None:
    api.post("/transcribe").mock(return_value=httpx.Response(200, json=TRANSCRIPT))
    tool = EasyTranscriberTranscriptTool(api_key=TEST_KEY)

    message = tool.invoke(tool_call(tool.name, {"url": URL}))

    assert isinstance(message, ToolMessage)
    assert message.tool_call_id == "call_1"
    assert message.artifact == TRANSCRIPT


def test_summary_tool_chains_transcribe_and_summarize(api: respx.MockRouter) -> None:
    api.post("/transcribe").mock(return_value=httpx.Response(200, json=TRANSCRIPT))
    summarize = api.post("/summarize").mock(return_value=httpx.Response(200, json=SUMMARY))
    tool = EasyTranscriberSummaryTool(api_key=TEST_KEY)

    message = tool.invoke(tool_call(tool.name, {"url": URL, "language": "Spanish"}))

    assert "Summary:\n## Summary" in message.content
    assert message.artifact["credits_used"] == 2
    assert message.artifact["cached_transcript"] is False
    assert json.loads(summarize.calls.last.request.content) == {
        "transcript": TRANSCRIPT["transcript"],
        "video_id": "dQw4w9WgXcQ",
        "language": "Spanish",
    }


def test_summary_tool_omits_language_when_not_given(api: respx.MockRouter) -> None:
    api.post("/transcribe").mock(return_value=httpx.Response(200, json=TRANSCRIPT))
    summarize = api.post("/summarize").mock(return_value=httpx.Response(200, json=SUMMARY))
    EasyTranscriberSummaryTool(api_key=TEST_KEY).invoke({"url": URL})
    assert "language" not in json.loads(summarize.calls.last.request.content)


def test_search_tool(api: respx.MockRouter) -> None:
    route = api.get("/youtube/search").mock(return_value=httpx.Response(200, json=SEARCH))
    tool = EasyTranscriberSearchTool(api_key=TEST_KEY)

    content = tool.invoke({"query": "python asyncio"})

    assert content == (
        "1. Asyncio in 20 minutes | Example channel | 2026-01-15T12:00:00Z | "
        "https://www.youtube.com/watch?v=abcdefghijk"
    )
    params = route.calls.last.request.url.params
    assert params["q"] == "python asyncio"
    assert params["type"] == "video"


def test_search_tool_no_results(api: respx.MockRouter) -> None:
    api.get("/youtube/search").mock(return_value=httpx.Response(200, json=dict(SEARCH, results=[])))
    content = EasyTranscriberSearchTool(api_key=TEST_KEY).invoke({"query": "zzz"})
    assert content == "No videos found."


def test_api_errors_become_tool_exceptions(api: respx.MockRouter) -> None:
    api.post("/transcribe").mock(
        return_value=httpx.Response(
            402, json={"error": {"code": "insufficient_credits", "message": "Not enough credits"}}
        )
    )
    tool = EasyTranscriberTranscriptTool(api_key=TEST_KEY)
    with pytest.raises(ToolException, match="insufficient_credits"):
        tool.invoke({"url": URL})


def test_handle_tool_error_returns_message_to_agent(api: respx.MockRouter) -> None:
    api.post("/transcribe").mock(
        return_value=httpx.Response(
            401, json={"error": {"code": "unauthorized", "message": "Invalid key"}}
        )
    )
    tool = EasyTranscriberTranscriptTool(api_key=TEST_KEY, handle_tool_error=True)
    message = tool.invoke(tool_call(tool.name, {"url": URL}))
    assert "401 (unauthorized)" in message.content


def test_error_message_has_one_full_stop(api: respx.MockRouter) -> None:
    api.post("/transcribe").mock(
        return_value=httpx.Response(
            402,
            json={
                "error": {
                    "code": "insufficient_credits",
                    "message": "Not enough credits. Upgrade your plan at /pricing.",
                }
            },
        )
    )
    tool = EasyTranscriberTranscriptTool(api_key=TEST_KEY, handle_tool_error=True)
    content = tool.invoke({"url": URL})
    assert "/pricing. Check the balance" in content
    assert ".." not in content


def test_transport_errors_become_tool_exceptions(api: respx.MockRouter) -> None:
    api.post("/transcribe").mock(side_effect=httpx.ConnectError("connection refused"))
    tool = EasyTranscriberTranscriptTool(api_key=TEST_KEY)
    with pytest.raises(ToolException, match="request failed"):
        tool.invoke({"url": URL})


def test_timeout_is_returned_to_agent(api: respx.MockRouter) -> None:
    api.get("/youtube/search").mock(side_effect=httpx.ReadTimeout("timed out"))
    tool = EasyTranscriberSearchTool(api_key=TEST_KEY, handle_tool_error=True)
    message = tool.invoke(tool_call(tool.name, {"query": "x"}))
    assert "timed out" in message.content
    assert TEST_KEY not in message.content


async def test_async_transport_errors_become_tool_exceptions(api: respx.MockRouter) -> None:
    api.post("/transcribe").mock(side_effect=httpx.ConnectError("connection refused"))
    tool = EasyTranscriberSummaryTool(api_key=TEST_KEY)
    with pytest.raises(ToolException, match="request failed"):
        await tool.ainvoke({"url": URL})


async def test_async_tools(api: respx.MockRouter) -> None:
    api.post("/transcribe").mock(return_value=httpx.Response(200, json=TRANSCRIPT))
    api.post("/summarize").mock(return_value=httpx.Response(200, json=SUMMARY))
    api.get("/youtube/search").mock(return_value=httpx.Response(200, json=SEARCH))

    transcript = await EasyTranscriberTranscriptTool(api_key=TEST_KEY).ainvoke({"url": URL})
    summary = await EasyTranscriberSummaryTool(api_key=TEST_KEY).ainvoke({"url": URL})
    search = await EasyTranscriberSearchTool(api_key=TEST_KEY).ainvoke({"query": "x"})

    assert transcript.endswith(TRANSCRIPT["transcript"])
    assert summary.endswith(SUMMARY["summary"])
    assert search.startswith("1. Asyncio")


def test_toolkit_shares_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EASYTRANSCRIBER_API_KEY", TEST_KEY)
    tools = EasyTranscriberToolkit(base_url="http://127.0.0.1:9/api/v1").get_tools()
    assert [t.name for t in tools] == [
        "easytranscriber_youtube_transcript",
        "easytranscriber_youtube_summary",
        "easytranscriber_youtube_search",
    ]
    assert all(t.base_url == "http://127.0.0.1:9/api/v1" for t in tools)
    assert all(t.api_key.get_secret_value() == TEST_KEY for t in tools)
