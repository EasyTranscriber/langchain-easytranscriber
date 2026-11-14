# langchain-easytranscriber

LangChain integration for [EasyTranscriber](https://www.easytranscriber.com): load YouTube
transcripts as Documents, and give agents tools to get a transcript, summarize a video, or
search YouTube.

- `EasyTranscriberLoader`: YouTube URLs to `Document`s (transcript text plus metadata, optional AI summary)
- `EasyTranscriberTranscriptTool`: transcript of a YouTube video
- `EasyTranscriberSummaryTool`: Markdown AI summary of a YouTube video, in 65 languages
- `EasyTranscriberSearchTool`: YouTube video search, up to 10 results
- `EasyTranscriberToolkit`: the three tools with one API key

Transcripts come from the video's captions, or from speech-to-text when a video has no
captions. Videos up to 4 hours are supported.

## Install

```bash
pip install -U langchain-easytranscriber
```

Python 3.10+, `langchain-core` 1.x. Built on the
[`easytranscriber-sdk`](https://pypi.org/project/easytranscriber-sdk/) client.

## API key

Create a free account at [easytranscriber.com](https://www.easytranscriber.com) and a key in
[Dashboard > API Keys](https://www.easytranscriber.com/dashboard/api-keys). Signup includes
5 credits, and 1 credit is added each day while the balance is below 5. API access is
included on every plan.

```bash
export EASYTRANSCRIBER_API_KEY="et_..."
```

Or pass `api_key="et_..."` to any class.

## Credits

| Call | Credits |
|---|---|
| Transcript of a cached video | 0 |
| Transcript of a new video, up to 2 hours | 1 |
| Transcript of a new video, over 2 hours (4-hour maximum) | 2 |
| Summary | 1, on top of the transcript |
| YouTube search | 1 |

Plans and prices: [pricing](https://www.easytranscriber.com/pricing).

## Document loader

```python
from langchain_easytranscriber import EasyTranscriberLoader

loader = EasyTranscriberLoader(
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/9bZkp7q19f0",
    ],
    include_summary=True,  # adds metadata["summary"], 1 credit per video
    summary_language="English",  # omit to match each video's language
)

for doc in loader.lazy_load():
    print(doc.metadata["title"], doc.metadata["credits_used"])
    print(doc.page_content[:200])
```

Each `Document` has the transcript as `page_content` and this metadata:

| Key | Value |
|---|---|
| `source` | the URL you passed |
| `video_id` | YouTube video ID |
| `title` | video title, or `None` |
| `method` | how the transcript was made: `yt-dlp` or `innertube` (captions), `deepgram` (speech-to-text) |
| `cached` | `True` when served from cache at 0 credits |
| `credits_used` | credits spent on this video, including the summary when requested |
| `summary` | Markdown summary, only with `include_summary=True` |

Use `continue_on_failure=True` to skip videos that fail instead of raising.
`await loader.aload()` and `loader.alazy_load()` use the async client.

Accepted URLs: `youtube.com/watch?v=`, `youtu.be/`, `/shorts/` and `/embed/` links. Bare video
IDs are not accepted.

## Tools

```python
from langchain_easytranscriber import EasyTranscriberTranscriptTool

tool = EasyTranscriberTranscriptTool()
print(tool.invoke({"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}))
```

Invoked with a `ToolCall`, each tool returns a `ToolMessage` whose `artifact` holds the API
response, including `credits_used`:

```python
message = tool.invoke(
    {
        "args": {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
        "id": "1",
        "name": tool.name,
        "type": "tool_call",
    }
)
message.artifact["cached"], message.artifact["credits_used"]
```

### In an agent

```python
# pip install -qU "langchain[anthropic]"
from langchain.agents import create_agent
from langchain_easytranscriber import EasyTranscriberToolkit

agent = create_agent(
    model="anthropic:claude-sonnet-5-5",
    tools=EasyTranscriberToolkit().get_tools(),
)
result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Find a recent video on Python asyncio and summarize it."}
        ]
    }
)
print(result["messages"][-1].content)
```

API errors (for example `insufficient_credits` or `unauthorized`), timeouts and connection
errors are raised as `ToolException`. Pass `handle_tool_error=True` to a tool to return the
message to the agent instead.

## Development

```bash
uv run --group test pytest                      # unit tests, mocked HTTP, no credits
uv run --group lint ruff check . && uv run --group lint ruff format --check .
uv run --group typing mypy
EASYTRANSCRIBER_API_KEY=et_... uv run --group test pytest tests/integration_tests  # live, spends credits
```

Unit tests include LangChain's standard `ToolsUnitTests` from `langchain-tests`.

## Links

- REST API docs: https://www.easytranscriber.com/docs/api
- Python and JavaScript SDKs: https://www.easytranscriber.com/sdk
- MCP server: https://www.easytranscriber.com/docs/mcp

EasyTranscriber is not affiliated with YouTube or Google. MIT licensed.
