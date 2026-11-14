"""Live tests against the real API. They spend credits, so they only run when
EASYTRANSCRIBER_API_KEY is set and are not collected by the default test run:

    EASYTRANSCRIBER_API_KEY=et_... uv run --group test pytest tests/integration_tests

The transcript tests use a cached video (0 credits). The standard search tests
call the search tool several times, at 1 credit per call.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_easytranscriber import (
    EasyTranscriberLoader,
    EasyTranscriberSearchTool,
    EasyTranscriberTranscriptTool,
)

URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

pytestmark = pytest.mark.skipif(
    not os.environ.get("EASYTRANSCRIBER_API_KEY"), reason="EASYTRANSCRIBER_API_KEY not set"
)


class TestTranscriptToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return EasyTranscriberTranscriptTool

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"url": URL}


class TestSearchToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return EasyTranscriberSearchTool

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"query": "python asyncio tutorial"}


def test_loader_live() -> None:
    docs = EasyTranscriberLoader(URL).load()
    assert docs[0].metadata["video_id"] == "dQw4w9WgXcQ"
    assert docs[0].page_content
