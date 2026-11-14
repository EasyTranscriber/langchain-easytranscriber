"""LangChain standard unit tests (langchain-tests) for each tool."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from langchain_tests.unit_tests import ToolsUnitTests

from langchain_easytranscriber import (
    EasyTranscriberSearchTool,
    EasyTranscriberSummaryTool,
    EasyTranscriberTranscriptTool,
)

from .conftest import TEST_KEY, URL


class _Base(ToolsUnitTests):
    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"api_key": TEST_KEY}

    @property
    def init_from_env_params(self) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
        return (
            {"EASYTRANSCRIBER_API_KEY": TEST_KEY},
            {},
            {"base_url": "https://www.easytranscriber.com/api/v1"},
        )


class TestTranscriptToolStandard(_Base):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return EasyTranscriberTranscriptTool

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"url": URL}


class TestSummaryToolStandard(_Base):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return EasyTranscriberSummaryTool

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"url": URL, "language": "English"}


class TestSearchToolStandard(_Base):
    @property
    def tool_constructor(self) -> type[BaseTool]:
        return EasyTranscriberSearchTool

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"query": "python asyncio tutorial"}
