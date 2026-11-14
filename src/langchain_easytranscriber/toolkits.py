"""Toolkit bundling the EasyTranscriber tools."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, BaseToolkit
from langchain_core.utils import secret_from_env
from pydantic import Field, SecretStr

from langchain_easytranscriber._client import API_KEY_ENV, DEFAULT_BASE_URL, DEFAULT_TIMEOUT
from langchain_easytranscriber.tools import (
    EasyTranscriberSearchTool,
    EasyTranscriberSummaryTool,
    EasyTranscriberTranscriptTool,
)


class EasyTranscriberToolkit(BaseToolkit):
    """Transcript, summary and search tools sharing one API key.

    Setup:
        ```bash
        pip install -U langchain-easytranscriber
        export EASYTRANSCRIBER_API_KEY="et_..."
        ```

    Instantiation:
        ```python
        from langchain_easytranscriber import EasyTranscriberToolkit

        tools = EasyTranscriberToolkit().get_tools()
        ```
    """

    api_key: SecretStr = Field(default_factory=secret_from_env(API_KEY_ENV, default=""))
    base_url: str = DEFAULT_BASE_URL
    timeout: float = DEFAULT_TIMEOUT
    max_retries: int = 2

    def get_tools(self) -> list[BaseTool]:
        """Return the transcript, summary and search tools."""
        settings: dict[str, Any] = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        return [
            EasyTranscriberTranscriptTool(**settings),
            EasyTranscriberSummaryTool(**settings),
            EasyTranscriberSearchTool(**settings),
        ]
