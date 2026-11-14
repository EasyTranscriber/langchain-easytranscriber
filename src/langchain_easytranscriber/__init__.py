"""LangChain integration for EasyTranscriber: YouTube transcripts, AI summaries and search."""

from importlib import metadata

from langchain_easytranscriber.document_loaders import EasyTranscriberLoader
from langchain_easytranscriber.toolkits import EasyTranscriberToolkit
from langchain_easytranscriber.tools import (
    EasyTranscriberSearchTool,
    EasyTranscriberSummaryTool,
    EasyTranscriberTranscriptTool,
)

try:
    __version__ = metadata.version(__package__ or __name__)
except metadata.PackageNotFoundError:
    __version__ = ""

__all__ = [
    "EasyTranscriberLoader",
    "EasyTranscriberSearchTool",
    "EasyTranscriberSummaryTool",
    "EasyTranscriberToolkit",
    "EasyTranscriberTranscriptTool",
    "__version__",
]
