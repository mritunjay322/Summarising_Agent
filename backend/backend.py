"""
backend.py - Backend orchestrator
Coordinates input.py and Summarizing.py from the backend package.
"""

from typing import Callable, Union

from backend.input import InputProcessor
from backend.Summarizing import summarize, SummaryResult


class SummarizerAgent:
    """
    Single entry point for the entire backend.
    Initialize once with your llm_call, then call process() for each request.
    """

    def __init__(self, llm_call: Callable[[str], str]):
        self.llm_call = llm_call

    def process_text(self, text: str) -> SummaryResult:
        """Process raw text input."""
        processor = InputProcessor()
        clean_text = processor.from_text(text).text
        
        return summarize(clean_text, self.llm_call)

    def process_file(self, file_bytes: bytes, filename: str) -> SummaryResult:
        """Process uploaded file."""
        processor = InputProcessor()
        clean_text = processor.from_file(file_bytes, filename).text
        
        return summarize(clean_text, self.llm_call)

    def process(self, source: Union[str, bytes], filename: str = None) -> SummaryResult:
        """
        Auto-detect and process:
        - bytes → file (requires filename)
        - str starting with http → URL
        - str → text
        """
        if isinstance(source, bytes):
            if not filename:
                raise ValueError("filename required for file upload")
            return self.process_file(source, filename)

        if isinstance(source, str) and source.startswith(("http://", "https://")):
            processor = InputProcessor()
            clean_text = processor.from_url(source).text
            return summarize(clean_text, self.llm_call)

        return self.process_text(source)