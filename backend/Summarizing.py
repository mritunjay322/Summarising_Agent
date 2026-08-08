"""
summarising.py - Summarization module
Takes clean text and returns a summary via llm_call.
"""

import re
import time
from typing import Callable, List
from dataclasses import dataclass


@dataclass
class SummaryResult:
    summary: str
    input_word_count: int
    output_word_count: int
    compression_ratio: float
    processing_time: float


def summarize(text: str, llm_call: Callable[[str], str], max_words: int = 1500) -> SummaryResult:
    """
    Summarize text using the provided llm_call function.
    Handles chunking for long texts automatically.
    """
    start = time.time()
    words = text.split()
    
    if len(words) <= max_words:
        summary = _single_pass(text, llm_call)
    else:
        summary = _map_reduce(text, llm_call, max_words)
    
    elapsed = time.time() - start
    in_w = len(words)
    out_w = len(summary.split())
    
    return SummaryResult(
        summary=summary,
        input_word_count=in_w,
        output_word_count=out_w,
        compression_ratio=round(out_w / in_w, 3) if in_w else 0,
        processing_time=round(elapsed, 2)
    )


def _single_pass(text: str, llm_call: Callable[[str], str]) -> str:
    """Summarize in one LLM call."""
    prompt = f"Provide a comprehensive summary of the following text:\n\n{text}"
    return llm_call(prompt).strip()


def _map_reduce(text: str, llm_call: Callable[[str], str], max_words: int) -> str:
    """Chunk long text, summarize each chunk, then synthesize."""
    chunks = _chunk_text(text, max_words)
    
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        prompt = f"Summarize this section ({i+1}/{len(chunks)}):\n\n{chunk}"
        chunk_summaries.append(llm_call(prompt).strip())
    
    combined = "\n\n".join(chunk_summaries)
    final_prompt = f"Synthesize the following summaries into one coherent summary:\n\n{combined}"
    return llm_call(final_prompt).strip()


def _chunk_text(text: str, max_words: int, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        start = end - overlap if end < len(words) else end
    
    return chunks