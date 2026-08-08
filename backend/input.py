"""
input.py - Backend input module for Summarizing Agent
Handles text, file, and URL input processing. No UI dependencies.
"""

import io
import re
import os
from typing import Optional, Union
from dataclasses import dataclass
from pathlib import Path

import requests
import PyPDF2
import docx


@dataclass
class InputResult:
    """Standardized result from any input source."""
    text: str
    source: str          # 'text', 'file', 'url'
    source_name: str     # filename or 'user_input' or url
    word_count: int
    char_count: int


class InputProcessor:
    """
    Backend processor for all input types.
    Stateless - no Streamlit or UI dependencies.
    """
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS = {'.txt', '.pdf', '.docx'}
    
    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    
    def from_text(self, text: str) -> InputResult:
        """Process raw text input."""
        if not text or not isinstance(text, str):
            raise ValueError("Text input must be a non-empty string")
        
        cleaned = self._clean_text(text)
        return InputResult(
            text=cleaned,
            source='text',
            source_name='user_input',
            word_count=len(cleaned.split()),
            char_count=len(cleaned)
        )
    
    def from_file(self, file_path: Union[str, Path, bytes], 
                  filename: Optional[str] = None) -> InputResult:
        """
        Process file input from path, bytes, or file-like object.
        
        Args:
            file_path: Path string, Path object, or bytes
            filename: Required when passing bytes; used to detect file type
        """
        # Handle different input types
        if isinstance(file_path, (str, Path)):
            path = Path(file_path)
            filename = path.name
            with open(path, 'rb') as f:
                file_bytes = f.read()
        elif isinstance(file_path, bytes):
            if not filename:
                raise ValueError("filename required when passing bytes")
            file_bytes = file_path
        else:
            raise TypeError("file_path must be str, Path, or bytes")
        
        # Validate extension
        ext = Path(filename).suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}. Supported: {self.SUPPORTED_EXTENSIONS}")
        
        # Extract text
        text = self._extract_from_bytes(file_bytes, ext)
        cleaned = self._clean_text(text)
        
        return InputResult(
            text=cleaned,
            source='file',
            source_name=filename,
            word_count=len(cleaned.split()),
            char_count=len(cleaned)
        )
    
    def from_url(self, url: str, extract_article: bool = True) -> InputResult:
        """
        Fetch and process content from a URL.
        
        Args:
            url: The webpage URL
            extract_article: If True, attempts article extraction (removes nav, ads, etc.)
        """
        if not url.startswith(('http://', 'https://')):
            raise ValueError("URL must start with http:// or https://")
        
        try:
            response = self._session.get(url, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to fetch URL: {e}")
        
        raw_html = response.text
        
        if extract_article:
            text = self._extract_article_text(raw_html, url)
        else:
            text = self._strip_html(raw_html)
        
        cleaned = self._clean_text(text)
        
        return InputResult(
            text=cleaned,
            source='url',
            source_name=url,
            word_count=len(cleaned.split()),
            char_count=len(cleaned)
        )
    
    # ------------------------------------------------------------------ #
    # Extractors
    # ------------------------------------------------------------------ #
    
    def _extract_from_bytes(self, data: bytes, ext: str) -> str:
        """Route to correct extractor based on file extension."""
        extractors = {
            '.txt': self._extract_txt,
            '.pdf': self._extract_pdf,
            '.docx': self._extract_docx,
        }
        return extractors[ext](data)
    
    def _extract_txt(self, data: bytes) -> str:
        """Extract text from TXT bytes."""
        # Try UTF-8 first, fallback to latin-1
        for encoding in ('utf-8', 'latin-1', 'cp1252'):
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode text file with any supported encoding")
    
    def _extract_pdf(self, data: bytes) -> str:
        """Extract text from PDF bytes."""
        pdf_file = io.BytesIO(data)
        reader = PyPDF2.PdfReader(pdf_file)
        
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        
        return "\n\n".join(text_parts)
    
    def _extract_docx(self, data: bytes) -> str:
        """Extract text from DOCX bytes."""
        doc_file = io.BytesIO(data)
        doc = docx.Document(doc_file)
        
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
    
    def _extract_article_text(self, html: str, url: str) -> str:
        """
        Attempts to extract main article content from HTML.
        Falls back to basic HTML stripping if extraction fails.
        """
        try:
            # Try using trafilatura if available (best for articles)
            import trafilatura
            extracted = trafilatura.extract(html, url=url, include_comments=False)
            if extracted and len(extracted) > 200:
                return extracted
        except ImportError:
            pass
        
        try:
            # Fallback to newspaper3k
            from newspaper import Article
            article = Article(url)
            article.set_html(html)
            article.parse()
            if article.text and len(article.text) > 200:
                return article.text
        except ImportError:
            pass
        
        # Final fallback: basic HTML stripping
        return self._strip_html(html)
    
    def _strip_html(self, html: str) -> str:
        """Basic HTML tag removal using regex."""
        # Remove script and style tags entirely
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove tags
        text = re.sub(r'<[^>]+>', ' ', html)
        
        # Decode common HTML entities
        import html as html_module
        text = html_module.unescape(text)
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """Normalize whitespace and clean up text."""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        # Fix common encoding issues
        text = text.replace('\xa0', ' ')
        # Strip leading/trailing whitespace
        return text.strip()


# ---------------------------------------------------------------------- #
# Convenience Functions (module-level API)
# ---------------------------------------------------------------------- #

_processor = InputProcessor()

def process_text(text: str) -> InputResult:
    """Process raw text. Raises ValueError on invalid input."""
    return _processor.from_text(text)

def process_file(file_path: Union[str, Path, bytes], 
                 filename: Optional[str] = None) -> InputResult:
    """Process a file. Raises ValueError/ConnectionError on failure."""
    return _processor.from_file(file_path, filename)

def process_url(url: str, extract_article: bool = True) -> InputResult:
    """Process a URL. Raises ConnectionError/ValueError on failure."""
    return _processor.from_url(url, extract_article)


# ---------------------------------------------------------------------- #
# Example Usage / Testing
# ---------------------------------------------------------------------- #

if __name__ == "__main__":
    # Test with text
    result = process_text("  This is a   test.  It has extra   spaces.  ")
    print(f"Text: {result.word_count} words, {result.char_count} chars")
    print(f"Preview: {result.text[:50]}...")
    
    # Test with URL (requires internet)
    # result = process_url("https://example.com")
    # print(f"URL: {result.word_count} words")