"""Turn any supported input into page images (in memory, nothing written to disk).

Accepted inputs:
- Member 1's hand-off dict: ``preprocess.process_upload(...).ocr_payload()``
  (``content`` = prepared PNG bytes or the original PDF bytes; ``warnings`` are kept)
- a file path (image or PDF), raw file bytes, a PIL image, or an RGB numpy array

PDF pages are rendered with PyMuPDF at ``PDF_DPI`` (default 200), up to
``EXTRACTION_MAX_PAGES`` pages (default 3).
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass
class LoadedDocument:
    pages: list[Image.Image]
    warnings: list[str] = field(default_factory=list)


def _render_pdf(data: bytes, warnings: list[str]) -> list[Image.Image]:
    import fitz  # PyMuPDF

    max_pages = int(os.getenv("EXTRACTION_MAX_PAGES", "3"))
    zoom = int(os.getenv("PDF_DPI", "200")) / 72
    pages = []
    with fitz.open(stream=data, filetype="pdf") as pdf:
        if pdf.page_count == 0:
            raise ValueError("PDF has no pages")
        if pdf.page_count > max_pages:
            warnings.append(f"pdf_truncated: only first {max_pages} of {pdf.page_count} pages processed")
        for page in list(pdf)[:max_pages]:
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            pages.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    return pages


def _from_bytes(data: bytes, warnings: list[str]) -> list[Image.Image]:
    if data.startswith(b"%PDF"):
        try:
            return _render_pdf(data, warnings)
        except ValueError:
            raise
        except Exception as exc:  # PyMuPDF raises its own error types for corrupt files
            raise ValueError(f"unreadable PDF ({type(exc).__name__})") from exc
    with Image.open(io.BytesIO(data)) as img:
        return [img.convert("RGB")]


def load_document(source: Any) -> LoadedDocument:
    """Raises FileNotFoundError / ValueError / OSError for unusable input."""
    warnings: list[str] = []
    if isinstance(source, dict):  # Member 1 hand-off payload
        content = source.get("content")
        if not isinstance(content, bytes | bytearray) or not content:
            raise ValueError("payload has no content bytes")
        warnings += [f"input: {w}" for w in source.get("warnings") or [] if isinstance(w, str)]
        return LoadedDocument(_from_bytes(bytes(content), warnings), warnings)
    if isinstance(source, str | Path):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"file not found: {path.name}")
        return LoadedDocument(_from_bytes(path.read_bytes(), warnings), warnings)
    if isinstance(source, bytes | bytearray):
        return LoadedDocument(_from_bytes(bytes(source), warnings), warnings)
    if isinstance(source, Image.Image):
        return LoadedDocument([source.convert("RGB")], warnings)
    if hasattr(source, "shape") and hasattr(source, "dtype"):  # numpy array, RGB
        return LoadedDocument([Image.fromarray(source).convert("RGB")], warnings)
    raise ValueError("unsupported input type")
