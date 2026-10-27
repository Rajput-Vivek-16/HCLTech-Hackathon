"""Member 2: OCR + VLM extraction subsystem. See docs/MEMBER2_EXTRACTION.md."""

from __future__ import annotations

from pathlib import Path

from .ai_extractor import extract_with_vlm
from .ocr_engine import extract_with_ocr
from .schemas import CANONICAL_FIELDS, ExtractionResult, FieldResult

__all__ = [
    "CANONICAL_FIELDS",
    "ExtractionResult",
    "FieldResult",
    "extract_both",
    "extract_with_ocr",
    "extract_with_vlm",
]


def extract_both(image: str | Path | bytes, *, document_type: str | None = None) -> dict[str, ExtractionResult]:
    """Run both branches independently on the same image. No comparison,
    validation or routing is done here (that is Member 3's job)."""
    return {
        "ocr": extract_with_ocr(image, document_type=document_type),
        "vlm": extract_with_vlm(image, document_type=document_type),
    }
