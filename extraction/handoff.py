"""Hand Member 2's results to Member 3's rules engine.

Usage:
    from extraction import extract_both
    from extraction.handoff import to_rules_engine_inputs
    from rules_engine import process_extraction

    results = extract_both(image)
    decision = process_extraction(**to_rules_engine_inputs(results["ocr"], results["vlm"]))

Only field names are renamed (e.g. vendor -> vendor_name). Values stay exactly as printed.
"""

from __future__ import annotations

from typing import Any

from .schemas import ExtractionResult

# Our field name -> rules_engine canonical field name (only where they differ).
RULES_ENGINE_NAMES = {"vendor": "vendor_name", "tax": "tax_amount", "total": "total_amount"}
KNOWN_DOCUMENT_TYPES = {"invoice", "delivery_note"}  # receipts are "invoice"


def _values(result: ExtractionResult | None) -> dict[str, str] | None:
    """Fields that have a value, with rules_engine names. None if the branch failed or is absent."""
    if result is None or not result.ok:
        return None
    return {
        RULES_ENGINE_NAMES.get(name, name): value
        for name, value in result.record().items()
        if value is not None and name != "document_type"
    }


def to_rules_engine_inputs(ocr: ExtractionResult | None, vlm: ExtractionResult | None) -> dict[str, Any]:
    """Build the keyword arguments for ``rules_engine.process_extraction``."""
    ocr_conf = None
    if ocr is not None and ocr.ok:
        # The rules engine weighs OCR *reading* confidence and does its own agreement check,
        # so pass PaddleOCR's score for the evidence text (fallback: our heuristic score).
        ocr_conf = {
            RULES_ENGINE_NAMES.get(name, name): (
                f.evidence.ocr_score if f.evidence and f.evidence.ocr_score is not None else f.confidence
            )
            for name, f in ocr.fields.items()
            if f.value is not None and name != "document_type"
        }
    # Prefer the VLM's document type, then OCR's; unknown types use the engine's "generic" rules.
    doc_type = "generic"
    for res in (vlm, ocr):
        dt = res.fields["document_type"].value if res is not None and res.ok else None
        if dt in KNOWN_DOCUMENT_TYPES:
            doc_type = dt
            break
    warnings = [w for res in (ocr, vlm) if res is not None for w in res.warnings]
    return {
        "ocr_json": _values(ocr),
        "ai_json": _values(vlm),
        "ocr_confidence": ocr_conf,
        "document_type": doc_type,
        "quality_warnings": warnings or None,
    }
