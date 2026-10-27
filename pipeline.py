"""End-to-end glue: Member 1 payload -> Member 2 extraction -> Member 3 rules engine.

Everything stays in memory; nothing is written to disk. Used by app.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from extraction import extract_with_ocr, extract_with_vlm
from extraction.handoff import RULES_ENGINE_NAMES, to_rules_engine_inputs
from extraction.schemas import ExtractionResult
from rules_engine import process_extraction, revalidate_corrected_fields

OUR_NAMES = {v: k for k, v in RULES_ENGINE_NAMES.items()}  # rules name -> extraction name


def _plain(value: Any) -> Any:
    """Enum -> its string value; everything else unchanged."""
    return getattr(value, "value", value)


@dataclass
class PipelineRun:
    ocr: ExtractionResult | None
    vlm: ExtractionResult | None
    rules: Any  # rules_engine DocumentResult
    timings: dict[str, float] = field(default_factory=dict)
    corrections: dict[str, str] = field(default_factory=dict)

    @property
    def decision(self) -> str:
        return str(_plain(self.rules.to_dict()["document_decision"]))


def run_pipeline(
    payload: Any,
    *,
    use_ocr: bool = True,
    use_vlm: bool = True,
    document_type: str | None = None,
) -> PipelineRun:
    """Run the selected extraction branches on one document, then the rules engine."""
    timings: dict[str, float] = {}
    ocr = vlm = None
    if use_ocr:
        t0 = time.perf_counter()
        ocr = extract_with_ocr(payload, document_type=document_type)
        timings["ocr"] = time.perf_counter() - t0
    if use_vlm:
        t0 = time.perf_counter()
        vlm = extract_with_vlm(payload, document_type=document_type)
        timings["vlm"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    rules = process_extraction(**to_rules_engine_inputs(ocr, vlm))
    timings["rules"] = time.perf_counter() - t0
    return PipelineRun(ocr=ocr, vlm=vlm, rules=rules, timings=timings)


def _branch_field(result: ExtractionResult | None, name: str) -> tuple[str | None, float | None]:
    if result is None or not result.ok or name not in result.fields:
        return None, None
    f = result.fields[name]
    return f.value, (f.confidence if f.value is not None else None)


def review_rows(run: PipelineRun) -> list[dict[str, Any]]:
    """One row per field for the dashboard: both branches + the rules engine verdict."""
    rules_fields = run.rules.to_dict()["fields"]
    rows = []
    for name, f in rules_fields.items():
        ours = OUR_NAMES.get(name, name)
        ocr_v, ocr_c = _branch_field(run.ocr, ours)
        vlm_v, vlm_c = _branch_field(run.vlm, ours)
        if f.get("normalized_value") is None and ocr_v is None and vlm_v is None and not f.get("evidence_missing"):
            continue  # optional field absent everywhere
        reasons = list(f.get("validation_reasons") or []) + list(f.get("decision_reasons") or [])
        rows.append(
            {
                "field": name,
                "ocr_value": ocr_v,
                "ocr_conf": ocr_c,
                "vlm_value": vlm_v,
                "vlm_conf": vlm_c,
                "value": f.get("normalized_value"),
                "confidence": f.get("confidence_score"),
                "validation": str(_plain(f.get("validation_status"))),
                "agreement": str(_plain(f.get("agreement_status"))),
                "decision": str(_plain(f.get("decision"))),
                "reasons": "; ".join(dict.fromkeys(str(r) for r in reasons)),
            }
        )
    # Extracted fields the rules engine does not validate (shown, marked as such).
    for ours in ("document_type", "invoice_number", "order_number"):
        name = RULES_ENGINE_NAMES.get(ours, ours)
        if name in rules_fields:
            continue
        ocr_v, ocr_c = _branch_field(run.ocr, ours)
        vlm_v, vlm_c = _branch_field(run.vlm, ours)
        if ocr_v is None and vlm_v is None:
            continue
        rows.append(
            {
                "field": name,
                "ocr_value": ocr_v,
                "ocr_conf": ocr_c,
                "vlm_value": vlm_v,
                "vlm_conf": vlm_c,
                "value": vlm_v if vlm_v is not None else ocr_v,
                "confidence": vlm_c if vlm_v is not None else ocr_c,
                "validation": "not_validated",
                "agreement": "match" if ocr_v == vlm_v else ("mismatch" if ocr_v and vlm_v else "single_source"),
                "decision": "info",
                "reasons": "Extracted only; not checked by the rules engine.",
            }
        )
    return rows


def published_fields(run: PipelineRun) -> dict[str, dict[str, Any]]:
    """Shape expected by app.publish_extraction_result: {field: {value, confidence, status}}."""
    return {
        r["field"]: {"value": r["value"], "confidence": r["confidence"], "status": r["decision"]}
        for r in review_rows(run)
    }


def revalidate(run: PipelineRun, corrections: dict[str, str]) -> PipelineRun:
    """Apply reviewer corrections and re-run the rules engine checks (Member 3)."""
    run.rules = revalidate_corrected_fields(run.rules, corrections)
    run.corrections.update(corrections)
    return run


def export_record(run: PipelineRun) -> dict[str, Any]:
    """Final record (field -> value) plus the decision, for download."""
    return {
        "decision": run.decision,
        "fields": {r["field"]: r["value"] for r in review_rows(run)},
        "corrected_fields": sorted(run.corrections),
    }
