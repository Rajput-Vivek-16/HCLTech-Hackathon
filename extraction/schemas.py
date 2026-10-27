"""Output format shared by the OCR and VLM branches.

Every field = the value exactly as printed on the document + its own confidence.
``ExtractionResult.record()`` gives just the values; ``fields`` adds the metadata.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"

Source = Literal["ocr", "vlm"]
ConfidenceMethod = Literal[
    "ocr_heuristic",  # OCR recognition score x label/position/ambiguity factors
    "token_probability",  # VLM: geometric mean of the model's token probabilities for the value
    "vlm_fallback_heuristic",  # VLM gave no usable confidence; conservative default
    "caller_provided",  # value supplied by the caller (e.g. known document type)
    "none",  # no value found -> confidence 0.0
]

# Receipts, bills and tax invoices are one type ("invoice"): same fields, same checks.
DOCUMENT_TYPES = ("invoice", "delivery_note", "form")
DOCUMENT_TYPE_ALIASES = {
    "receipt": "invoice",
    "tax_invoice": "invoice",
    "bill": "invoice",
    "cash_bill": "invoice",
    "receipt_invoice": "invoice",
    "delivery_order": "delivery_note",
}


def normalize_document_type(value: str | None) -> str | None:
    """Map any type label to one of DOCUMENT_TYPES, or None if unknown."""
    if not value:
        return None
    key = "_".join(value.strip().lower().replace("/", " ").replace("-", " ").split())
    key = DOCUMENT_TYPE_ALIASES.get(key, key)
    return key if key in DOCUMENT_TYPES else None


CANONICAL_FIELDS = (
    "document_type",
    "vendor",
    "address",
    "invoice_number",
    "order_number",
    "date",
    "currency",
    "subtotal",
    "tax_rate",
    "tax",
    "rounding_adjustment",
    "total",
)


class Evidence(BaseModel):
    text: str | None = None  # the line/text the value was read from
    bbox: list[float] | None = None  # [x_min, y_min, x_max, y_max] in pixels (OCR only)
    ocr_score: float | None = None  # PaddleOCR reading confidence of the evidence line(s), 0-1


class Candidate(BaseModel):
    value: str
    score: float
    evidence: Evidence | None = None


class FieldResult(BaseModel):
    value: str | None = None  # verbatim text from the document, never reformatted
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_method: ConfidenceMethod = "none"
    source: Source
    evidence: Evidence | None = None
    candidates: list[Candidate] = Field(default_factory=list)  # other possible values
    warnings: list[str] = Field(default_factory=list)


class ExtractionError(BaseModel):
    code: str  # invalid_input | ocr_unavailable | ocr_failed | vlm_unavailable | vlm_failed | vlm_invalid_response
    message: str


class ExtractionResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    source: Source
    engine: str  # "paddleocr" or the VLM model id
    fields: dict[str, FieldResult]  # always contains every CANONICAL_FIELDS key
    extra_fields: dict[str, str] = Field(default_factory=dict)  # unmapped fields, kept not dropped
    raw_text: str | None = None  # full OCR text (OCR branch only, in memory)
    warnings: list[str] = Field(default_factory=list)
    errors: list[ExtractionError] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def record(self) -> dict[str, str | None]:
        """Business record: field name -> value as printed (or None)."""
        return {name: self.fields[name].value for name in CANONICAL_FIELDS}

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def empty_fields(source: Source) -> dict[str, FieldResult]:
    return {name: FieldResult(source=source) for name in CANONICAL_FIELDS}


def failed_result(source: Source, engine: str, code: str, message: str) -> ExtractionResult:
    return ExtractionResult(
        source=source,
        engine=engine,
        fields=empty_fields(source),
        errors=[ExtractionError(code=code, message=message)],
    )
