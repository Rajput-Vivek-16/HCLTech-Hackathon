"""VLM branch: Qwen2-VL-2B-Instruct running LOCALLY. The image never leaves the machine.

The model reads the image and writes a JSON object with the canonical fields.
Values are copied as printed. Per-field confidence = geometric mean of the model's
own probabilities for the tokens of that value ("token_probability"): how sure the
model was while writing it. It is not a calibrated accuracy.

Heavy dependencies (torch, transformers) are optional: pip install -e ".[vlm]".
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from .patterns import canonical_key
from .schemas import (
    CANONICAL_FIELDS,
    ExtractionResult,
    FieldResult,
    empty_fields,
    failed_result,
    normalize_document_type,
)

log = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
FALLBACK_CONFIDENCE = 0.3  # used when token probabilities are unavailable
EMPTY_VALUES = {"", "null", "none", "n/a", "unknown"}

PROMPT = f"""Read this business document image. Return ONLY a JSON object with exactly these keys:
{", ".join(CANONICAL_FIELDS)}.
Rules:
- document_type is one of: invoice (use this for receipts, bills and tax invoices too), delivery_note, form
  (or null if unclear).
- vendor is the full company name that issued the document; address is the vendor's address.
- date is the document date.
- total is the final bill amount (TOTAL / GRAND TOTAL / AMOUNT DUE), NOT the cash given or change.
- subtotal is the amount before tax; tax is the tax AMOUNT (e.g. 0.00), never the total.
- invoice_number only if labelled invoice/bill/receipt number; company registration numbers are not invoices.
- Copy every other value EXACTLY as printed (same date order, number format, currency symbol).
- Use null for anything not printed on the document. Do not guess or calculate."""


@dataclass
class VLMOutput:
    """Raw model output: generated text and (start, end, probability) per generated token."""

    text: str
    token_spans: list[tuple[int, int, float]] = field(default_factory=list)


class VLMEngine(Protocol):
    name: str

    def generate(self, image: Any, prompt: str) -> VLMOutput: ...


class QwenEngine:
    """Loads Qwen2-VL once and reuses it. Weights are cached by Hugging Face under
    ~/.cache/huggingface (model files only, no document data)."""

    def __init__(self, model_id: str | None = None) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.name = model_id or os.getenv("VLM_MODEL") or DEFAULT_MODEL
        self._torch = torch
        dtype = getattr(torch, os.getenv("VLM_DTYPE", "bfloat16"))
        device = os.getenv("VLM_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        max_pixels = int(os.getenv("VLM_MAX_PIXELS", str(448 * 28 * 28)))  # caps image tokens (~768)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(self.name, torch_dtype=dtype).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(self.name, min_pixels=128 * 28 * 28, max_pixels=max_pixels)
        self.device = device
        self.max_new_tokens = int(os.getenv("VLM_MAX_NEW_TOKENS", "256"))

    def generate(self, image: Any, prompt: str) -> VLMOutput:
        torch = self._torch
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        chat = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[chat], images=[image], return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        gen_ids = out.sequences[0, inputs["input_ids"].shape[1] :].tolist()
        probs = [
            float(torch.softmax(score[0].float(), dim=-1)[tok]) for score, tok in zip(out.scores, gen_ids, strict=False)
        ]
        tok = self.processor.tokenizer
        spans: list[tuple[int, int, float]] = []
        prev = 0
        for i, p in enumerate(probs):  # map each token to its character span in the text
            end = len(tok.decode(gen_ids[: i + 1], skip_special_tokens=True))
            spans.append((prev, end, p))
            prev = end
        return VLMOutput(text=tok.decode(gen_ids, skip_special_tokens=True), token_spans=spans)


_ENGINE: VLMEngine | None = None


def get_engine() -> VLMEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = QwenEngine()
    return _ENGINE


class VLMField(BaseModel):
    model_config = ConfigDict(extra="ignore")
    value: str | None = None

    @field_validator("value", mode="before")
    @classmethod
    def _text_only(cls, v: Any) -> Any:
        if isinstance(v, bool | list | dict):
            raise ValueError("expected text")
        return str(v) if isinstance(v, int | float) else v


def extract_json(text: str) -> Any:
    """Pull the first JSON object out of the model text (handles ```json fences)."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def value_confidence(output: VLMOutput, key: str, value: str) -> float | None:
    """Geometric mean probability of the tokens that wrote ``value`` for ``key``."""
    if not output.token_spans:
        return None
    m = re.search(rf'"{re.escape(key)}"\s*:\s*"?', output.text)
    if not m:
        return None
    start = output.text.find(value, m.end()) if value else -1
    if start == -1:
        return None
    end = start + len(value)
    probs = [p for s, e, p in output.token_spans if s < end and e > start and p > 0]
    if not probs:
        return None
    return round(math.exp(sum(math.log(p) for p in probs) / len(probs)), 4)


def parse_vlm_output(data: Any, model: str, output: VLMOutput | None = None) -> ExtractionResult:
    """Validate the model's JSON field by field and convert it to the shared format."""
    if not isinstance(data, dict):
        return failed_result("vlm", model, "vlm_invalid_response", "model did not return a JSON object")
    result = ExtractionResult(source="vlm", engine=model, fields=empty_fields("vlm"))
    seen: set[str] = set()
    for key, raw in data.items():
        name = canonical_key(key)
        if name not in CANONICAL_FIELDS or name in seen:
            result.extra_fields[key] = str(raw)[:200]
            continue
        seen.add(name)
        fr = result.fields[name]
        try:
            vf = VLMField.model_validate(raw if isinstance(raw, dict) else {"value": raw})
        except ValidationError:
            fr.warnings.append("invalid_type")
            continue
        value = vf.value.strip() if vf.value else None
        if value is None or value.lower() in EMPTY_VALUES:
            fr.warnings.append("not_found")
            continue
        if name == "document_type":
            value = normalize_document_type(value)
            if value is None:
                fr.warnings.append("unsupported_value")
                continue
        fr.value = value
        conf = value_confidence(output, key, value) if output else None
        if conf is None:
            fr.confidence, fr.confidence_method = FALLBACK_CONFIDENCE, "vlm_fallback_heuristic"
            fr.warnings.append("missing_confidence")
        else:
            fr.confidence, fr.confidence_method = conf, "token_probability"
    for name in CANONICAL_FIELDS:
        if name not in seen:
            result.fields[name].warnings.append("missing_in_response")
    return result


def extract_with_vlm(
    image: Any,
    *,
    engine: VLMEngine | None = None,
    document_type: str | None = None,
) -> ExtractionResult:
    """Public VLM-branch entry point (local Qwen2-VL). Accepts the same inputs as OCR
    (see extraction/inputs.py); for PDFs only the first page is read. Never raises for
    model problems; see ``result.errors``."""
    model = (engine.name if engine else None) or os.getenv("VLM_MODEL") or DEFAULT_MODEL
    from .inputs import load_document

    try:
        doc = load_document(image)
    except (FileNotFoundError, ValueError, OSError) as exc:
        return failed_result("vlm", model, "invalid_input", str(exc))
    img = doc.pages[0]
    try:
        engine = engine or get_engine()
    except ImportError:
        return failed_result(
            "vlm", model, "vlm_unavailable", 'torch/transformers not installed: pip install -e ".[vlm]"'
        )
    except Exception as exc:  # model download/load problems
        log.warning("vlm load failed: %s", type(exc).__name__)
        return failed_result("vlm", model, "vlm_unavailable", type(exc).__name__)
    try:
        output = engine.generate(img, PROMPT)
    except Exception as exc:  # e.g. out of memory
        log.warning("vlm generation failed: %s", type(exc).__name__)
        return failed_result("vlm", model, "vlm_failed", type(exc).__name__)

    result = parse_vlm_output(extract_json(output.text), engine.name, output)
    result.warnings = doc.warnings + result.warnings
    if len(doc.pages) > 1:
        result.warnings.append("vlm_first_page_only")
    if document_type and result.ok and normalize_document_type(document_type):
        result.fields["document_type"] = FieldResult(
            source="vlm",
            value=normalize_document_type(document_type),
            confidence=1.0,
            confidence_method="caller_provided",
        )
    log.info("vlm finished: %d fields with values", sum(f.value is not None for f in result.fields.values()))
    return result
