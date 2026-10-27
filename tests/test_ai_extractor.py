import json

import pytest

import extraction.ai_extractor as ai
from extraction.ai_extractor import VLMOutput, extract_json, extract_with_vlm, parse_vlm_output, value_confidence

from .conftest import PNG_BYTES, FakeEngine

VALID = {
    "document_type": "invoice",
    "vendor": "BREWERY TAP",
    "address": None,
    "invoice_number": None,
    "order_number": "11649",
    "date": "30/03/2018",
    "currency": "RM",
    "subtotal": "179.00",
    "tax_rate": "6%",
    "tax": "10.74",
    "rounding_adjustment": "0.01",
    "total": "189.75",
}


def test_valid_output_values_verbatim_with_token_confidence():
    eng = FakeEngine("```json\n" + json.dumps(VALID) + "\n```")
    res = extract_with_vlm(PNG_BYTES, engine=eng)
    assert res.ok and res.source == "vlm" and res.engine == "fake-vlm"
    assert res.record() == VALID
    t = res.fields["total"]
    assert t.confidence == 0.9 and t.confidence_method == "token_probability"
    assert res.fields["address"].confidence == 0.0


def test_value_confidence_uses_only_that_values_tokens():
    text = '{"vendor": "ACME", "total": "9.90"}'
    pos = text.index("9.90")
    spans = [(i, i + 1, 0.5 if pos <= i < pos + 4 else 0.99) for i in range(len(text))]
    out = VLMOutput(text=text, token_spans=spans)
    assert value_confidence(out, "total", "9.90") == 0.5
    assert value_confidence(out, "vendor", "ACME") == 0.99
    assert value_confidence(VLMOutput(text=text), "total", "9.90") is None


def test_no_probabilities_falls_back():
    res = extract_with_vlm(PNG_BYTES, engine=FakeEngine(json.dumps({"total": "5.00"}), probs=False))
    t = res.fields["total"]
    assert t.confidence_method == "vlm_fallback_heuristic" and "missing_confidence" in t.warnings


def test_missing_keys_and_bad_types():
    res = parse_vlm_output(
        {"vendor": ["a"], "subtotal": 12.5, "total": "null", "document_type": "passport", "x": 1}, "m"
    )
    assert res.fields["vendor"].value is None and "invalid_type" in res.fields["vendor"].warnings
    assert res.fields["subtotal"].value == "12.5"
    assert res.fields["total"].value is None and "not_found" in res.fields["total"].warnings
    assert res.fields["document_type"].value is None and "unsupported_value" in res.fields["document_type"].warnings
    assert "missing_in_response" in res.fields["date"].warnings
    assert res.extra_fields == {"x": "1"}


def test_aliases():
    assert parse_vlm_output({"company": "UNIHAKKA"}, "m").record()["vendor"] == "UNIHAKKA"


@pytest.mark.parametrize("text", ["no json here", "{broken", "[1, 2]"])
def test_invalid_model_text(text):
    res = extract_with_vlm(PNG_BYTES, engine=FakeEngine(text))
    assert res.errors[0].code == "vlm_invalid_response"


def test_extract_json_handles_fences():
    assert extract_json('Here:\n```json\n{"a": 1}\n```') == {"a": 1}


def test_generation_failure():
    res = extract_with_vlm(PNG_BYTES, engine=FakeEngine(exc=RuntimeError("out of memory")))
    assert res.errors[0].code == "vlm_failed" and all(v is None for v in res.record().values())


def test_model_unavailable(monkeypatch):
    def boom():
        raise ImportError("no torch")

    monkeypatch.setattr(ai, "get_engine", boom)
    assert extract_with_vlm(PNG_BYTES).errors[0].code == "vlm_unavailable"


def test_input_validation(tmp_path):
    eng = FakeEngine("{}")
    assert extract_with_vlm(b"not an image", engine=eng).errors[0].code == "invalid_input"
    assert extract_with_vlm(tmp_path / "nope.png", engine=eng).errors[0].code == "invalid_input"
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.7")
    assert extract_with_vlm(pdf, engine=eng).errors[0].code == "invalid_input"
    assert eng.calls == 0


def test_caller_document_type():
    res = extract_with_vlm(PNG_BYTES, engine=FakeEngine(json.dumps(VALID)), document_type="invoice")
    assert res.fields["document_type"].value == "invoice"
    assert res.fields["document_type"].confidence_method == "caller_provided"
