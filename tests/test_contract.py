import json

from extraction import extract_both
from extraction.ai_extractor import parse_vlm_output
from extraction.field_mapper import map_fields
from extraction.schemas import CANONICAL_FIELDS

from .conftest import PNG_BYTES, lines


def test_both_branches_share_contract(receipt):
    ocr = map_fields(receipt)
    vlm = parse_vlm_output({"total": {"value": "189.75", "confidence": 0.9, "evidence": "TOTAL"}}, "m")
    for res in (ocr, vlm):
        assert list(res.record()) == list(CANONICAL_FIELDS)
        assert set(res.fields) == set(CANONICAL_FIELDS)
        for fr in res.fields.values():
            assert 0.0 <= fr.confidence <= 1.0
            assert fr.source == res.source
            if fr.value is None:
                assert fr.confidence == 0.0 or fr.confidence_method == "caller_provided"
        json.dumps(res.to_json_dict())  # serializable
    assert ocr.record()["total"] == vlm.record()["total"] == "189.75"


def test_metadata_not_in_record():
    rec = map_fields(lines("TOTAL 1.00")).record()
    assert not {"confidence", "evidence", "source", "warnings"} & set(rec)


def test_extract_both_independent(monkeypatch, tmp_path):
    import extraction.ai_extractor as ai
    import extraction.ocr_engine as oe

    def no_model():
        raise ImportError()

    monkeypatch.setattr(oe, "get_engine", no_model)
    monkeypatch.setattr(ai, "get_engine", no_model)
    img = tmp_path / "x.png"
    img.write_bytes(PNG_BYTES)
    out = extract_both(img)
    assert out["ocr"].errors[0].code == "ocr_unavailable"
    assert out["vlm"].errors[0].code == "vlm_unavailable"
