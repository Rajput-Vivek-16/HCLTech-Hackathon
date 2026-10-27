from extraction.ai_extractor import parse_vlm_output
from extraction.field_mapper import map_fields
from extraction.handoff import to_rules_engine_inputs
from extraction.schemas import failed_result
from rules_engine import process_extraction


def test_names_renamed_values_unchanged(receipt):
    ocr = map_fields(receipt)
    vlm = parse_vlm_output({"vendor": {"value": "BREWERY TAP", "confidence": 0.9}, "total": {"value": "189.75"}}, "m")
    kw = to_rules_engine_inputs(ocr, vlm)
    assert kw["ocr_json"]["vendor_name"] == "BREWERY TAP SDN BHD"
    assert kw["ocr_json"]["total_amount"] == "189.75" and kw["ocr_json"]["date"] == "30/03/2018"
    assert kw["ai_json"] == {"vendor_name": "BREWERY TAP", "total_amount": "189.75"}
    assert set(kw["ocr_confidence"]) == set(kw["ocr_json"])
    assert kw["document_type"] == "invoice"


def test_failed_branch_is_none_and_engine_accepts_it(receipt):
    kw = to_rules_engine_inputs(map_fields(receipt), failed_result("vlm", "m", "vlm_timeout", "x"))
    assert kw["ai_json"] is None
    doc = process_extraction(**kw).to_dict()
    assert doc["fields"]["date"]["normalized_value"] == "2018-03-30"
    assert doc["document_decision"]  # Member 3 decides; we only check it runs


def test_ocr_confidence_passed_is_the_reading_score(receipt):
    kw = to_rules_engine_inputs(map_fields(receipt), None)
    assert kw["ocr_confidence"]["total_amount"] == 0.98  # PaddleOCR score, not the mapper's ranking score


def test_review_hides_optional_fields_absent_everywhere(receipt):
    from rules_integration import review_fields_from_rules

    doc = process_extraction(**to_rules_engine_inputs(map_fields(receipt), None))
    fields = review_fields_from_rules(doc)
    assert "discount_amount" not in fields  # not printed, not required
    assert {"vendor_name", "date", "total_amount"} <= set(fields)  # required ones always shown
