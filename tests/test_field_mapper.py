from extraction.field_mapper import classify_document, map_fields
from extraction.ocr_engine import OCRResult
from extraction.schemas import CANONICAL_FIELDS

from .conftest import lines


def test_receipt_fields_verbatim(receipt):
    res = map_fields(receipt)
    assert res.record() == {
        "document_type": "invoice",
        "vendor": "BREWERY TAP SDN BHD",
        "address": "12, JALAN TAMPOI 7/4, 81200 JOHOR BAHRU, JOHOR",
        "invoice_number": None,
        "order_number": "11649",
        "date": "30/03/2018",  # original order kept
        "currency": "MYR",
        "subtotal": "179.00",
        "tax_rate": "6%",
        "tax": "10.74",
        "rounding_adjustment": "0.01",
        "total": "189.75",  # not CASH 200.00, the largest number
    }
    total = res.fields["total"]
    assert total.source == "ocr" and total.confidence_method == "ocr_heuristic"
    assert 0 < total.confidence <= 1
    assert total.evidence and "189.75" in total.evidence.text and total.evidence.bbox
    assert res.raw_text and "BREWERY" in res.raw_text


def test_date_formats_not_changed():
    assert map_fields(lines("DATE: 03-30-2018")).record()["date"] == "03-30-2018"
    assert map_fields(lines("DATE: 2018-03-30")).record()["date"] == "2018-03-30"
    assert map_fields(lines("Date 19 APR 2018")).record()["date"] == "19 APR 2018"


def test_missing_fields_are_null_with_zero_confidence():
    res = map_fields(lines("SOME SHOP", "BREAD 2.00"))
    for name in ("total", "tax", "invoice_number", "date"):
        f = res.fields[name]
        assert f.value is None and f.confidence == 0.0 and "not_found" in f.warnings


def test_empty_ocr():
    res = map_fields(OCRResult())
    assert set(res.record()) == set(CANONICAL_FIELDS)
    assert all(v is None for v in res.record().values())


def test_value_on_same_row_by_bbox():
    ocr = lines(("TOTAL", 0.99, [10, 100, 60, 115]), ("45.00", 0.99, [300, 101, 350, 116]))
    assert map_fields(ocr).record()["total"] == "45.00"


def test_competing_totals_are_flagged_ambiguous():
    f = map_fields(lines("TOTAL 10.00", "TOTAL 12.00")).fields["total"]
    assert "ambiguous" in f.warnings and len(f.candidates) == 1
    assert f.confidence < 0.7


def test_due_date_ranked_below_document_date():
    f = map_fields(lines("DUE DATE: 2018-05-01", "DATE: 2018-04-01")).fields["date"]
    assert f.value == "2018-04-01" and f.candidates[0].value == "2018-05-01"


def test_currency_symbol_kept():
    assert map_fields(lines("TOTAL $9.90")).record()["currency"] == "$"


def test_low_confidence_ocr_lowers_score():
    hi = map_fields(lines(("TOTAL 9.90", 0.99))).fields["total"]
    lo = map_fields(lines(("TOTAL 9.90", 0.30))).fields["total"]
    assert lo.confidence < hi.confidence
    assert "low_ocr_confidence" in lo.warnings


def test_invoice_number_and_doc_type():
    res = map_fields(lines("ACME LTD", "TAX INVOICE", "INVOICE NO: INV-2024/001", "BILL TO: X"))
    assert res.record()["invoice_number"] == "INV-2024/001"
    assert res.record()["document_type"] == "invoice"


def test_doc_type_unknown_and_caller_hint():
    assert classify_document(lines("hello").lines).value is None
    res = map_fields(lines("hello"), document_type="form")
    assert res.fields["document_type"].confidence_method == "caller_provided"


def test_delivery_note_classification():
    assert classify_document(lines("DELIVERY ORDER", "D/O NO 55", "RECEIVED BY").lines).value == "delivery_note"


def test_idr_receipt_without_decimals_due_label_and_item_lines():
    ocr = lines(
        "HOMEWICEWICC",
        "1 RB. COKLAT BAR",
        "8,000",
        "1 RB. CREPES TUNA",
        "9,000",
        "SUBTOTAL",
        "24,500",
        "DUE",
        "24,500",
        "CASH",
        "50,000",
        "CHANGE",
        "25,500",
    )
    rec = map_fields(ocr).record()
    assert rec["subtotal"] == "24,500" and rec["total"] == "24,500"  # not CASH 50,000
    assert rec["vendor"] == "HOMEWICEWICC"  # item lines are never the vendor


def test_due_date_is_not_a_total():
    assert map_fields(lines("DUE DATE 12,500")).record()["total"] is None


def test_vendor_split_over_two_lines_is_joined():
    rec = map_fields(lines("PASARAYA BORONG PINTAR", "SDN BHD", "BR No.: (124525-H)")).record()
    assert rec["vendor"] == "PASARAYA BORONG PINTAR SDN BHD"


def test_evidence_keeps_ocr_reading_score():
    f = map_fields(lines(("TOTAL 9.90", 0.97))).fields["total"]
    assert f.evidence.ocr_score == 0.97 and f.confidence < 0.97


def test_receipts_and_invoices_are_one_type():
    from extraction.schemas import normalize_document_type

    assert classify_document(lines("RECEIPT", "CASH 10.00", "CHANGE 0.00").lines).value == "invoice"
    assert classify_document(lines("TAX INVOICE", "BILL TO: X").lines).value == "invoice"
    assert normalize_document_type("Receipt") == normalize_document_type("tax invoice") == "invoice"
    assert normalize_document_type("passport") is None
