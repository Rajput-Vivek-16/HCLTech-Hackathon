import fitz
import pytest

from extraction.inputs import load_document
from extraction.ocr_engine import extract_with_ocr

from .conftest import PNG_BYTES, FakeOCREngine


def _pdf(pages: int) -> bytes:
    doc = fitz.open()
    for i in range(pages):
        doc.new_page(width=200, height=100).insert_text((20, 50), f"TOTAL {i + 1}.00")
    data = doc.tobytes()
    doc.close()
    return data


def test_member1_payload_with_warnings():
    payload = {"content": PNG_BYTES, "mime_type": "image/png", "warnings": ["Image may be blurry"]}
    doc = load_document(payload)
    assert len(doc.pages) == 1 and doc.warnings == ["input: Image may be blurry"]


def test_pdf_pages_rendered_and_capped(monkeypatch):
    monkeypatch.setenv("EXTRACTION_MAX_PAGES", "2")
    doc = load_document(_pdf(3))
    assert len(doc.pages) == 2 and doc.pages[0].mode == "RGB"
    assert any(w.startswith("pdf_truncated") for w in doc.warnings)


@pytest.mark.parametrize("bad", [b"%PDF-1.7 broken", b"not an image", {"content": b""}, 42])
def test_bad_input_is_value_error(bad):
    with pytest.raises((ValueError, OSError)):
        load_document(bad)


def test_ocr_on_pdf_payload_stacks_page_boxes():
    raw = [{"rec_texts": ["TOTAL 9.90"], "rec_scores": [0.99], "rec_boxes": [[0, 10, 100, 20]]}]
    payload = {"content": _pdf(2), "mime_type": "application/pdf", "warnings": []}
    res = extract_with_ocr(payload, engine=FakeOCREngine(raw))
    assert res.ok
    boxes = [c.evidence.bbox for c in [res.fields["total"], *res.fields["total"].candidates] if c.evidence]
    assert res.record()["total"] == "9.90" and boxes[0][1] == 10.0  # page 1 box unshifted
