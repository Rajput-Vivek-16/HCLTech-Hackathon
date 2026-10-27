import pytest

from extraction.ocr_engine import extract_with_ocr, parse_paddle_output, run_ocr

from .conftest import PNG_BYTES, FakeOCREngine

V3 = [
    {
        "rec_texts": ["SHOP SDN BHD", "TOTAL 9.90", ""],
        "rec_scores": [0.97, 0.95, 0.1],
        "rec_polys": [[[0, 0], [100, 0], [100, 10], [0, 10]], [[0, 20], [100, 20], [100, 30], [0, 30]], None],
    }
]
V2 = [[[[[0, 0], [100, 0], [100, 10], [0, 10]], ("TOTAL 9.90", 0.9)], ["broken"]]]


def test_parse_v3():
    r = parse_paddle_output(V3)
    assert [ln.text for ln in r.lines] == ["SHOP SDN BHD", "TOTAL 9.90"]
    assert r.lines[1].bbox == [0.0, 20.0, 100.0, 30.0]
    assert r.lines[0].confidence == 0.97


def test_parse_v2_skips_malformed():
    r = parse_paddle_output(V2)
    assert len(r.lines) == 1 and "ocr_malformed_entry" in r.warnings


@pytest.mark.parametrize("raw", [None, [], [None], [{"rec_texts": []}]])
def test_parse_empty(raw):
    assert "ocr_empty" in parse_paddle_output(raw).warnings


def test_parse_unexpected_type_and_bad_score():
    assert "ocr_unrecognized_output" in parse_paddle_output([42]).warnings
    r = parse_paddle_output([{"rec_texts": ["A"], "rec_scores": ["x"], "rec_boxes": [[1, 2, 3, 4]]}])
    assert r.lines[0].confidence == 0.0 and r.lines[0].bbox == [1.0, 2.0, 3.0, 4.0]


def test_extract_with_ocr_end_to_end(tmp_path):
    img = tmp_path / "doc.png"
    img.write_bytes(PNG_BYTES)
    res = extract_with_ocr(img, engine=FakeOCREngine(V3))
    assert res.ok and str(res.record()["total"]) == "9.90" and res.source == "ocr"


def test_extract_with_ocr_errors(tmp_path):
    assert extract_with_ocr(tmp_path / "missing.png", engine=FakeOCREngine()).errors[0].code == "invalid_input"
    img = tmp_path / "doc.png"
    img.write_bytes(PNG_BYTES)
    res = extract_with_ocr(img, engine=FakeOCREngine(exc=RuntimeError("boom")))
    assert res.errors[0].code == "ocr_failed"
    assert all(v is None for v in res.record().values())


def test_run_ocr_numpy_input():
    import numpy as np

    eng = FakeOCREngine(V3)
    assert len(run_ocr(np.zeros((20, 20, 3), dtype=np.uint8), engine=eng).lines) == 2


@pytest.mark.integration
def test_real_paddleocr_smoke(tmp_path):
    pytest.importorskip("paddleocr")
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (400, 120), "white")
    ImageDraw.Draw(img).text((10, 40), "TOTAL 12.50", fill="black")
    path = tmp_path / "synthetic.png"
    img.save(path)
    res = extract_with_ocr(path)
    assert res.ok
