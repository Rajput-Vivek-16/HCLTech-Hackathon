import hashlib
import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from preprocess import MAX_PDF_PAGES, process_upload


def _png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _text_image(w=1000, h=1400):
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    for y in range(100, h - 100, 40):
        d.line([(80, y), (w - 80, y)], fill="black", width=3)
    return img


def _pdf_bytes(pages=1, password=None):
    doc = fitz.open()
    for _ in range(pages):
        doc.new_page().insert_text((72, 72), "hello")
    kwargs = {}
    if password:
        kwargs = dict(encryption=fitz.PDF_ENCRYPT_AES_256, user_pw=password, owner_pw=password)
    data = doc.tobytes(**kwargs)
    doc.close()
    return data


def test_png_gives_png_and_hash():
    raw = _png_bytes(_text_image())
    doc = process_upload(raw, "a.png")
    assert doc.ocr_bytes.startswith(b"\x89PNG")
    assert doc.sha256 == hashlib.sha256(raw).hexdigest()
    assert doc.original_bytes == raw


def test_large_image_resized_keeping_aspect():
    raw = _png_bytes(_text_image(3000, 4000))
    doc = process_upload(raw, "big.png", long_edge=1920)
    p = doc.pages[0]
    assert max(p.width, p.height) <= 1920 + 5  # allow tiny deskew expansion


def test_pdf_passthrough():
    raw = _pdf_bytes()
    doc = process_upload(raw, "a.pdf")
    assert doc.ocr_bytes == doc.original_bytes == raw
    assert doc.page_count == 1 and doc.is_pdf and not doc.pages


def test_corrupt_pdf_rejected():
    with pytest.raises(ValueError):
        process_upload(b"%PDF-1.4 garbage", "bad.pdf")


def test_encrypted_pdf_rejected():
    with pytest.raises(ValueError, match="password"):
        process_upload(_pdf_bytes(password="secret"), "locked.pdf")


def test_too_many_pages_rejected():
    with pytest.raises(ValueError, match="pages"):
        process_upload(_pdf_bytes(pages=MAX_PDF_PAGES + 1), "long.pdf")


def test_corrupt_image_rejected():
    with pytest.raises(ValueError):
        process_upload(b"\x89PNG\r\n\x1a\nnot really", "bad.png")


def test_extension_content_mismatch_rejected():
    with pytest.raises(ValueError):
        process_upload(_png_bytes(_text_image()), "fake.pdf")


def test_empty_and_unsupported():
    with pytest.raises(ValueError):
        process_upload(b"", "a.png")
    with pytest.raises(ValueError):
        process_upload(b"hello", "a.txt")


def test_oversize_rejected():
    raw = _png_bytes(_text_image())
    with pytest.raises(ValueError):
        process_upload(raw, "a.png", max_size_mb=0.0001)


def test_blur_warning():
    flat = Image.new("RGB", (1000, 1400), "white")
    doc = process_upload(_png_bytes(flat), "flat.png")
    assert any("blurry" in w for w in doc.warnings)


def test_low_resolution_warning():
    doc = process_upload(_png_bytes(_text_image(300, 400)), "small.png")
    assert any("Low resolution" in w for w in doc.warnings)


def test_skew_detected_and_corrected():
    tilted = _text_image().rotate(-5, expand=True, fillcolor="white")
    doc = process_upload(_png_bytes(tilted), "tilt.png")
    angle = doc.pages[0].deskew_angle
    assert 3.0 <= abs(angle) <= 7.0, f"estimated angle {angle}"
    # Sign check: a second pass on the output should find near-zero skew.
    again = process_upload(doc.ocr_bytes, "again.png")
    assert abs(again.pages[0].deskew_angle) < abs(angle)


def test_deskew_toggle_off():
    tilted = _text_image().rotate(-5, expand=True, fillcolor="white")
    doc = process_upload(_png_bytes(tilted), "tilt.png", deskew=False)
    assert doc.pages[0].deskew_angle == 0.0


def test_exif_rotation_applied():
    img = Image.new("RGB", (400, 800), "white")
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW to display
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    doc = process_upload(buf.getvalue(), "phone.jpg", deskew=False)
    assert doc.pages[0].original_image.size == (800, 400)


def test_transparent_png_not_black():
    rgba = Image.new("RGBA", (800, 800), (0, 0, 0, 0))
    doc = process_upload(_png_bytes(rgba), "t.png", deskew=False)
    px = np.asarray(doc.pages[0].processed_image)[0, 0]
    assert tuple(px) == (255, 255, 255)


def test_pdf_preview_renders_pages_without_touching_ocr_bytes():
    from preprocess import render_pdf_preview

    raw = _pdf_bytes(pages=2)
    imgs = render_pdf_preview(raw)
    assert len(imgs) == 2
    assert all(im.width > 100 and im.height > 100 for im in imgs)
    doc = process_upload(raw, "a.pdf")
    assert doc.ocr_bytes == raw  # preview must not change the OCR payload


def test_pdf_preview_rejects_encrypted():
    from preprocess import render_pdf_preview

    with pytest.raises(ValueError):
        render_pdf_preview(_pdf_bytes(password="secret"))


# ---------------------------------------------------------------- new: noise / orientation
def _real_text_page(w=1000, h=1400):
    from PIL import ImageFont

    img = Image.new("RGB", (w, h), (235, 235, 235))
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=30)
    lines = [
        "TAX INVOICE",
        "ABC TRADING SDN BHD",
        "Invoice No: INV-20418",
        "Date: 14/03/2018",
        "Item A 2 x 5.00 10.00",
        "Item B 1 x 10.00 10.00",
        "Subtotal 20.00",
        "GST 6% 1.20",
        "TOTAL RM 21.20",
        "Thank you for shopping",
    ] * 3
    for i, line in enumerate(lines):
        d.text((80, 80 + i * 40), line, fill="black", font=font)
    return img


def _add_noise(img, sigma, seed=0):
    arr = np.asarray(img).astype(np.float32)
    noisy = arr + np.random.default_rng(seed).normal(0, sigma, arr.shape)
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))


def test_noise_estimator_separates_clean_and_noisy():
    from preprocess import NOISE_SIGMA_THRESHOLD, _estimate_noise_sigma

    page = _real_text_page()
    assert _estimate_noise_sigma(page) < NOISE_SIGMA_THRESHOLD
    assert _estimate_noise_sigma(_add_noise(page, 12)) > NOISE_SIGMA_THRESHOLD


def test_noisy_image_is_denoised_and_clean_one_is_not():
    from preprocess import _estimate_noise_sigma

    page = _real_text_page()
    noisy = process_upload(_png_bytes(_add_noise(page, 12)), "n.png", deskew=False, auto_orient=False)
    clean = process_upload(_png_bytes(page), "c.png", deskew=False, auto_orient=False)
    assert noisy.pages[0].denoised and not clean.pages[0].denoised
    assert _estimate_noise_sigma(noisy.pages[0].processed_image) < noisy.pages[0].noise_sigma / 2
    assert any("Noise reduced" in n for n in noisy.notes)


def test_denoise_can_be_turned_off():
    page = _add_noise(_real_text_page(), 12)
    doc = process_upload(_png_bytes(page), "n.png", deskew=False, auto_orient=False, denoise=False)
    assert not doc.pages[0].denoised


def test_manual_rotation_is_clockwise():
    img = Image.new("RGB", (800, 1000), "white")
    ImageDraw.Draw(img).rectangle([0, 0, 100, 100], fill=(255, 0, 0))  # red, top-left
    doc = process_upload(_png_bytes(img), "m.png", auto_orient=False, deskew=False, manual_rotation=90)
    out = np.asarray(doc.pages[0].processed_image)
    assert out.shape[:2] == (800, 1000)  # width/height swapped
    assert tuple(out[10, -10]) == (255, 0, 0)  # red moved to top-right
    assert doc.pages[0].rotation_applied == 90 and doc.pages[0].orientation_method == "manual"


def test_invalid_manual_rotation_rejected():
    with pytest.raises(ValueError):
        process_upload(_png_bytes(_text_image()), "a.png", manual_rotation=45)


@pytest.mark.skipif(__import__("shutil").which("tesseract") is None, reason="Tesseract not installed")
@pytest.mark.parametrize("cw", [90, 180, 270])
def test_auto_orient_fixes_rotated_text_page(cw):
    page = _real_text_page()
    rotated = page.rotate(-cw, expand=True, fillcolor=(235, 235, 235))  # rotate CLOCKWISE by cw
    doc = process_upload(_png_bytes(rotated), "r.png", deskew=False, denoise=False)
    pg = doc.pages[0]
    assert pg.rotation_applied == (360 - cw) % 360
    assert pg.processed_image.size == page.size


@pytest.mark.skipif(__import__("shutil").which("tesseract") is None, reason="Tesseract not installed")
def test_upright_page_not_rotated():
    doc = process_upload(_png_bytes(_real_text_page()), "u.png", deskew=False, denoise=False)
    assert doc.pages[0].rotation_applied == 0


def test_payload_includes_preprocessing_summary():
    doc = process_upload(_png_bytes(_text_image()), "a.png", auto_orient=False)
    assert "rotation_applied" in doc.ocr_payload()["preprocessing"]
    assert process_upload(_pdf_bytes(), "a.pdf").ocr_payload()["preprocessing"] == {}
