from __future__ import annotations

from typing import Any

import pytest

from extraction.ocr_engine import OCRLine, OCRResult


def _png() -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format="PNG")
    return buf.getvalue()


PNG_BYTES = _png()  # tiny blank synthetic image


def lines(*rows: str | tuple[str, float] | tuple[str, float, list[float]]) -> OCRResult:
    out = []
    for i, r in enumerate(rows):
        if isinstance(r, str):
            r = (r, 0.98)
        text, conf = r[0], r[1]
        bbox = r[2] if len(r) > 2 else [10.0, 20.0 * i, 200.0, 20.0 * i + 15]  # type: ignore[misc]
        out.append(OCRLine(text=text, confidence=conf, bbox=bbox))
    return OCRResult(lines=out)


@pytest.fixture
def receipt() -> OCRResult:
    return lines(
        "BREWERY TAP SDN BHD",
        "12, JALAN TAMPOI 7/4,",
        "81200 JOHOR BAHRU, JOHOR",
        "TEL: 07-1234567",
        "TAX INVOICE",
        "ORDER NO: 11649",
        "DATE: 30/03/2018 21:14",
        "BEER 2 x 89.50   179.00",
        "SUB TOTAL   179.00",
        "GST 6%   10.74",
        "ROUNDING   0.01",
        "TOTAL (MYR)   189.75",
        "CASH   200.00",
        "CHANGE   10.25",
        "THANK YOU",
    )


class FakeOCREngine:
    def __init__(self, raw: Any = None, exc: Exception | None = None) -> None:
        self.raw, self.exc = raw, exc

    def predict(self, img: Any) -> Any:
        if self.exc:
            raise self.exc
        return self.raw


class FakeEngine:
    """Stands in for the local Qwen2-VL model in tests."""

    name = "fake-vlm"

    def __init__(self, text: str = "", exc: Exception | None = None, probs: bool = True) -> None:
        self.text, self.exc, self.probs = text, exc, probs
        self.calls = 0

    def generate(self, image: Any, prompt: str) -> Any:
        from extraction.ai_extractor import VLMOutput

        self.calls += 1
        if self.exc:
            raise self.exc
        spans = [(i, i + 1, 0.9) for i in range(len(self.text))] if self.probs else []
        return VLMOutput(text=self.text, token_spans=spans)
