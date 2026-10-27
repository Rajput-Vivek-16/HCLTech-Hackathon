"""Map OCR lines to canonical fields. Values are copied exactly as printed.

How a value is chosen: find a label (e.g. "SUB TOTAL", "GST", "DATE"), then take the
value on the same line, on the same row (by bounding box), or on the next line.
Layouts differ, so every field keeps its alternatives in ``candidates``.

Confidence (``ocr_heuristic``) for a candidate =
    OCR recognition score (min of label line and value line)
    x label weight      (how specific the label is, e.g. GRAND TOTAL 1.0, TOTAL 0.85)
    x position factor   (same line 1.0, same row 0.95, next line 0.85)
    x 0.6 if a different value scores >= 80% of the best (ambiguous)
It is a ranking score in [0, 1], not a probability. No validation or routing here.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from .ocr_engine import ENGINE_NAME, OCRLine, OCRResult
from .patterns import MONEY_RE, PERCENT_RE, find_amounts, find_currencies, find_dates
from .schemas import Candidate, Evidence, ExtractionResult, FieldResult, empty_fields, normalize_document_type

LOW_OCR_CONFIDENCE = 0.5
AMBIGUITY_RATIO = 0.8

Label = tuple[re.Pattern[str], float]


def _rx(p: str) -> re.Pattern[str]:
    return re.compile(p, re.I)


MONEY_LABELS: dict[str, list[Label]] = {
    "total": [
        (_rx(r"\bGRAND\s*TOTAL\b"), 1.0),
        (_rx(r"\bROUNDED\s*TOTAL\b"), 1.0),
        (_rx(r"\bTOTAL\s*(AMOUNT|AMT|DUE|PAYABLE)\b"), 1.0),
        (_rx(r"\bAMOUNT\s*DUE\b"), 0.95),
        (_rx(r"^\s*(BALANCE\s*)?DUE\b(?!\s*DATE)"), 0.9),
        (_rx(r"\bNET+\s*(TOTAL|AMOUNT)\b"), 0.95),
        (_rx(r"\bTOTAL\b"), 0.85),
    ],
    "subtotal": [
        (_rx(r"\bSUB\s*-?\s*TOTAL\b"), 1.0),
        (_rx(r"\bTOTAL\s*(EXCL\w*|BEFORE)\b"), 0.9),
    ],
    "tax": [(_rx(r"\b(GST|SST|VAT|TAX)\b"), 0.9)],
    "rounding_adjustment": [(_rx(r"\bROUND(ING)?\b"), 1.0)],
}
MONEY_EXCLUDE: dict[str, re.Pattern[str]] = {
    "total": _rx(r"SUB\s*-?\s*TOTAL|TOTAL\s*(QTY|QUANTITY|ITEMS?|DISC\w*|SAVING\w*|EXCL\w*|BEFORE)|ROUND"),
    "subtotal": _rx(r"$^"),
    "tax": _rx(r"TAX\s*INVOICE|(GST|SST|TAX|VAT)\s*(ID|REG\w*|NO\b|NUM)|\bTOTAL\b|EXCL"),
    "rounding_adjustment": _rx(r"$^"),
}
ANY_MONEY_LABEL = [rx for labels in MONEY_LABELS.values() for rx, _ in labels]
ID_LABELS = {
    "invoice_number": _rx(r"\b(?:TAX\s+)?(?:INVOICE|INV|BILL)\s*(?:NO|NUMBER|NUM|#)\.?\s*[:#.]?\s*([A-Z0-9][\w\-/]*)?"),
    "order_number": _rx(r"\bORDER\s*(?:NO|NUMBER|ID|#)\.?\s*[:#.]?\s*([A-Z0-9][\w\-/]*)?"),
}
DATE_LABEL = _rx(r"\bDATE\b|\bDATED\b")
OTHER_DATE_LABEL = _rx(r"\b(DUE|ORDER|DELIVERY|EXPIRY|EXP)\s*DATE\b")
COMPANY_SUFFIX = _rx(
    r"\b(SDN\.?\s*BHD|BHD|LTD|LIMITED|LLC|INC|PLC|GMBH|PTE|CO\.|CORP\w*|ENTERPRISE|TRADING|RESTAURANT|STORE|MART|CAFE)\b"
)
VENDOR_SKIP = _rx(r"TAX\s*INVOICE|RECEIPT|INVOICE|\bTEL\b|PHONE|FAX|GST|\bREG\b|WELCOME|@|WWW|HTTP")
SUFFIX_ONLY = _rx(r"\s*(SDN\.?\s*BHD\.?|BHD|S/B|LTD\.?|LIMITED|LLC|INC\.?|PLC|PTE\.?\s*LTD\.?|GMBH)\s*")
ITEM_LINE = _rx(r"^\s*\d+\s*(X|PCS|RB\.?|@)?\s+[A-Z]")  # "1 RB. COKLAT BAR", "2 x COFFEE"
ADDRESS_HINT = _rx(
    r"\b(JALAN|JLN|ROAD|RD|STREET|ST|TAMAN|LORONG|AVENUE|AVE|BLOCK|BLK|LOT|LEVEL|FLOOR|NO\.?\s*\d|\d{5,6})\b"
)
ADDRESS_STOP = _rx(r"\bTEL\b|PHONE|FAX|GST|EMAIL|INVOICE|RECEIPT|DATE|CASHIER")
ADDRESS_SKIP = _rx(r"\b(ROC|CO\.?\s*(REG\w*|NO)|COMPANY\s*(REG\w*|NO)|REG\w*\s*NO)\b|\(\d{3,}-[A-Z]\)")
DOC_TYPE_KEYWORDS: dict[str, list[re.Pattern[str]]] = {
    "delivery_note": [_rx(r"DELIVERY\s*(NOTE|ORDER)"), _rx(r"\bD/?O\s*NO\b"), _rx(r"RECEIVED\s*BY")],
    "invoice": [  # receipts and invoices are one type
        _rx(r"\bINVOICE\b"),
        _rx(r"BILL\s*TO"),
        _rx(r"DUE\s*DATE"),
        _rx(r"\bRECEIPT\b"),
        _rx(r"\bCASH\b"),
        _rx(r"\bCHANGE\b"),
        _rx(r"THANK\s*YOU"),
        _rx(r"\bTOTAL\b"),
    ],
    "form": [_rx(r"\bFORM\b"), _rx(r"SIGNATURE"), _rx(r"\bNAME\s*:")],
}


def _ev(line: OCRLine, label: OCRLine | None = None) -> Evidence:
    text = line.text if label is None or label is line else f"{label.text} | {line.text}"
    score = line.confidence if label is None else min(line.confidence, label.confidence)
    return Evidence(text=text, bbox=line.bbox, ocr_score=round(score, 4))


def _same_row(a: OCRLine, b: OCRLine) -> bool:
    if not a.bbox or not b.bbox:
        return False
    ca, cb = (a.bbox[1] + a.bbox[3]) / 2, (b.bbox[1] + b.bbox[3]) / 2
    h = max(a.bbox[3] - a.bbox[1], 1.0)
    return abs(ca - cb) <= h * 0.6 and b.bbox[0] >= a.bbox[0]


def _value_lines(lines: Sequence[OCRLine], i: int) -> list[tuple[OCRLine, float]]:
    """Lines that may hold the value for a label on line i, with position factor."""
    out: list[tuple[OCRLine, float]] = [(lines[i], 1.0)]
    out += [(ln, 0.95) for j, ln in enumerate(lines) if j != i and _same_row(lines[i], ln)]
    if i + 1 < len(lines) and not any(ln is lines[i + 1] for ln, _ in out):
        out.append((lines[i + 1], 0.85))
    return out


def _choose(cands: list[Candidate]) -> FieldResult:
    res = FieldResult(source="ocr")
    if not cands:
        res.warnings.append("not_found")
        return res
    best: dict[str, Candidate] = {}
    for c in cands:
        if c.value not in best or c.score > best[c.value].score:
            best[c.value] = c
    ranked = sorted(best.values(), key=lambda c: c.score, reverse=True)
    top = ranked[0]
    conf = top.score
    if len(ranked) > 1 and ranked[1].score >= AMBIGUITY_RATIO * top.score:
        conf *= 0.6
        res.warnings.append("ambiguous")
    res.value = top.value
    res.confidence = round(min(max(conf, 0.0), 1.0), 4)
    res.confidence_method = "ocr_heuristic"
    res.evidence = top.evidence
    res.candidates = ranked[1:5]
    if conf < LOW_OCR_CONFIDENCE:
        res.warnings.append("low_confidence")
    return res


def _money(lines: Sequence[OCRLine], field: str) -> FieldResult:
    cands: list[Candidate] = []
    for i, line in enumerate(lines):
        if MONEY_EXCLUDE[field].search(line.text):
            continue
        weight = next((w for rx, w in MONEY_LABELS[field] if rx.search(line.text)), None)
        if weight is None:
            continue
        for vline, pos in _value_lines(lines, i):
            if vline is not line and (
                MONEY_EXCLUDE[field].search(vline.text) or any(rx.search(vline.text) for rx in ANY_MONEY_LABEL)
            ):
                continue  # that line belongs to another label
            amounts = find_amounts(vline.text)
            if not amounts:
                continue
            score = min(line.confidence, vline.confidence) * weight * pos
            cands.append(Candidate(value=amounts[-1], score=score, evidence=_ev(vline, line)))  # rightmost
            break
    return _choose(cands)


def _tax_rate(lines: Sequence[OCRLine]) -> FieldResult:
    cands = []
    for line in lines:
        if MONEY_LABELS["tax"][0][0].search(line.text) and (m := PERCENT_RE.search(line.text)):
            cands.append(Candidate(value=m.group(0), score=line.confidence * 0.9, evidence=_ev(line)))
    return _choose(cands)


def _ids(lines: Sequence[OCRLine], field: str) -> FieldResult:
    cands = []
    for i, line in enumerate(lines):
        m = ID_LABELS[field].search(line.text)
        if not m:
            continue
        value, src, pos = m.group(1), line, 1.0
        if not value and i + 1 < len(lines) and re.fullmatch(r"[A-Z0-9][\w\-/]*", lines[i + 1].text.strip(), re.I):
            value, src, pos = lines[i + 1].text.strip(), lines[i + 1], 0.85
        if value and re.search(r"\d", value):
            score = min(line.confidence, src.confidence) * 0.9 * pos
            cands.append(Candidate(value=value, score=score, evidence=_ev(src, line)))
    return _choose(cands)


def _date(lines: Sequence[OCRLine]) -> FieldResult:
    cands = []
    for line in lines:
        weight = 0.95 if DATE_LABEL.search(line.text) else 0.8
        if OTHER_DATE_LABEL.search(line.text):
            weight = 0.4  # due/order/delivery date is not the document date
        for raw in find_dates(line.text):
            cands.append(Candidate(value=raw, score=line.confidence * weight, evidence=_ev(line)))
    return _choose(cands)


def _vendor_address(lines: Sequence[OCRLine]) -> tuple[FieldResult, FieldResult]:
    cands: list[tuple[int, Candidate]] = []
    for i, line in enumerate(lines[:8]):
        letters = sum(ch.isalpha() for ch in line.text)
        if (
            letters < 3
            or letters < len(line.text) * 0.5
            or VENDOR_SKIP.search(line.text)
            or ITEM_LINE.search(line.text)
            or find_dates(line.text)
        ):
            continue
        bonus = (0.3 if COMPANY_SUFFIX.search(line.text) else 0.0) + (0.1 if not cands else 0.0)
        score = min(line.confidence * (0.5 + bonus), 0.9)
        cands.append((i, Candidate(value=line.text, score=score, evidence=_ev(line))))
    for k, (i, c) in enumerate(cands):
        if SUFFIX_ONLY.fullmatch(c.value) and i > 0 and sum(ch.isalpha() for ch in lines[i - 1].text) >= 3:
            prev = lines[i - 1]  # name continues from the line above
            text = f"{prev.text} {c.value}"
            score = min(c.score, prev.confidence * 0.8)
            ev = Evidence(
                text=text,
                bbox=c.evidence.bbox if c.evidence else None,
                ocr_score=round(min(prev.confidence, lines[i].confidence), 4),
            )
            cands[k] = (i, Candidate(value=text, score=score, evidence=ev))
    vendor = _choose([c for _, c in cands])
    address = FieldResult(source="ocr", warnings=["not_found"])
    if vendor.value is None:
        return vendor, address
    start = next(i for i, c in cands if c.value == vendor.value) + 1
    parts: list[OCRLine] = []
    for line in lines[start : start + 6]:
        if ADDRESS_STOP.search(line.text):
            break
        if ADDRESS_SKIP.search(line.text):
            continue  # company registration number, not part of the address
        if ADDRESS_HINT.search(line.text) or (parts and len(parts) < 4 and not MONEY_RE.search(line.text)):
            parts.append(line)
        elif parts:
            break
    if parts:
        text = " ".join(p.text for p in parts)  # lines joined with a space, text unchanged
        boxes = [p.bbox for p in parts if p.bbox]
        bbox = (
            [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
            if boxes
            else None
        )
        score = min(p.confidence for p in parts) * 0.6
        evidence = Evidence(text=text, bbox=bbox, ocr_score=round(min(p.confidence for p in parts), 4))
        address = _choose([Candidate(value=text, score=score, evidence=evidence)])
    return vendor, address


def _currency(lines: Sequence[OCRLine]) -> FieldResult:
    cands = [
        Candidate(value=cur, score=line.confidence * 0.7, evidence=_ev(line))
        for line in lines
        for cur in find_currencies(line.text)
    ]
    return _choose(cands)


def classify_document(lines: Sequence[OCRLine]) -> FieldResult:
    """Keyword heuristic. Returns null when no keywords match."""
    text = "\n".join(line.text for line in lines)
    hits = {t: sum(bool(rx.search(text)) for rx in kws) for t, kws in DOC_TYPE_KEYWORDS.items()}
    ranked = sorted(hits.items(), key=lambda kv: kv[1], reverse=True)
    res = FieldResult(source="ocr")
    if ranked[0][1] == 0:
        res.warnings.append("not_found")
        return res
    top, n = ranked[0]
    conf = min(0.4 + 0.15 * n, 0.85)
    if ranked[1][1] == n:
        conf *= 0.6
        res.warnings.append("ambiguous")
        res.candidates = [Candidate(value=t, score=float(h)) for t, h in ranked[1:] if h == n]
    res.value, res.confidence, res.confidence_method = top, round(conf, 4), "ocr_heuristic"
    return res


def map_fields(ocr: OCRResult, *, document_type: str | None = None) -> ExtractionResult:
    """Convert an ``OCRResult`` into the shared ``ExtractionResult`` format."""
    lines = ocr.lines
    result = ExtractionResult(
        source="ocr",
        engine=ENGINE_NAME,
        fields=empty_fields("ocr"),
        raw_text=ocr.text or None,
        warnings=list(ocr.warnings),
    )
    if not lines:
        for f in result.fields.values():
            f.warnings.append("not_found")
        return result
    if any(line.confidence < LOW_OCR_CONFIDENCE for line in lines):
        result.warnings.append("low_ocr_confidence_lines_present")

    fields: dict[str, FieldResult] = {}
    if document_type and normalize_document_type(document_type):
        fields["document_type"] = FieldResult(
            source="ocr",
            value=normalize_document_type(document_type),
            confidence=1.0,
            confidence_method="caller_provided",
        )
    else:
        fields["document_type"] = classify_document(lines)
    fields["vendor"], fields["address"] = _vendor_address(lines)
    fields["invoice_number"] = _ids(lines, "invoice_number")
    fields["order_number"] = _ids(lines, "order_number")
    fields["date"] = _date(lines)
    fields["currency"] = _currency(lines)
    for money in ("subtotal", "tax", "rounding_adjustment", "total"):
        fields[money] = _money(lines, money)
    fields["tax_rate"] = _tax_rate(lines)

    for name, fr in fields.items():
        ev_text = fr.evidence.text if fr.evidence else None
        if ev_text and any(ln.confidence < LOW_OCR_CONFIDENCE and ln.text in ev_text for ln in lines):
            fr.warnings.append("low_ocr_confidence")
        result.fields[name] = fr
    return result
