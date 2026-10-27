"""Regexes that FIND values in text. They never reformat what they find:
a date printed as 30/03/2018 is returned as "30/03/2018"."""

from __future__ import annotations

import re

# Raw dataset / model field names -> canonical field names (keys only, never values).
ALIASES: dict[str, str] = {
    "company": "vendor",
    "merchant": "vendor",
    "store": "vendor",
    "store_name": "vendor",
    "seller": "vendor",
    "supplier": "vendor",
    "vendor_name": "vendor",
    "invoice_no": "invoice_number",
    "invoice_id": "invoice_number",
    "inv_no": "invoice_number",
    "order_no": "order_number",
    "order_id": "order_number",
    "invoice_date": "date",
    "transaction_date": "date",
    "sub_total": "subtotal",
    "tax_amount": "tax",
    "gst": "tax",
    "rounding": "rounding_adjustment",
    "total_amount": "total",
    "grand_total": "total",
    "amount_due": "total",
    "doc_type": "document_type",
}

_MON = r"(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\.?"
DATE_RE = re.compile(
    r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b"  # 2018-03-30
    r"|\b\d{1,2}[-/.]\d{1,2}[-/.](?:\d{4}|\d{2})\b"  # 30/03/2018, 03-30-18
    rf"|\b\d{{1,2}}[\s\-/.]*{_MON}[\s\-/.,]*(?:\d{{4}}|\d{{2}})\b"  # 19 APR 2018
    rf"|\b{_MON}\s+\d{{1,2}},?\s+(?:\d{{4}}|\d{{2}})\b",  # April 19, 2018
    re.I,
)
# Amounts: 2 decimals (9.90 / 1,234.50 / 12,50 / -0.01) or thousands-grouped without
# decimals (24,500 / 1.250.000, common for IDR, JPY and similar currencies).
MONEY_RE = re.compile(r"(?<![\d.,])-?\s?(?:\d{1,3}(?:[,.]\d{3})+(?:[.,]\d{2})?|\d+[.,]\d{2})(?![\d])")
PERCENT_RE = re.compile(r"\d{1,2}(?:\.\d+)?\s?%")
CURRENCY_RE = re.compile(
    r"\b(?:MYR|USD|EUR|GBP|INR|SGD|IDR|AUD|JPY|CNY|THB|PHP|HKD|RM|RP)(?![A-Za-z])|US\$|S\$|[$€£₹¥]"
)


def canonical_key(name: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return ALIASES.get(key, key)


def find_dates(text: str) -> list[str]:
    return [m.group(0) for m in DATE_RE.finditer(text)]


def find_amounts(text: str) -> list[str]:
    return [m.group(0).strip() for m in MONEY_RE.finditer(text)]


def find_currencies(text: str) -> list[str]:
    """Currency codes/symbols exactly as printed, in order of first appearance."""
    return list(dict.fromkeys(m.group(0) for m in CURRENCY_RE.finditer(text)))
