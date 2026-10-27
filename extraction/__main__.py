"""CLI for one file, several files, or whole folders.

    python -m extraction receipt.jpg --branch ocr            # one file  -> pretty JSON
    python -m extraction data/samples --branch both          # a folder  -> one JSON line per file
    python -m extraction a.jpg b.png c.pdf --branch vlm > out.jsonl

Prints to stdout only; nothing is saved unless you redirect it (and then the output
contains document data, so keep it out of Git).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from . import extract_with_ocr, extract_with_vlm

SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".pdf"}


def iter_files(inputs: list[str]) -> Iterator[Path]:
    """Expand folders (recursively) into supported files, sorted for a stable order."""
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            yield from sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in SUFFIXES)
        else:
            yield p


def run_one(path: Path, branch: str, include_raw_text: bool = False) -> dict[str, Any]:
    """Run the chosen branch(es) on one file (image or PDF)."""
    out: dict[str, Any] = {"file": str(path)}
    branches = ["ocr", "vlm"] if branch == "both" else [branch]
    for b in branches:
        res = (extract_with_ocr if b == "ocr" else extract_with_vlm)(path)
        d = res.to_json_dict()
        if not include_raw_text:
            d.pop("raw_text", None)
        out[b] = d
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m extraction")
    ap.add_argument("inputs", nargs="+", help="image/PDF files and/or folders")
    ap.add_argument("--branch", choices=["ocr", "vlm", "both"], default="ocr")
    ap.add_argument("--include-raw-text", action="store_true", help="also print full OCR text")
    args = ap.parse_args(argv)
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    files = list(iter_files(args.inputs))
    if not files:
        print("no supported files found", file=sys.stderr)
        return 2
    failures = 0
    single = len(files) == 1
    for i, path in enumerate(files, 1):
        row = run_one(path, args.branch, args.include_raw_text)
        failures += any(row[b].get("errors") for b in row if b != "file")
        if single:
            json.dump(row, sys.stdout, indent=2, ensure_ascii=False)
            print()
        else:
            print(json.dumps(row, ensure_ascii=False), flush=True)  # JSON Lines
            print(f"[{i}/{len(files)}] {path.name}", file=sys.stderr)
    if not single:
        print(f"done: {len(files)} files, {failures} with errors", file=sys.stderr)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
