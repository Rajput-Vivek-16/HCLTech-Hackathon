# Member 2 — OCR and VLM Extraction

Two independent branches read the **same image** and return the **same contract**:

| Branch | Module | Engine | Image leaves machine? |
|---|---|---|---|
| OCR | `extraction/ocr_engine.py` + `extraction/field_mapper.py` | PaddleOCR (local) | **No** |
| VLM | `extraction/ai_extractor.py` | Qwen2-VL-2B-Instruct, run locally with Hugging Face `transformers` | **No** |

Nothing here validates, compares, routes or stores results. That is Member 3 (rules/routing) and Member 4 (review/export).

## 1. Public interface

```python
from extraction import extract_with_ocr, extract_with_vlm, extract_both

ocr = extract_with_ocr(image, document_type=None)  # ExtractionResult
vlm = extract_with_vlm(image_or_pdf, document_type=None)
both = extract_both(image)  # {"ocr": ..., "vlm": ...}, run independently, no comparison
```

**Input (from Member 1):** a file path (`str`/`pathlib.Path`) or the file's bytes.

- **OCR branch:** PNG/JPEG, or a decoded numpy array.
- **VLM branch:** any image Pillow can open, or a PIL image. PDFs are not accepted; Member 1 converts them to images.

These functions do not raise on OCR or API failures; failures are reported in `result.errors`. If Member 1 already knows the type, pass `document_type` and it is recorded with `confidence_method="caller_provided"`.

## 2. Output format (`extraction/schemas.py`)

**Values are copied exactly as printed.** Nothing is reformatted:

- A date stays `30/03/2018`, `03-30-2018` or `2018-03-30`, as printed.
- Amounts keep their format (`1,234.50`, `1.234,56`).
- Currency is the symbol or code as printed (`RM`, `$`, `MYR`).
- Tax rate keeps the percent sign (`6%`).

Every value is a string, or `null` when it is absent. Converting values (ISO dates, `Decimal` amounts) is left to whoever consumes them (Member 3).

```jsonc
{
  "schema_version": "1.0",
  "source": "ocr",                 // "ocr" | "vlm"
  "engine": "paddleocr",           // or the model id
  "fields": {                      // ALWAYS has every field below
    "total": {
      "value": "189.75",           // as printed
      "confidence": 0.79,          // 0..1, see §3
      "confidence_method": "ocr_heuristic",
      "source": "ocr",
      "evidence": {"text": "TOTAL (MYR) | 189.75", "bbox": [10, 220, 200, 235]},
      "candidates": [],            // other possible values (e.g. a second TOTAL line)
      "warnings": []               // not_found, ambiguous, low_confidence, low_ocr_confidence, invalid_type...
    }
  },
  "extra_fields": {},              // fields with no slot below (kept, not dropped)
  "raw_text": "...",               // OCR only; in memory, the CLI hides it unless --include-raw-text
  "warnings": [], "errors": [{"code": "vlm_timeout", "message": "..."}]
}
```

`result.record()` returns the values only:

`document_type, vendor, address, invoice_number, order_number, date, currency, subtotal, tax_rate, tax, rounding_adjustment, total`

- A missing value is always `null`, with confidence `0.0`.
- `document_type` is a label we assign (`invoice | delivery_note | form`; receipts, bills and tax invoices are all `invoice`), not text copied from the document.
- Field-name aliases such as `company` or `merchant` → `vendor` are listed in `patterns.ALIASES`.

**Error codes:** `invalid_input`, `ocr_unavailable`, `ocr_failed`, `vlm_unavailable`, `vlm_failed`, `vlm_invalid_response`.

**Handing results to Member 3** (`rules_engine/`):

```python
from extraction import extract_both
from extraction.handoff import to_rules_engine_inputs
from rules_engine import process_extraction

results = extract_both(image)
decision = process_extraction(**to_rules_engine_inputs(results["ocr"], results["vlm"]))
```

`to_rules_engine_inputs` only renames fields to the rules engine's names (`vendor` → `vendor_name`, `tax` → `tax_amount`, `total` → `total_amount`). Values are passed exactly as printed; the rules engine converts dates and amounts itself. A failed branch is passed as `None`. VLM confidence isn't passed, because `process_extraction` only takes OCR confidence.

## 3. Confidence

| Method | Meaning |
|---|---|
| `ocr_heuristic` | PaddleOCR recognition score (the lower of the label line and value line) × label weight (e.g. `GRAND TOTAL` 1.0, `TOTAL` 0.85) × pairing factor (same line 1.0, same row by bbox 0.95, next line 0.85) × 0.6 if a different value scores ≥ 80% of the best. Vendor and address use fixed layout heuristics, capped at 0.9 and 0.6× respectively. |
| `token_probability` | VLM: geometric mean of the model's token probabilities for that value. Not calibrated. |
| `vlm_fallback_heuristic` | Token probabilities unavailable: 0.3. |
| `caller_provided` | The value came from the caller: 1.0. |
| `none` | No value: 0.0. |

All of these are ranking scores, not accuracy measurements. Check them against labelled samples before choosing router thresholds.

## 4. Field mapping (OCR), and how varied layouts are handled

`patterns.py` regexes only **find** values; `field_mapper.py` decides which value belongs to which field. For each field:

1. **Find a label:** for example `GRAND TOTAL`, `SUB TOTAL`, `GST`, `ROUNDING`, `DATE`, `INVOICE NO`.
2. **Take the value next to it:** the value can be on the same line, on the same row (using the OCR bounding boxes, so a label on the left and an amount on the right still match), or on the next line.
3. **Skip misleading lines:** `SUB TOTAL` doesn't count for total, and `TAX INVOICE` or `GST ID` don't count for tax. A `DUE DATE` or `DELIVERY DATE` scores below a plain `DATE`. The largest number is never assumed to be the total.
4. **Keep the alternatives:** every candidate goes into `candidates`, and close competitors set the `ambiguous` warning.

Vendor is the first text-heavy line near the top, with a bonus for company suffixes (`SDN BHD`, `LTD`, …). Address is the lines just below the vendor that contain address hints. Document type comes from keywords.

This handles common receipt and invoice layouts, but rules can't cover every layout. Unusual labels or table-style invoices will come back empty or `ambiguous`, which is why the VLM branch exists alongside it.

## 5. VLM: where the model is used, and which one

A VLM is used **only** in `extraction/ai_extractor.py`; the OCR branch uses no language model. The VLM is **Qwen2-VL-2B-Instruct** (Apache-2.0 licence), run **locally** with Hugging Face `transformers`. There is **no training or fine-tuning**: the model reads each image fresh with a fixed prompt, so no labels are needed, and it works on new, unlabelled documents.

| Setting | Value |
|---|---|
| Install | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`, then `pip install -e ".[vlm]"` (on a CUDA machine, install the matching CUDA torch instead) |
| Model download | About 4.5 GB, once, into `~/.cache/huggingface` (model files only) |
| Memory | About 5–6 GB RAM in bfloat16 on CPU |
| Speed | Slow on CPU (tens of seconds to minutes per image); much faster on a GPU. `VLM_MAX_PIXELS` trades detail for speed. |
| Output | The model is asked for one JSON object. The JSON is pulled out of its reply and every field is checked; non-JSON output becomes `vlm_invalid_response`. |
| Confidence | `token_probability`: the geometric mean of the model's own probabilities for the tokens that wrote that value. It measures how sure the model was, not how often it is right. If probabilities are unavailable, the value gets 0.3 (`vlm_fallback_heuristic`). |
| Env vars | `VLM_MODEL`, `VLM_DTYPE` (default `bfloat16`), `VLM_MAX_PIXELS`, `VLM_MAX_NEW_TOKENS`, `VLM_DEVICE` |
| Cost / privacy | Free, and **images never leave the machine**. |

Limitations: a 2B model can misread small or faint text and sometimes returns malformed JSON. Treat its output as one opinion for Member 3 to compare with OCR.

**Document type with no labels:** both branches classify every document themselves, OCR by keywords and the VLM as part of its answer, giving `invoice` (receipts, bills and tax invoices) `| delivery_note | form`, or `null` with a warning when unsure. A label from Member 1, when available, overrides both.

## 6. Data lifecycle

| # | Stage | In memory | Temp files | Persistent | External |
|---|---|---|---|---|---|
| 1 | Image from Member 1 | path / bytes / ndarray | none created by us | owned by Member 1 | no |
| 2 | Into PaddleOCR | path or ndarray (bytes are decoded in memory) | none | PaddleOCR model **weights** cached in `~/.paddlex` (no document data) | no; weights download once on first run |
| 3 | Raw OCR text, boxes, scores | `OCRResult` | none | no | no |
| 4 | OCR candidates | `FieldResult.candidates` | none | no | no |
| 5 | Image to VLM | PIL image / tensors | none | no (model weights only, in `~/.cache/huggingface`) | **no**, runs locally |
| 6 | VLM response | generated text → `ExtractionResult` | none | no | no |
| 7–8 | Normalized JSON, confidence, evidence | `ExtractionResult` | none | no | no |
| 9 | Handed to Member 3 | returned Python object | none | Member 3/4 decide | no |
| 10 | Logs / artifacts | — | none | logs hold **counts and error class names only**, never document text | no |

Everything can be discarded when the caller drops the result. CI never handles real documents or keys.

## 7. Setup and commands

| Step | Linux (bash) | Windows (PowerShell) |
|---|---|---|
| venv | `python3 -m venv .venv && source .venv/bin/activate` | `py -3.12 -m venv .venv; .\.venv\Scripts\Activate.ps1` |
| dev install | `pip install -e ".[dev]"` | same |
| OCR install (optional, ~1 GB) | `pip install -e ".[ocr]"` | same |
| VLM install (optional, ~5 GB) | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install -e ".[vlm]"` | same |
| settings | `cp .env.example .env` (optional) | `Copy-Item .env.example .env` |
| lint / format | `ruff check . && ruff format --check .` | `ruff check .; ruff format --check .` |
| types | `mypy` | same |
| tests + coverage | `pytest -m "not integration"` | same |
| real PaddleOCR smoke test | `pytest -m integration` | same |
| security | `bandit -c pyproject.toml -r extraction && pip-audit --skip-editable` | `bandit -c pyproject.toml -r extraction; pip-audit --skip-editable` |
| pre-commit | `pre-commit install && pre-commit run --all-files` | same |
| OCR branch | `python -m extraction data/samples/r1.jpg --branch ocr` | `python -m extraction data\samples\r1.jpg --branch ocr` |
| VLM branch | `python -m extraction data/samples/r1.jpg --branch vlm` | same |

Windows notes: if PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. PaddlePaddle publishes CPU wheels for Windows x64 on Python 3.9–3.13. If `pip install paddlepaddle` fails, use the official index (`pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/stable/cpu/`) or run the OCR branch on Linux/WSL2. The VLM branch and all unit tests work without Paddle.

## 8. CI

- `.github/workflows/ci.yml`: Ruff lint and format, mypy, pytest with coverage, pre-commit.
- `.github/workflows/security.yml`: pip-audit and Bandit, on push, on PR, and weekly.

Both workflows use read-only permissions, pip caching and SHA-pinned actions, and **do not install PaddleOCR or call the API**. There is no Dockerfile, so there is no Docker build job; add one if a Dockerfile is introduced. No coverage threshold is enforced.

## 9. Limitations

- The OCR heuristics are tuned for English/Malay receipts (SROIE style). Invoices with tables, multi-page PDFs (rasterising them is Member 1's job) and handwriting will need more label patterns.
- Vendor and address detection is positional; logos without text, or a header line missed by OCR, will produce the wrong vendor.
- Confidence scores are uncalibrated.
- The VLM can hallucinate, and a 2B model misreads small text more often than large models.
- Automated tests use fakes for PaddleOCR and Qwen2-VL; real models run only when you run them locally.
