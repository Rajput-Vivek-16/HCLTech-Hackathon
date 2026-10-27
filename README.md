# HCLTech-Hackathon
System Builders Team (AI Engineer)

## Case 3 — Messy Document Extraction with Human Review

This project prepares invoices, delivery notes, receipts, and forms for a
downstream multimodal OCR and extraction pipeline. The target workflow is to
extract structured fields with per-field confidence, validate totals and dates,
and send only uncertain fields to a human reviewer.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Ingestion and preprocessing contract

`preprocess.py` accepts PDF, PNG, JPG, and JPEG uploads and performs:

- extension, size, empty-file, and decode/corruption checks;
- SHA-256 hashing for exact duplicate detection;
- RGB normalization, long-edge downscaling, blur and low-resolution warnings;
- optional image deskewing with OpenCV;
- preservation of the original upload bytes.

PDFs are **not** converted to images by this module. PyMuPDF opens each PDF
only to validate it and read its page count. The exact original PDF bytes are
placed in `PreparedDocument.ocr_bytes` for the OCR module, so the OCR teammate
must support PDF input or perform conversion internally.

Images are passed to OCR as prepared PNG bytes. A downstream adapter can use:

```python
from preprocess import process_upload

document = process_upload(file_bytes, filename)
payload = document.ocr_payload()

# payload["content"]   -> original PDF bytes or prepared PNG bytes
# payload["filename"]  -> matching filename and extension
# payload["mime_type"] -> application/pdf or image/png
# payload["warnings"]  -> quality/preprocessing warnings
```

`document.original_bytes` is retained for audit and downstream fallback.
`document.sha256` identifies the exact uploaded file. Image previews and
per-page details are available in `document.pages`.

## Human Review Dashboard contract

The Streamlit app stores the first unique uploaded document once in session
state so the review dashboard does not need a second uploader:

```python
st.session_state["document_bytes"]  # original uploaded bytes
st.session_state["document_type"]   # image/png, image/jpeg, or application/pdf
st.session_state["document_id"]     # stable ID for this document
st.session_state["review_run_id"]   # new ID when preprocessing runs again
st.session_state["extraction_result"] = {
    "document_id": document_id,
    "fields": validated_fields,
}
```

After OCR and validation, publish results with:

```python
from app import publish_extraction_result

publish_extraction_result(document_id, validated_fields)
```

The review dashboard can reuse `document_bytes` for its document panel and
`extraction_result["fields"]` for editable review fields.

## Project structure

```text
├── app.py
├── preprocess.py
├── requirements.txt
├── sample_documents/
└── README.md
```

The Streamlit app displays metadata, quality warnings, original/prepared image
previews, and a direct-PDF download for OCR hand-off. Extraction results,
validation, confidence scoring, and the human-review queue remain integration
surfaces for the other team members.

## Member 2 — OCR & VLM extraction

`extraction/`: PaddleOCR and a local Qwen2-VL-2B, both producing one shared JSON output format. Everything runs locally.
See [docs/MEMBER2_EXTRACTION.md](docs/MEMBER2_EXTRACTION.md) for the interface, data lifecycle, privacy and setup.

Quick start: `python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]" && pytest`
