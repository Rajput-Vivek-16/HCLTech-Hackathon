"""
app.py - Streamlit frontend for Messy Document Extraction
=========================================================
This is the main application file for the Messy Document Extraction tool.

Features
--------
* Upload PDF / PNG / JPG documents with validation (size, type, corruption,
  password-protected PDFs, page cap); rejected files are listed clearly
* Exact-duplicate detection via SHA-256
* Image cleanup: orientation (auto via Tesseract OSD, or manual), noise
  reduction, deskew, resize - each toggleable, each reported
* Before / after previews, quality metrics, PDF page previews
* Hand-off to the OCR module through ``PreparedDocument.ocr_payload()``,
  published in ``st.session_state["ocr_payloads"]``
* Review contract for the dashboard: ``document_bytes``, ``document_type``,
  ``document_id``, ``review_run_id``, ``extraction_result``
"""

from __future__ import annotations

import html
import os
import time
import uuid

import pandas as pd
import streamlit as st

from preprocess import (
    ALLOWED_EXTENSIONS,
    DEFAULT_MAX_SIZE_MB,
    MAX_PDF_PAGES,
    PreparedDocument,
    process_upload,
    render_pdf_preview,
)


def publish_extraction_result(
    document_id: str,
    validated_fields: dict,
) -> None:
    """Publish validated OCR output for the human-review dashboard."""
    st.session_state["extraction_result"] = {
        "document_id": document_id,
        "fields": validated_fields,
    }


def stretch(fn, *args, **kwargs):
    """Call a Streamlit widget at full width on old and new Streamlit versions."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:
        return fn(*args, width="stretch", **kwargs)


# ---------------------------------------------------------------------------
# Page config and styling (uses Streamlit theme variables: light + dark mode)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DocExtract - Messy Document Extraction",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; max-width: 1320px; }
    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border-radius: 12px; padding: 14px 18px;
        border: 1px solid rgba(128,128,128,0.18);
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    div[data-testid="stMetricValue"] { font-weight: 700; }
    .hero {
        padding: 22px 28px; border-radius: 16px; margin-bottom: 14px; color: #fff;
        background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 45%, #db2777 100%);
        box-shadow: 0 6px 24px rgba(79,70,229,0.25);
    }
    .hero h1 { margin: 0; color: #fff; font-size: 2.1rem; }
    .hero p { opacity: 0.92; margin: 6px 0 0 0; font-size: 1.02rem; }
    .hero .badges { margin-top: 12px; }
    .hero .badge { display: inline-block; padding: 3px 11px; margin-right: 6px; border-radius: 999px;
                   background: rgba(255,255,255,0.18); font-size: 0.8rem; }
    .section-title { font-size: 1.35rem; font-weight: 700; margin: 6px 0 10px 0; }
    .decision { padding: 16px 20px; border-radius: 14px; margin: 8px 0 14px 0;
                border-left: 6px solid; background: var(--secondary-background-color); }
    .decision h3 { margin: 0 0 6px 0; }
    .decision ul { margin: 4px 0 0 18px; padding: 0; font-size: 0.92rem; opacity: 0.9; }
    .decision.ok   { border-color: #16a34a; }
    .decision.warn { border-color: #f59e0b; }
    .decision.bad  { border-color: #dc2626; }
    .legend span { display: inline-block; margin-right: 14px; font-size: 0.85rem; }
    .doctype { padding: 14px 20px; border-radius: 14px; margin: 6px 0 4px 0;
               background: linear-gradient(90deg, rgba(79,70,229,0.12), rgba(219,39,119,0.08));
               border: 1px solid rgba(79,70,229,0.35); }
    .doctype-main { font-size: 1.45rem; font-weight: 700; margin-right: 12px; }
    .doctype-note { font-size: 0.9rem; opacity: 0.8; }
    .doctype-detail { font-size: 0.85rem; opacity: 0.75; margin-top: 4px; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 6px 14px; }

    .steps { display: flex; gap: 8px; margin: 10px 0 18px 0; flex-wrap: wrap; }
    .step { flex: 1; min-width: 150px; padding: 10px 14px; border-radius: 10px;
            background: var(--secondary-background-color);
            border: 1px solid rgba(128,128,128,0.25); font-size: 0.9rem; }
    .step b { display: block; font-size: 0.78rem; opacity: 0.65;
              text-transform: uppercase; letter-spacing: 0.04em; }
    .step.done { border-color: rgba(40,167,69,0.7); background: rgba(40,167,69,0.10); }
    .step.active { border-color: var(--primary-color); }

    .chip { display: inline-block; padding: 3px 10px; border-radius: 999px;
            font-size: 0.8rem; margin: 0 6px 6px 0;
            border: 1px solid rgba(128,128,128,0.35); }
    .chip.ok   { background: rgba(40,167,69,0.14);  border-color: rgba(40,167,69,0.6); }
    .chip.warn { background: rgba(255,193,7,0.18);  border-color: rgba(255,193,7,0.7); }
    .chip.dup  { background: rgba(220,53,69,0.12);  border-color: rgba(220,53,69,0.6); }
    .chip.info { background: rgba(23,162,184,0.12); border-color: rgba(23,162,184,0.6); }
    .note { font-size: 0.88rem; opacity: 0.85; margin: 2px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


def chip(text: str, kind: str = "info") -> str:
    return f'<span class="chip {kind}">{html.escape(text)}</span>'


# ---------------------------------------------------------------------------
# Sidebar - grouped settings
# ---------------------------------------------------------------------------
ROTATION_CHOICES = {"Off (auto-detect)": 0, "90° clockwise": 90, "180°": 180, "270° clockwise": 270}

with st.sidebar:
    st.header("⚙️ Settings")

    with st.expander("📁 File limits", expanded=False):
        max_size = st.number_input(
            "Max file size (MB)",
            min_value=1,
            max_value=100,
            value=DEFAULT_MAX_SIZE_MB,
            step=1,
        )
        st.caption(f"Types: {', '.join(sorted(ALLOWED_EXTENSIONS))} · PDFs up to {MAX_PDF_PAGES} pages")

    with st.expander("🖼️ Image cleanup", expanded=True):
        long_edge = st.slider(
            "Resize long edge (px)",
            min_value=640,
            max_value=3840,
            value=1920,
            step=64,
        )
        auto_orient = st.checkbox(
            "Auto-fix page orientation",
            value=True,
            help="Uses Tesseract OSD to fix 90/180/270° rotation. If it is unavailable or unsure, nothing is rotated.",
        )
        rotation_label = st.selectbox(
            "Manual rotation override",
            list(ROTATION_CHOICES),
            help="Use when a page is still sideways or upside down. Overrides auto-orientation.",
        )
        manual_rotation = ROTATION_CHOICES[rotation_label]
        do_denoise = st.checkbox(
            "Reduce noise",
            value=True,
            help="Only applied when noise is detected, so clean scans are left untouched.",
        )
        do_deskew = st.checkbox("Straighten slight tilt (deskew)", value=True)

    with st.expander("🤖 Extraction", expanded=True):
        use_ocr = st.checkbox("PaddleOCR (text + layout rules)", value=True)
        use_vlm = st.checkbox(
            "Qwen2-VL-2B (local vision model)",
            value=True,
            help="Runs fully on this machine. On CPU it takes about 1-2 minutes per document.",
        )
        doc_type_label = st.selectbox(
            "Document type",
            ["Auto-detect", "Receipt / Invoice", "Delivery note", "Form"],
            help="Auto-detect lets both branches classify the document.",
        )
        doc_type_override = {"Receipt / Invoice": "invoice", "Delivery note": "delivery_note", "Form": "form"}.get(
            doc_type_label
        )
        ocr_speed = st.radio(
            "OCR model",
            ["Accurate", "Fast"],
            horizontal=True,
            help="Fast uses PaddleOCR mobile models: about 2x quicker, slightly less accurate.",
        )
        os.environ["OCR_MODEL_SIZE"] = "mobile" if ocr_speed == "Fast" else "server"
        auto_run = st.toggle("Run automatically on upload", value=True)

    with st.expander("🎨 Display", expanded=False):
        show_boxes = st.checkbox("Draw OCR evidence boxes on the image", value=True)
        min_conf = st.slider("Flag fields below confidence", 0.0, 1.0, 0.80, 0.05)

    st.divider()
    if st.button("🗑️ Clear processed cache"):
        for key in ("processed_cache", "review_runs", "timings", "pipeline_runs", "approved_records"):
            st.session_state[key] = {}
        st.cache_data.clear()
        st.rerun()

# ---------------------------------------------------------------------------
# Header and upload
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>📄 DocExtract</h1>
      <p>Messy documents in, structured fields out. Only the uncertain fields go to a human.</p>
      <div class="badges">
        <span class="badge">🧹 Preprocessing</span><span class="badge">🔤 PaddleOCR</span>
        <span class="badge">👁️ Qwen2-VL (local)</span><span class="badge">✅ Rules engine</span>
        <span class="badge">🔒 Nothing leaves this machine</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.session_state.setdefault("uploader_key", 0)
col_upload, col_clear = st.columns([6, 1], vertical_alignment="center")
with col_upload:
    uploaded_files = st.file_uploader(
        "Upload document(s)",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Supported: PDF, PNG, JPG/JPEG. Scans, photos and PDFs all work.",
        key=f"uploader_{st.session_state.uploader_key}",
    )
with col_clear:
    if st.button("🧹 Clear all", width="stretch", help="Remove all uploads, results and review state"):
        st.session_state.uploader_key += 1  # a new key gives an empty uploader
        for key in (
            "processed_cache",
            "review_runs",
            "timings",
            "pipeline_runs",
            "approved_records",
            "document_ids",
        ):
            st.session_state[key] = {}
        for key in (
            "extraction_result",
            "document_bytes",
            "document_type",
            "document_id",
            "review_run_id",
            "ocr_payloads",
            "_docsense_review_states",
        ):
            st.session_state.pop(key, None)
        st.cache_data.clear()
        st.rerun()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key in ("processed_cache", "document_ids", "review_runs", "timings"):
    st.session_state.setdefault(key, {})

extraction_result = st.session_state.get("extraction_result") or {"fields": {}}
has_fields = bool(extraction_result.get("fields"))


if not uploaded_files:
    st.info("👆 Upload a document to get started.")
    st.stop()

# ---------------------------------------------------------------------------
# Process uploads (cached per file AND per setting, so toggles take effect)
# ---------------------------------------------------------------------------
results: list[tuple[str, PreparedDocument, str]] = []  # (label, document, cache_key)
rejected: list[tuple[str, str]] = []

for idx, uf in enumerate(uploaded_files):
    label = f"{idx + 1}. {uf.name}"
    cache_key = (
        f"{uf.name}_{uf.size}_{getattr(uf, 'file_id', '')}_{max_size}_{long_edge}_"
        f"{do_deskew}_{do_denoise}_{auto_orient}_{manual_rotation}"
    )

    if cache_key in st.session_state.processed_cache:
        results.append((label, st.session_state.processed_cache[cache_key], cache_key))
        continue

    try:
        t0 = time.perf_counter()
        prepared = process_upload(
            uf.getvalue(),
            uf.name,
            max_size_mb=max_size,
            long_edge=long_edge,
            deskew=do_deskew,
            denoise=do_denoise,
            auto_orient=auto_orient,
            manual_rotation=manual_rotation,
        )
        elapsed = time.perf_counter() - t0
        document_id = st.session_state.document_ids.setdefault(prepared.sha256, str(uuid.uuid4()))
        st.session_state.review_runs[cache_key] = {
            "document_id": document_id,
            "review_run_id": str(uuid.uuid4()),
        }
        st.session_state.timings[cache_key] = elapsed
        st.session_state.processed_cache[cache_key] = prepared
        results.append((label, prepared, cache_key))
        st.toast(f"{uf.name} processed in {elapsed:.2f}s", icon="✅")
    except ValueError as exc:  # expected rejections: corrupt, protected, too big...
        rejected.append((uf.name, str(exc)))
    except Exception as exc:  # unexpected: never crash the whole page
        rejected.append((uf.name, f"Unexpected error while processing ({exc})."))


if rejected:
    with st.expander(f"⛔ {len(rejected)} file(s) rejected", expanded=True):
        for name, message in rejected:
            st.error(f"**{name}**: {message}")

if not results:
    st.warning("No valid documents to display.")
    st.stop()

# ---------------------------------------------------------------------------
# Exact-duplicate detection (first occurrence wins)
# ---------------------------------------------------------------------------
first_seen: dict[str, str] = {}
duplicates: dict[str, str] = {}  # label -> label of first copy
for label, prepared, _ in results:
    original_label = first_seen.setdefault(prepared.sha256, label)
    if original_label != label:
        duplicates[label] = original_label

# Publish the hand-off for the OCR module (skip exact duplicates).
st.session_state["ocr_payloads"] = [prepared.ocr_payload() for label, prepared, _ in results if label not in duplicates]

# Publish the first unique upload as the single-document review contract.
# The dashboard reuses these bytes; it must not ask the user to upload again.
primary = next(((lbl, doc, k) for lbl, doc, k in results if lbl not in duplicates), None)
if primary is not None:
    _, primary_document, primary_key = primary
    primary_run = st.session_state.review_runs[primary_key]
    st.session_state["document_bytes"] = primary_document.original_bytes
    st.session_state["document_type"] = (
        "application/pdf"
        if primary_document.is_pdf
        else ("image/jpeg" if primary_document.extension in {".jpg", ".jpeg"} else "image/png")
    )
    st.session_state["document_id"] = primary_run["document_id"]
    st.session_state["review_run_id"] = primary_run["review_run_id"]
    existing = st.session_state.get("extraction_result")
    if not isinstance(existing, dict) or existing.get("document_id") != primary_run["document_id"]:
        publish_extraction_result(primary_run["document_id"], {})
        extraction_result = st.session_state["extraction_result"]
        has_fields = bool(extraction_result["fields"])


# ---------------------------------------------------------------------------
# PDF preview helper (cached; display only, OCR still gets the original bytes)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Rendering PDF preview...")
def _pdf_preview_images(pdf_bytes: bytes):
    return render_pdf_preview(pdf_bytes, dpi=110)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
n_unique = len(results) - len(duplicates)
n_warn = sum(1 for _, p, _ in results if p.warnings)
total_pages = sum(p.page_count for _, p, _ in results)

summary = st.columns(4)
summary[0].metric("Documents accepted", len(results))
summary[1].metric("Unique (sent to OCR)", n_unique)
summary[2].metric("With quality warnings", n_warn)
summary[3].metric("Total pages", total_pages)

# ---------------------------------------------------------------------------
# Per-document cards
# ---------------------------------------------------------------------------
st.subheader("🖼️ Documents")

for position, (label, prepared, cache_key) in enumerate(results):
    is_dup = label in duplicates
    icon = "🔁" if is_dup else ("⚠️" if prepared.warnings else "✅")
    with st.expander(f"{icon} {label}  -  {prepared.page_count} page(s)", expanded=(position == 0)):
        # Status chips
        chips = []
        if is_dup:
            chips.append(chip(f"Duplicate of {duplicates[label]}", "dup"))
        elif prepared.warnings:
            chips.append(chip(f"{len(prepared.warnings)} quality warning(s)", "warn"))
        else:
            chips.append(chip("No quality issues", "ok"))
        chips.append(chip(prepared.extension.upper().lstrip("."), "info"))
        if not prepared.is_pdf:
            pg = prepared.pages[0]
            if pg.rotation_applied:
                chips.append(chip(f"Rotated {pg.rotation_applied}°", "info"))
            if pg.denoised:
                chips.append(chip("Noise reduced", "info"))
            if pg.deskew_angle:
                chips.append(chip(f"Deskewed {pg.deskew_angle:.1f}°", "info"))
        st.markdown("".join(chips), unsafe_allow_html=True)
        if is_dup:
            st.caption("Excluded from the OCR hand-off.")

        tab_preview, tab_quality, tab_handoff = st.tabs(["Preview", "Quality report", "OCR hand-off"])

        # ---- Preview ----------------------------------------------------
        with tab_preview:
            if prepared.is_pdf:
                try:
                    pages = _pdf_preview_images(prepared.original_bytes)
                    if len(pages) == 1:
                        stretch(st.image, pages[0])
                    else:
                        for tab, img in zip(
                            st.tabs([f"Page {n}" for n in range(1, len(pages) + 1)]),
                            pages,
                            strict=True,
                        ):
                            with tab:
                                stretch(st.image, img)
                except Exception as exc:
                    st.warning(f"Could not render a preview ({exc}). The PDF is still valid.")
                st.caption("Preview only. The original PDF bytes are sent unchanged to OCR.")
            else:
                pg = prepared.pages[0]
                col_orig, col_proc = st.columns(2)
                with col_orig:
                    st.markdown("**Original**")
                    stretch(st.image, pg.original_image)
                with col_proc:
                    st.markdown("**Processed - what OCR receives**")
                    stretch(st.image, pg.processed_image)

        # ---- Quality ----------------------------------------------------
        with tab_quality:
            if prepared.warnings:
                st.markdown(
                    "".join(chip(w, "warn") for w in prepared.warnings),
                    unsafe_allow_html=True,
                )
            for note in prepared.notes:
                st.markdown(f'<div class="note">• {html.escape(note)}</div>', unsafe_allow_html=True)
            if not (prepared.warnings or prepared.notes):
                st.caption("Nothing to report.")

            cols = st.columns(4)
            cols[0].metric("Size", f"{prepared.file_size / (1024 * 1024):.2f} MB")
            cols[1].metric("Processing time", f"{st.session_state.timings.get(cache_key, 0):.2f} s")
            if prepared.is_pdf:
                cols[2].metric("Pages", prepared.page_count)
                cols[3].metric("Deskew / noise", "n/a (PDF)")
                if prepared.metadata:
                    with st.expander("PDF metadata"):
                        st.json(prepared.metadata)
            else:
                pg = prepared.pages[0]
                cols[2].metric("Sharpness", f"{pg.sharpness:.0f}")
                cols[3].metric("Noise level", f"{pg.noise_sigma:.1f}")
                cols2 = st.columns(4)
                cols2[0].metric("Resolution", f"{pg.width}×{pg.height}")
                cols2[1].metric("Rotation applied", f"{pg.rotation_applied}°")
                cols2[2].metric("Orientation check", pg.orientation_method)
                cols2[3].metric("Deskew applied", f"{pg.deskew_angle:.1f}°" if pg.deskew_angle else "none")

        # ---- Hand-off ---------------------------------------------------
        with tab_handoff:
            payload = prepared.ocr_payload()
            st.json({k: v for k, v in payload.items() if k != "content"} | {"content_bytes": len(payload["content"])})
            if prepared.is_pdf:
                st.download_button(
                    "⬇️ Download original PDF (as sent to OCR)",
                    data=prepared.ocr_bytes,
                    file_name=prepared.ocr_filename,
                    mime=prepared.mime_type,
                    key=f"pdf_{position}",
                )
            else:
                st.download_button(
                    "⬇️ Download processed image (as sent to OCR)",
                    data=prepared.pages[0].processed_bytes("PNG"),
                    file_name=prepared.ocr_filename,
                    mime="image/png",
                    key=f"dl_{position}",
                )

# ---------------------------------------------------------------------------
# Extraction (Member 2) + validation and routing (Member 3)
# ---------------------------------------------------------------------------
import json  # noqa: E402

from PIL import ImageDraw  # noqa: E402

from extraction.inputs import load_document  # noqa: E402
from pipeline import export_record, published_fields, review_rows, run_pipeline  # noqa: E402
from review import render_review  # noqa: E402

DECISION_STYLE = {
    "auto_accepted": ("ok", "✅ Auto-accepted", "All checks passed with high confidence."),
    "human_review": ("warn", "👀 Needs human review", "Some fields are uncertain. Check them below."),
    "retry_extraction": ("bad", "🔁 Retry extraction", "Extraction looks unreliable; try again or fix the image."),
    "rejected": ("bad", "⛔ Rejected", "The document failed critical checks."),
}
FIELD_ICON = {"auto_accept": "✅", "human_review": "👀", "retry_extraction": "🔁", "reject": "⛔", "info": "ℹ️"}


@st.cache_resource(show_spinner="Loading models (first time only)...")
def _warm_models(ocr_on: bool, vlm_on: bool, ocr_size: str) -> bool:
    """Load PaddleOCR / Qwen2-VL once per server process so each document only pays for inference."""
    from extraction import ai_extractor, ocr_engine

    if ocr_on:
        ocr_engine.get_engine()
    if vlm_on:
        ai_extractor.get_engine()
    return True


st.session_state.setdefault("pipeline_runs", {})
st.session_state.setdefault("approved_records", {})

st.divider()
st.markdown('<div class="section-title">🧠 Extraction & validation</div>', unsafe_allow_html=True)

unique_docs = [(label, prepared, key) for label, prepared, key in results if label not in duplicates]
options_key = f"ocr={use_ocr}:{ocr_speed}|vlm={use_vlm}|type={doc_type_override}"
branches_on = use_ocr or use_vlm

col_sel, col_run, col_all = st.columns([4, 1.3, 1.3])
selected_label = col_sel.selectbox("Document", [label for label, _, _ in unique_docs], label_visibility="collapsed")
sel_label, sel_doc, sel_key = next(d for d in unique_docs if d[0] == selected_label)
run_key = f"{sel_key}|{options_key}"
clicked_one = col_run.button("▶️ Run extraction", type="primary", disabled=not branches_on, width="stretch")
clicked_all = col_all.button(
    f"⏩ Run all ({len(unique_docs)})", disabled=not branches_on or len(unique_docs) < 2, width="stretch"
)
if not branches_on:
    st.warning("Turn on at least one extraction branch in the sidebar.")


def _run(label: str, prepared: PreparedDocument, key: str) -> None:
    steps = " → ".join(s for s, on in (("PaddleOCR", use_ocr), ("Qwen2-VL", use_vlm), ("rules engine", True)) if on)
    with st.status(f"Extracting **{label}**: {steps}", expanded=False) as status:
        if use_vlm:
            st.write("Qwen2-VL runs locally; on CPU this takes about 1-2 minutes.")
        run = run_pipeline(prepared.ocr_payload(), use_ocr=use_ocr, use_vlm=use_vlm, document_type=doc_type_override)
        st.session_state.pipeline_runs[f"{key}|{options_key}"] = run
        status.update(label=f"{label}: {run.decision.replace('_', ' ')}", state="complete")


if branches_on and (clicked_one or clicked_all or auto_run):
    try:
        _warm_models(use_ocr, use_vlm, os.environ["OCR_MODEL_SIZE"])
    except Exception as exc:  # missing optional model packages: branch reports its own error
        st.warning(f"Model preload failed ({type(exc).__name__}); the branch will report the error.")
if clicked_one or (auto_run and branches_on and run_key not in st.session_state.pipeline_runs):
    _run(sel_label, sel_doc, sel_key)
if clicked_all:
    for label, prepared, key in unique_docs:
        if f"{key}|{options_key}" not in st.session_state.pipeline_runs:
            _run(label, prepared, key)

run = st.session_state.pipeline_runs.get(run_key)
document_id = st.session_state.review_runs[sel_key]["document_id"]

if run is None:
    st.info("Choose the branches in the sidebar, then press **▶️ Run extraction**.")
else:
    rows = review_rows(run)
    publish_extraction_result(document_id, published_fields(run))
    rules = run.rules.to_dict()

    # ---- Decision banner ---------------------------------------------------
    # ---- Document type (classifier) -----------------------------------------
    DOC_ICON = {"invoice": "🧾", "delivery_note": "🚚", "form": "📝"}
    DOC_LABEL = {"invoice": "Receipt / Invoice", "delivery_note": "Delivery Note", "form": "Form"}

    def _type_of(branch):
        if branch is None or not branch.ok:
            return None, None
        f = branch.fields["document_type"]
        return f.value, (f.confidence if f.value else None)

    ocr_t, ocr_tc = _type_of(run.ocr)
    vlm_t, vlm_tc = _type_of(run.vlm)
    final_t = vlm_t or ocr_t  # same precedence as the hand-off to the rules engine
    if doc_type_override:
        source_note = "set manually in the sidebar"
    elif ocr_t and vlm_t:
        source_note = "OCR and VLM agree" if ocr_t == vlm_t else f"VLM says {vlm_t}, OCR says {ocr_t}"
    elif final_t:
        source_note = "from " + ("VLM" if vlm_t else "OCR") + " only"
    else:
        source_note = "neither branch could tell"
    parts = [
        f"OCR: <b>{html.escape(ocr_t or '—')}</b>" + (f" ({ocr_tc:.2f})" if ocr_tc else ""),
        f"VLM: <b>{html.escape(vlm_t or '—')}</b>" + (f" ({vlm_tc:.2f})" if vlm_tc else ""),
        f"rules engine checks it as <b>{html.escape(str(rules.get('document_type', 'generic')))}</b>",
    ]
    st.markdown(
        f'<div class="doctype"><span class="doctype-main">{DOC_ICON.get(final_t or "", "❓")} '
        f"{html.escape(DOC_LABEL.get(final_t or '', 'Unknown type'))}</span>"
        f'<span class="doctype-note">{html.escape(source_note)}</span>'
        f'<div class="doctype-detail">{" · ".join(parts)}</div></div>',
        unsafe_allow_html=True,
    )

    css, title, blurb = DECISION_STYLE.get(run.decision, ("warn", run.decision, ""))
    reasons = "".join(f"<li>{html.escape(str(r))}</li>" for r in rules.get("decision_reasons", [])[:4])
    st.markdown(
        f'<div class="decision {css}"><h3>{title}</h3><div>{blurb}</div><ul>{reasons}</ul></div>',
        unsafe_allow_html=True,
    )

    # ---- KPIs ----------------------------------------------------------------
    found = [r for r in rows if r["value"] is not None]
    review_needed = [r for r in rows if r["decision"] not in ("auto_accept", "info")]
    comparable = [r for r in rows if r["agreement"] in ("match", "mismatch")]
    agree = sum(r["agreement"] == "match" for r in comparable)
    confs = [r["confidence"] for r in found if isinstance(r["confidence"], int | float)]
    k = st.columns(5)
    k[0].metric("Fields found", f"{len(found)}/{len(rows)}")
    k[1].metric("Need review", len(review_needed))
    k[2].metric("OCR ↔ VLM agreement", f"{agree}/{len(comparable)}" if comparable else "n/a")
    k[3].metric("Avg confidence", f"{sum(confs) / len(confs):.2f}" if confs else "n/a")
    k[4].metric("Time", " + ".join(f"{name} {secs:.0f}s" for name, secs in run.timings.items() if secs >= 0.5) or "<1s")

    for branch in (run.ocr, run.vlm):
        if branch is not None and branch.errors:
            st.error(
                f"**{branch.source.upper()}** failed: " + "; ".join(f"{e.code} ({e.message})" for e in branch.errors)
            )

    tab_fields, tab_compare, tab_evidence, tab_checks, tab_json = st.tabs(
        ["📋 Fields", "⚖️ OCR vs VLM", "🖼️ Evidence", "✅ Checks", "🧾 Raw JSON"]
    )

    # ---- Fields --------------------------------------------------------------
    with tab_fields:
        df = pd.DataFrame(
            [
                {
                    "": FIELD_ICON.get(r["decision"], "•"),
                    "Field": r["field"],
                    "Value": "" if r["value"] is None else str(r["value"]),
                    "Confidence": r["confidence"],
                    "Validation": r["validation"],
                    "Agreement": r["agreement"],
                    "Decision": r["decision"].replace("_", " "),
                    "Why": r["reasons"],
                }
                for r in rows
            ]
        )
        stretch(
            st.dataframe,
            df,
            hide_index=True,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0.0, max_value=1.0, format="%.2f"
                ),
                "Why": st.column_config.TextColumn("Why", width="large"),
            },
        )
        low = [r["field"] for r in found if isinstance(r["confidence"], int | float) and r["confidence"] < min_conf]
        if low:
            st.caption(f"⚠️ Below {min_conf:.2f} confidence: {', '.join(low)}")

    # ---- OCR vs VLM ----------------------------------------------------------
    with tab_compare:
        st.caption("Values exactly as printed on the document, with each branch's own confidence.")
        cmp = pd.DataFrame(
            [
                {
                    "Field": r["field"],
                    "PaddleOCR": r["ocr_value"] or "",
                    "OCR conf": r["ocr_conf"],
                    "Qwen2-VL": r["vlm_value"] or "",
                    "VLM conf": r["vlm_conf"],
                    "Agreement": {"match": "🟢 match", "mismatch": "🔴 mismatch"}.get(
                        r["agreement"], "⚪ " + r["agreement"]
                    ),
                }
                for r in rows
            ]
        )
        stretch(
            st.dataframe,
            cmp,
            hide_index=True,
            column_config={
                "OCR conf": st.column_config.ProgressColumn("OCR conf", min_value=0.0, max_value=1.0, format="%.2f"),
                "VLM conf": st.column_config.ProgressColumn("VLM conf", min_value=0.0, max_value=1.0, format="%.2f"),
            },
        )
        st.caption(
            "OCR confidence = recognition score × label × position (heuristic). "
            "VLM confidence = the model's token probabilities for that value. Neither is a calibrated accuracy."
        )

    # ---- Evidence ------------------------------------------------------------
    with tab_evidence:
        col_img, col_txt = st.columns([3, 2])
        with col_img:
            try:
                page_img = load_document(sel_doc.ocr_payload()).pages[0]
            except Exception as exc:
                page_img = None
                st.warning(f"Could not render the page ({exc}).")
            if page_img is not None:
                if show_boxes and run.ocr is not None and run.ocr.ok:
                    page_img = page_img.convert("RGB").copy()
                    draw = ImageDraw.Draw(page_img)
                    for name, f in run.ocr.fields.items():
                        if f.value is None or not f.evidence or not f.evidence.bbox:
                            continue
                        x0, y0, x1, y1 = f.evidence.bbox
                        if y0 > page_img.height:
                            continue  # box is on a later PDF page
                        color = (22, 163, 74) if f.confidence >= min_conf else (245, 158, 11)
                        draw.rectangle([x0 - 2, y0 - 2, x1 + 2, y1 + 2], outline=color, width=3)
                        draw.text((x0, max(0, y0 - 13)), name, fill=color)
                stretch(st.image, page_img)
                st.markdown(
                    '<div class="legend"><span>🟩 confident</span><span>🟧 below threshold</span></div>',
                    unsafe_allow_html=True,
                )
        with col_txt:
            st.markdown("**Warnings**")
            warns = [w for b in (run.ocr, run.vlm) if b is not None for w in b.warnings]
            if warns:
                st.markdown("".join(chip(w, "warn") for w in dict.fromkeys(warns)), unsafe_allow_html=True)
            else:
                st.caption("None")
            if run.ocr is not None and run.ocr.raw_text:
                with st.expander("Full OCR text", expanded=False):
                    st.code(run.ocr.raw_text, language=None)

    # ---- Checks --------------------------------------------------------------
    with tab_checks:
        checks = rules.get("cross_field_checks") or []
        if checks:
            stretch(
                st.dataframe,
                pd.DataFrame(
                    [
                        {
                            "Check": c.get("check_name"),
                            "Status": str(c.get("status")),
                            "Expected": c.get("expected_value"),
                            "Actual": c.get("actual_value"),
                            "Difference": c.get("difference"),
                            "Reason": "; ".join(c.get("reasons") or []),
                        }
                        for c in checks
                    ]
                ),
                hide_index=True,
            )
        else:
            st.caption("No cross-field checks were run.")
        crit = rules.get("critical_failures") or []
        if crit:
            st.error("Critical: " + ", ".join(map(str, crit)))
        else:
            st.success("No critical failures.")

    # ---- Raw JSON ------------------------------------------------------------
    with tab_json:
        raw = {
            "ocr": run.ocr.to_json_dict() if run.ocr else None,
            "vlm": run.vlm.to_json_dict() if run.vlm else None,
            "rules_engine": rules,
        }
        c1, c2 = st.columns(2)
        c1.download_button(
            "⬇️ Full results (JSON)",
            json.dumps(raw, indent=2, default=str),
            f"{document_id}_results.json",
            "application/json",
            width="stretch",
        )
        c2.download_button(
            "⬇️ Final record (JSON)",
            json.dumps(export_record(run), indent=2, default=str),
            f"{document_id}_record.json",
            "application/json",
            width="stretch",
        )
        st.json(raw, expanded=False)

    # -----------------------------------------------------------------------
    # Human review (Member 4): edit, re-validate, approve, save and export
    # -----------------------------------------------------------------------
    st.divider()
    st.markdown('<div class="section-title">👁️ Human review</div>', unsafe_allow_html=True)
    render_review(
        run.rules,
        document_bytes=sel_doc.original_bytes,
        document_type=(
            "application/pdf"
            if sel_doc.is_pdf
            else ("image/jpeg" if sel_doc.extension in {".jpg", ".jpeg"} else "image/png")
        ),
        document_id=document_id,
        review_run_id=st.session_state.review_runs[sel_key]["review_run_id"],
        authorized_export=True,
        show_header=False,
    )

# ---------------------------------------------------------------------------
# Integration contract (for teammates)
# ---------------------------------------------------------------------------
with st.expander("🔗 Integration contract (for teammates)"):
    st.markdown(
        """
**To OCR:** `st.session_state['ocr_payloads']` is a list with one dict per unique
document (`PreparedDocument.ocr_payload()`): `filename`, `content` (bytes),
`mime_type`, `sha256`, `warnings`, `page_count`, `is_pdf`, `preprocessing`.
PDFs arrive as the original bytes (the OCR module renders pages); images arrive
as prepared PNG bytes. The untouched upload is `PreparedDocument.original_bytes`.

**To the review dashboard:** `document_bytes` and `document_type` hold the single
original upload; `document_id`, `review_run_id` and `extraction_result` identify
this run. OCR/validation publishes results with
`publish_extraction_result(document_id, validated_fields)`.
        """
    )
