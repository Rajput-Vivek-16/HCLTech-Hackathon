"""Document ingestion and image preprocessing for the OCR pipeline.

Member 1 scope: accept PDF / JPG / JPEG / PNG, validate, preserve the original
bytes, run basic quality checks, and hand a prepared document to the OCR module.

Contract with the OCR module
----------------------------
* PDFs are validated but NEVER rendered here. ``ocr_bytes`` is the original PDF,
  byte for byte. The OCR module must render PDF pages itself (e.g. PyMuPDF at
  ~200 DPI) before running Tesseract.
* Images are decoded, EXIF-rotated, resized, optionally deskewed and returned
  as RGB PNG bytes in ``ocr_bytes``.
* ``original_bytes`` always holds the untouched upload.
* Use ``PreparedDocument.ocr_payload()`` as the hand-off dict.
* ``warnings`` are quality problems (reviewer should care); ``notes`` are
  informational only (what preprocessing did).
* Cleanup order for images: EXIF fix -> resize -> orientation (90/180/270)
  -> noise reduction (only if noisy) -> deskew (small angles).
* Auto-orientation uses Tesseract OSD (``pytesseract`` + the Tesseract binary).
  If it is unavailable or not confident, nothing is rotated and a note says so;
  the UI offers manual rotation instead.
"""

from __future__ import annotations

import hashlib
import io
import os
from dataclasses import dataclass, field

import cv2  # required: blur and skew checks are calibrated for OpenCV
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

try:
    import pymupdf as fitz
except ImportError:  # older PyMuPDF exposes the same API as ``fitz``
    import fitz

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
DEFAULT_MAX_SIZE_MB = 20
DEFAULT_LONG_EDGE = 1920
MAX_PDF_PAGES = 5
MAX_IMAGE_PIXELS = 50_000_000  # guards against decompression bombs
PDF_MIME_TYPE = "application/pdf"
IMAGE_MIME_TYPE = "image/png"
BLUR_THRESHOLD = 80.0  # Laplacian variance; starting guess, tune on samples
MIN_IMAGE_DIMENSION = 600  # px, shorter side
MAX_SKEW_DEGREES = 15.0  # ignore larger angles (likely not skew)
MIN_SKEW_DEGREES = 0.3  # ignore tiny angles
NOISE_SIGMA_THRESHOLD = 2.5  # estimated noise std (0-255); starting guess, tune
OSD_MIN_CONFIDENCE = 2.0  # Tesseract orientation_conf needed to rotate
OSD_MAX_EDGE = 1200  # shrink before OSD for speed
VALID_ROTATIONS = (0, 90, 180, 270)  # clockwise degrees

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

_MAGIC = {
    ".pdf": (b"%PDF",),
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class PageResult:
    """Prepared image and diagnostics for one image page."""

    page_num: int
    original_image: Image.Image
    processed_image: Image.Image
    width: int
    height: int
    warnings: list[str] = field(default_factory=list)
    deskew_angle: float = 0.0
    sharpness: float = 0.0
    noise_sigma: float = 0.0
    denoised: bool = False
    rotation_applied: int = 0  # clockwise degrees
    orientation_method: str = "none"  # none | manual | tesseract-osd | unavailable

    def processed_bytes(self, fmt: str = "PNG") -> bytes:
        """Return the prepared image encoded for OCR."""
        buffer = io.BytesIO()
        self.processed_image.save(buffer, format=fmt)
        return buffer.getvalue()


@dataclass
class PreparedDocument:
    """Stable hand-off object for the downstream OCR module."""

    filename: str
    original_bytes: bytes
    ocr_bytes: bytes
    mime_type: str
    sha256: str
    file_size: int
    extension: str
    page_count: int
    pages: list[PageResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def is_pdf(self) -> bool:
        return self.extension == ".pdf"

    @property
    def ocr_filename(self) -> str:
        """Filename matching the format of ``ocr_bytes``."""
        if self.is_pdf:
            return self.filename
        return f"{os.path.splitext(self.filename)[0]}.png"

    def ocr_payload(self) -> dict:
        """Dict handed to the OCR adapter."""
        return {
            "filename": self.ocr_filename,
            "content": self.ocr_bytes,
            "mime_type": self.mime_type,
            "sha256": self.sha256,
            "warnings": list(self.warnings),
            "page_count": self.page_count,
            "is_pdf": self.is_pdf,
            "preprocessing": self.preprocessing_summary(),
        }

    def preprocessing_summary(self) -> dict:
        """What was done to the image (empty for PDFs)."""
        if not self.pages:
            return {}
        pg = self.pages[0]
        return {
            "rotation_applied": pg.rotation_applied,
            "orientation_method": pg.orientation_method,
            "denoised": pg.denoised,
            "noise_sigma": round(pg.noise_sigma, 2),
            "deskew_angle": round(pg.deskew_angle, 2),
            "sharpness": round(pg.sharpness, 1),
        }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate_file(file_bytes: bytes, filename: str, max_size_mb: float) -> str:
    if not isinstance(file_bytes, bytes):
        raise ValueError("Uploaded content must be bytes.")
    if not filename or not filename.strip():
        raise ValueError("A filename is required.")
    if max_size_mb <= 0:
        raise ValueError("The maximum file size must be greater than zero.")

    extension = os.path.splitext(filename)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ValueError(f"Unsupported file type '{extension}'. Allowed: {allowed}")
    if not file_bytes:
        raise ValueError("Uploaded file is empty (0 bytes).")

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ValueError(f"File is {size_mb:.1f} MB, which exceeds the {max_size_mb:g} MB limit.")

    # Magic bytes: the content must match the claimed extension.
    if not file_bytes.startswith(_MAGIC[extension]):
        raise ValueError(
            f"File content does not look like a valid '{extension}' file (extension and content do not match)."
        )
    return extension


def _open_pdf(file_bytes: bytes):
    """Open a PDF for validation only. Rejects corrupt, empty, protected, long files."""
    try:
        document = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several error types
        raise ValueError(f"Cannot open PDF (corrupted or unsupported): {exc}") from exc

    if document.needs_pass or document.is_encrypted:
        document.close()
        raise ValueError("PDF is password-protected. Please upload an unlocked copy.")
    if document.page_count == 0:
        document.close()
        raise ValueError("PDF has zero pages.")
    if document.page_count > MAX_PDF_PAGES:
        count = document.page_count
        document.close()
        raise ValueError(f"PDF has {count} pages; the limit is {MAX_PDF_PAGES}.")
    return document


def _open_image(file_bytes: bytes) -> Image.Image:
    """Decode an image, apply EXIF rotation, flatten transparency, return RGB."""
    try:
        with Image.open(io.BytesIO(file_bytes)) as image:
            image.load()
            image = ImageOps.exif_transpose(image)  # fixes sideways phone photos
            if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
                rgba = image.convert("RGBA")
                background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                background.alpha_composite(rgba)
                return background.convert("RGB")
            return image.convert("RGB")
    except Image.DecompressionBombError as exc:
        raise ValueError("Image is too large to process safely.") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"Cannot open image (corrupted or unsupported): {exc}") from exc


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------
def _resize(image: Image.Image, long_edge: int) -> Image.Image:
    if long_edge <= 0:
        raise ValueError("The resize long edge must be greater than zero.")
    width, height = image.size
    longest = max(width, height)
    if longest <= long_edge:
        return image.copy()
    scale = long_edge / longest
    return image.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)


def _sharpness(image: Image.Image) -> float:
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _quality_warnings(image: Image.Image) -> tuple[list[str], float]:
    """Blur and resolution checks on the ORIGINAL image."""
    warnings: list[str] = []
    score = _sharpness(image)
    if score < BLUR_THRESHOLD:
        warnings.append(f"Image may be blurry (sharpness score {score:.1f}).")
    width, height = image.size
    if min(width, height) < MIN_IMAGE_DIMENSION:
        warnings.append(f"Low resolution ({width}x{height}); OCR accuracy may be reduced.")
    return warnings, score


def _estimate_skew_angle(image: Image.Image) -> float:
    """Median angle (degrees) of long near-horizontal lines; 0.0 if unsure."""
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(30, min(gray.shape) // 8),
        minLineLength=max(30, min(gray.shape) // 5),
        maxLineGap=20,
    )
    if lines is None:
        return 0.0
    angles: list[float] = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        angle = float(np.degrees(np.arctan2(int(y2) - int(y1), int(x2) - int(x1))))
        if -MAX_SKEW_DEGREES <= angle <= MAX_SKEW_DEGREES:
            angles.append(angle)
    if not angles:
        return 0.0
    angle = float(np.median(angles))
    return 0.0 if abs(angle) < MIN_SKEW_DEGREES else angle


def _deskew(image: Image.Image, angle: float) -> Image.Image:
    if angle == 0.0:
        return image
    # PIL rotates counter-clockwise for positive angles; a line sloping
    # down-right (positive angle in image coordinates) needs that rotation.
    return image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor="white")


def _estimate_noise_sigma(image: Image.Image) -> float:
    """Robust noise estimate (std on a 0-255 scale).

    Uses a Laplacian-difference kernel and the median absolute value, so text
    edges (a small share of pixels) barely affect it. Clean scans give ~0-1.
    Caveat: noise on pure-white (clipped) paper is under-estimated.
    """
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY).astype(np.float32)
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    response = cv2.filter2D(gray, cv2.CV_32F, kernel)
    # kernel energy = 36 -> response std = 6 * sigma; MAD -> std factor 0.6745
    return float(np.median(np.abs(response)) / (0.6745 * 6.0))


def _denoise(image: Image.Image, sigma: float) -> Image.Image:
    """Non-local-means on the luminance channel only (colour left alone).

    Roughly 1.5 s for a 1400x1920 page, so it is only run when noise is detected.
    """
    strength = float(np.clip(1.2 * sigma, 5.0, 12.0))
    lab = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.fastNlMeansDenoising(lab[..., 0], None, strength, 5, 11)
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def _rotate_cw(image: Image.Image, degrees: int) -> Image.Image:
    """Lossless clockwise rotation by a multiple of 90 degrees."""
    degrees %= 360
    if degrees == 0:
        return image
    transposes = {
        90: Image.Transpose.ROTATE_270,  # PIL's ROTATE_n is counter-clockwise
        180: Image.Transpose.ROTATE_180,
        270: Image.Transpose.ROTATE_90,
    }
    return image.transpose(transposes[degrees])


def _osd_rotation(image: Image.Image) -> tuple[int | None, float]:
    """Ask Tesseract OSD how far to rotate clockwise. (None, 0) if unavailable."""
    try:
        import pytesseract
    except ImportError:
        return None, 0.0
    small = image.copy()
    small.thumbnail((OSD_MAX_EDGE, OSD_MAX_EDGE))
    try:
        osd = pytesseract.image_to_osd(small, config="--psm 0", output_type=pytesseract.Output.DICT)
        return int(osd["rotate"]), float(osd["orientation_conf"])
    except Exception:  # binary missing, too little text, etc.
        return None, 0.0


def _auto_orient(image: Image.Image) -> tuple[Image.Image, int, str]:
    """Fix 90/180/270 rotation using Tesseract OSD when it is confident."""
    rotate, confidence = _osd_rotation(image)
    if rotate is None or confidence < OSD_MIN_CONFIDENCE:
        return image, 0, "unavailable"
    if rotate in (90, 180, 270):
        return _rotate_cw(image, rotate), rotate, "tesseract-osd"
    return image, 0, "tesseract-osd"


# ---------------------------------------------------------------------------
# PDF preview (display only)
# ---------------------------------------------------------------------------
def render_pdf_preview(pdf_bytes: bytes, dpi: int = 110) -> list[Image.Image]:
    """Render PDF pages to images so the UI can SHOW them.

    Preview only: these images are never sent to OCR. The OCR hand-off still
    uses the untouched PDF bytes (``PreparedDocument.ocr_bytes``).
    """
    document = _open_pdf(pdf_bytes)  # re-validates: encrypted / corrupt / too long
    images: list[Image.Image] = []
    try:
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for page in document:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    finally:
        document.close()
    return images


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def process_upload(
    file_bytes: bytes,
    filename: str,
    *,
    max_size_mb: float = DEFAULT_MAX_SIZE_MB,
    long_edge: int = DEFAULT_LONG_EDGE,
    deskew: bool = True,
    denoise: bool = True,
    auto_orient: bool = True,
    manual_rotation: int = 0,
) -> PreparedDocument:
    """Validate and prepare one document for downstream OCR.

    ``manual_rotation`` (0/90/180/270, clockwise) overrides auto-orientation.
    Raises ``ValueError`` with a user-friendly message for any rejected file.
    """
    if manual_rotation not in VALID_ROTATIONS:
        raise ValueError(f"manual_rotation must be one of {VALID_ROTATIONS}.")
    extension = _validate_file(file_bytes, filename, max_size_mb)
    digest = hashlib.sha256(file_bytes).hexdigest()

    # ---- PDF: validate only, pass through unchanged ----------------------
    if extension == ".pdf":
        document = _open_pdf(file_bytes)
        try:
            page_count = document.page_count
            raw_meta = document.metadata or {}
        finally:
            document.close()
        metadata = {k: v for k, v in raw_meta.items() if v}
        return PreparedDocument(
            filename=filename,
            original_bytes=file_bytes,
            ocr_bytes=file_bytes,
            mime_type=PDF_MIME_TYPE,
            sha256=digest,
            file_size=len(file_bytes),
            extension=extension,
            page_count=page_count,
            notes=["PDF passed through unchanged; the OCR module renders its pages."],
            metadata=metadata,
        )

    # ---- Image: decode, check quality, resize, orient, denoise, deskew ----
    original = _open_image(file_bytes)
    warnings, sharpness = _quality_warnings(original)
    notes: list[str] = []
    processed = _resize(original, long_edge)

    rotation, orientation_method = 0, "none"
    if manual_rotation:
        processed = _rotate_cw(processed, manual_rotation)
        rotation, orientation_method = manual_rotation, "manual"
        notes.append(f"Rotated {manual_rotation} degrees clockwise (manual).")
    elif auto_orient:
        processed, rotation, orientation_method = _auto_orient(processed)
        if rotation:
            notes.append(f"Auto-rotated {rotation} degrees clockwise (Tesseract OSD).")
        elif orientation_method == "unavailable":
            notes.append(
                "Orientation not verified (Tesseract OSD unavailable or not "
                "confident). If the page is sideways, use manual rotation."
            )

    noise_sigma = _estimate_noise_sigma(processed)
    denoised = False
    if denoise and noise_sigma >= NOISE_SIGMA_THRESHOLD:
        processed = _denoise(processed, noise_sigma)
        denoised = True
        notes.append(f"Noise reduced (estimated noise level {noise_sigma:.1f}).")

    angle = 0.0
    if deskew:
        angle = _estimate_skew_angle(processed)
        if angle:
            processed = _deskew(processed, angle)
            notes.append(f"Deskewed by {angle:.1f} degrees.")

    page = PageResult(
        page_num=1,
        original_image=original.copy(),
        processed_image=processed,
        width=processed.width,
        height=processed.height,
        warnings=warnings,
        deskew_angle=angle,
        sharpness=sharpness,
        noise_sigma=noise_sigma,
        denoised=denoised,
        rotation_applied=rotation,
        orientation_method=orientation_method,
    )
    return PreparedDocument(
        filename=filename,
        original_bytes=file_bytes,
        ocr_bytes=page.processed_bytes("PNG"),
        mime_type=IMAGE_MIME_TYPE,
        sha256=digest,
        file_size=len(file_bytes),
        extension=extension,
        page_count=1,
        pages=[page],
        warnings=list(warnings),
        notes=notes,
        metadata={"original_size": original.size, "processed_size": processed.size},
    )
