"""
postprocessing.py - OCR result cleaning and image noise removal.

Three-stage post-processing pipeline:

  Stage 1 — Confidence Filtering
      Remove any OCR text box whose confidence score < threshold (default 0.5).

  Stage 2 — Overlap / Noise Removal
      Suppress duplicate or heavily-overlapping boxes using IoU-based
      Non-Maximum Suppression (NMS). Also removes boxes whose text is
      blank, single-character, or matches common noise patterns.

  Stage 3 — Morphological Image Cleaning
      Apply OpenCV morphological operations (opening / closing) on the
      preprocessed image to erase background speckles and thin noise
      while preserving text strokes.

Usage (as module):
    from modules.postprocessing import OCRPostProcessor

    processor = OCRPostProcessor()

    # Clean OCR boxes
    clean_boxes = processor.clean_boxes(raw_ocr_boxes)

    # Clean image
    clean_img   = processor.clean_image(gray_img)

    # Both at once
    clean_img, clean_boxes = processor.run(gray_img, raw_ocr_boxes)

Standalone:
    py modules/postprocessing.py <ocr_result.json>  [image_path]
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Box = Dict[str, Any]   # single OCR dict from InvoiceOCR.run()


# ─── Default parameters ───────────────────────────────────────────────────────

DEFAULT_CONF_THRESHOLD  = 0.50   # drop boxes below this confidence
DEFAULT_IOU_THRESHOLD   = 0.45   # IoU above this → suppress the lower-conf box
DEFAULT_MIN_TEXT_LEN    = 2      # drop text shorter than this (noise/stray chars)

# Morphological defaults
DEFAULT_MORPH_KERNEL    = (3, 3) # structuring element size
DEFAULT_OPEN_ITER       = 1      # opening iterations  (removes small blobs)
DEFAULT_CLOSE_ITER      = 1      # closing iterations  (fills small holes)

# Regex patterns that flag likely noise
_NOISE_PATTERNS = re.compile(
    r"""^(
        [^a-zA-Z0-9]{1,3}    |   # only special chars
        [\|\-\_\=\~\.]{2,}   |   # repeated dashes/bars
        [ilIl1]{3,}          |   # repeated thin strokes mis-read as letters
        \s+                       # only whitespace
    )$""",
    re.VERBOSE,
)


# ─── Helper: axis-aligned bounding box ops ───────────────────────────────────

def _area(box: Box) -> float:
    return max(0.0, box["x_max"] - box["x_min"]) * max(0.0, box["y_max"] - box["y_min"])


def _iou(a: Box, b: Box) -> float:
    """Intersection-over-Union of two axis-aligned boxes."""
    ix1 = max(a["x_min"], b["x_min"])
    iy1 = max(a["y_min"], b["y_min"])
    ix2 = min(a["x_max"], b["x_max"])
    iy2 = min(a["y_max"], b["y_max"])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _is_noise(text: str, min_len: int) -> bool:
    """Return True if the text string looks like OCR noise."""
    t = text.strip()
    if len(t) < min_len:
        return True
    if _NOISE_PATTERNS.match(t):
        return True
    return False


# ─── Stage 1: Confidence filtering ───────────────────────────────────────────

def filter_by_confidence(
    boxes: List[Box],
    threshold: float = DEFAULT_CONF_THRESHOLD,
) -> List[Box]:
    """
    Remove OCR boxes whose confidence score is below `threshold`.

    Args:
        boxes     : Raw OCR result list from InvoiceOCR.run()
        threshold : Minimum acceptable confidence (default 0.50)

    Returns:
        Filtered list, preserving original order.
    """
    before = len(boxes)
    kept = [b for b in boxes if b.get("confidence", 0.0) >= threshold]
    dropped = before - len(kept)
    logger.info(
        f"[Stage 1] Confidence filter (≥{threshold}): "
        f"kept {len(kept)}/{before} boxes, dropped {dropped}."
    )
    return kept


# ─── Stage 2: Overlap and noise removal ──────────────────────────────────────

def remove_noise_boxes(
    boxes: List[Box],
    min_text_len: int = DEFAULT_MIN_TEXT_LEN,
) -> List[Box]:
    """
    Remove boxes whose recognised text is blank, too short, or
    matches known noise patterns (repeated dashes, special chars, etc.).

    Args:
        boxes        : OCR box list.
        min_text_len : Minimum character count to keep a box.

    Returns:
        Filtered list.
    """
    before = len(boxes)
    kept = [b for b in boxes if not _is_noise(b.get("text", ""), min_text_len)]
    logger.info(
        f"[Stage 2a] Noise-text filter: "
        f"kept {len(kept)}/{before} boxes."
    )
    return kept


def non_maximum_suppression(
    boxes: List[Box],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> List[Box]:
    """
    Suppress overlapping bounding boxes using IoU-based NMS.
    When two boxes overlap beyond `iou_threshold`, the one with the
    lower confidence is removed.

    Args:
        boxes         : OCR box list (any order).
        iou_threshold : If IoU ≥ this, suppress the lower-confidence box.

    Returns:
        De-duplicated list, sorted by confidence (descending).
    """
    if not boxes:
        return []

    # Sort by confidence descending
    sorted_boxes = sorted(boxes, key=lambda b: b.get("confidence", 0.0), reverse=True)
    keep = []
    suppressed = set()

    for i, anchor in enumerate(sorted_boxes):
        if i in suppressed:
            continue
        keep.append(anchor)
        for j, candidate in enumerate(sorted_boxes[i + 1:], start=i + 1):
            if j in suppressed:
                continue
            if _iou(anchor, candidate) >= iou_threshold:
                suppressed.add(j)

    logger.info(
        f"[Stage 2b] NMS (IoU≥{iou_threshold}): "
        f"kept {len(keep)}/{len(boxes)} boxes, "
        f"suppressed {len(boxes) - len(keep)}."
    )
    return keep


def remove_overlapping_boxes(
    boxes: List[Box],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    min_text_len: int    = DEFAULT_MIN_TEXT_LEN,
) -> List[Box]:
    """
    Combined Stage 2: remove noise text boxes, then apply NMS.

    Returns:
        Cleaned box list, sorted confidence-descending.
    """
    boxes = remove_noise_boxes(boxes, min_text_len)
    boxes = non_maximum_suppression(boxes, iou_threshold)
    return boxes


# ─── Stage 3: Morphological image cleaning ───────────────────────────────────

def apply_morphological_cleaning(
    img: np.ndarray,
    kernel_size  : Tuple[int, int] = DEFAULT_MORPH_KERNEL,
    open_iter    : int             = DEFAULT_OPEN_ITER,
    close_iter   : int             = DEFAULT_CLOSE_ITER,
    binarize     : bool            = True,
) -> np.ndarray:
    """
    Apply morphological operations to remove background noise from an image.

    Steps:
      1. Convert to grayscale if needed
      2. (Optional) Otsu binarization
      3. Morphological Opening  — erodes then dilates; removes small white speckles
      4. Morphological Closing  — dilates then erodes; fills small dark holes in text

    Args:
        img         : Input image (BGR or grayscale).
        kernel_size : Structuring element size (width, height).
        open_iter   : Opening iterations.
        close_iter  : Closing iterations.
        binarize    : Apply Otsu threshold before morphology.

    Returns:
        Cleaned grayscale image (uint8).
    """
    # ── Step 1: ensure grayscale ──────────────────────────────────────────
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # ── Step 2: binarize ──────────────────────────────────────────────────
    if binarize:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binary = gray

    # ── Step 3: morphological kernel ──────────────────────────────────────
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    # ── Step 4: Opening — remove isolated foreground speckles ─────────────
    if open_iter > 0:
        opened = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, kernel, iterations=open_iter
        )
    else:
        opened = binary

    # ── Step 5: Closing — fill small holes inside text strokes ────────────
    if close_iter > 0:
        closed = cv2.morphologyEx(
            opened, cv2.MORPH_CLOSE, kernel, iterations=close_iter
        )
    else:
        closed = opened

    logger.debug(
        f"[Stage 3] Morphological cleaning applied "
        f"(kernel={kernel_size}, open×{open_iter}, close×{close_iter})."
    )
    return closed


# ─── Main post-processor class ────────────────────────────────────────────────

class OCRPostProcessor:
    """
    Three-stage OCR post-processing pipeline:

      Stage 1 — Confidence Filtering  (drop boxes < threshold)
      Stage 2 — Overlap/Noise Removal (NMS + noise-text filter)
      Stage 3 — Morphological Cleaning (image-level noise removal)

    Usage:
        pp = OCRPostProcessor(conf_threshold=0.5, iou_threshold=0.45)

        # Clean boxes only
        clean_boxes = pp.clean_boxes(raw_boxes)

        # Clean image only
        clean_img   = pp.clean_image(gray_img)

        # Full pipeline
        clean_img, clean_boxes = pp.run(img, raw_boxes)
    """

    def __init__(
        self,
        conf_threshold : float             = DEFAULT_CONF_THRESHOLD,
        iou_threshold  : float             = DEFAULT_IOU_THRESHOLD,
        min_text_len   : int               = DEFAULT_MIN_TEXT_LEN,
        morph_kernel   : Tuple[int, int]   = DEFAULT_MORPH_KERNEL,
        open_iter      : int               = DEFAULT_OPEN_ITER,
        close_iter     : int               = DEFAULT_CLOSE_ITER,
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.min_text_len   = min_text_len
        self.morph_kernel   = morph_kernel
        self.open_iter      = open_iter
        self.close_iter     = close_iter

    # ── Public API ────────────────────────────────────────────────────────

    def clean_boxes(self, boxes: List[Box]) -> List[Box]:
        """
        Stage 1 + Stage 2: confidence filter → noise removal → NMS.

        Args:
            boxes: Raw OCR result from InvoiceOCR.run()

        Returns:
            Cleaned, de-overlapped box list.
        """
        boxes = filter_by_confidence(boxes, self.conf_threshold)
        boxes = remove_overlapping_boxes(boxes, self.iou_threshold, self.min_text_len)
        logger.info(f"clean_boxes: {len(boxes)} boxes retained after post-processing.")
        return boxes

    def clean_image(self, img: np.ndarray) -> np.ndarray:
        """
        Stage 3: apply morphological operations to remove image noise.

        Args:
            img: BGR or grayscale image array.

        Returns:
            Cleaned binary/grayscale image.
        """
        return apply_morphological_cleaning(
            img,
            kernel_size = self.morph_kernel,
            open_iter   = self.open_iter,
            close_iter  = self.close_iter,
        )

    def run(
        self,
        img   : np.ndarray,
        boxes : List[Box],
    ) -> Tuple[np.ndarray, List[Box]]:
        """
        Full pipeline: clean image AND clean boxes.

        Args:
            img   : Preprocessed invoice image (BGR or grayscale).
            boxes : Raw OCR result from InvoiceOCR.run()

        Returns:
            (clean_image, clean_boxes)
        """
        clean_img   = self.clean_image(img)
        clean_boxes = self.clean_boxes(boxes)
        return clean_img, clean_boxes

    # ── Summary helpers ───────────────────────────────────────────────────

    @staticmethod
    def summary(original: List[Box], cleaned: List[Box]) -> Dict[str, Any]:
        """Return a dict summarising what was removed."""
        dropped = len(original) - len(cleaned)
        return {
            "original_count" : len(original),
            "cleaned_count"  : len(cleaned),
            "dropped_count"  : dropped,
            "drop_rate_pct"  : round(dropped / max(len(original), 1) * 100, 1),
            "avg_confidence" : round(
                sum(b["confidence"] for b in cleaned) / max(len(cleaned), 1), 4
            ),
        }


# ─── Field extraction (preserved from original) ──────────────────────────────
# Import from the original extraction logic so nothing is broken downstream.

import re as _re

PATTERNS = {
    "invoice_number": [
        r"(?:invoice\s*(?:no|number|#|num)[:\s#]*)([\w\-/]+)",
        r"(?:inv[:\s#]*)([\w\-/]+)",
        r"(?:bill\s*(?:no|number)[:\s#]*)([\w\-/]+)",
    ],
    "date": [
        r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b",
        r"\b(\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})\b",
        r"\b(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4})\b",
    ],
    "total_amount": [
        r"(?:total\s*(?:amount|due|payable)?[:\s]*)[€$£₹]?\s*([\d,]+\.?\d*)",
        r"(?:grand\s*total[:\s]*)[€$£₹]?\s*([\d,]+\.?\d*)",
    ],
    "email": [r"[a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9.\-]+"],
}


# ─── Standalone entry-point ───────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: py modules/postprocessing.py <ocr_result.json> [image_path]")
        sys.exit(1)

    json_path = Path(sys.argv[1])
    img_path  = Path(sys.argv[2]) if len(sys.argv) >= 3 else None

    with open(json_path, encoding="utf-8") as f:
        raw_boxes = json.load(f)

    pp = OCRPostProcessor(conf_threshold=0.5, iou_threshold=0.45)

    print(f"\n{'='*65}")
    print(f"  OCR Post-Processing")
    print(f"  Source JSON : {json_path.name}")
    print(f"{'='*65}")

    # ── Stage 1 & 2: clean boxes ──────────────────────────────────────────
    clean_boxes = pp.clean_boxes(raw_boxes)
    stats = OCRPostProcessor.summary(raw_boxes, clean_boxes)

    print(f"\n  [Stage 1+2] Box Cleaning Summary")
    print(f"  Original boxes  : {stats['original_count']}")
    print(f"  Cleaned boxes   : {stats['cleaned_count']}")
    print(f"  Dropped         : {stats['dropped_count']} ({stats['drop_rate_pct']}%)")
    print(f"  Avg confidence  : {stats['avg_confidence']}")

    # ── Stage 3: clean image (if provided) ───────────────────────────────
    if img_path and img_path.exists():
        img = cv2.imread(str(img_path))
        clean_img = pp.clean_image(img)
        out = img_path.parent / f"{img_path.stem}_clean_morph.jpg"
        cv2.imwrite(str(out), clean_img)
        print(f"\n  [Stage 3] Morphological cleaned image → {out}")
    else:
        print("\n  [Stage 3] No image provided — skipping morphological cleaning.")

    # ── Print cleaned boxes ───────────────────────────────────────────────
    print(f"\n  Retained boxes:")
    print(f"  {'#':<4} {'TEXT':<40} {'CONF':>6}")
    print("  " + "-" * 54)
    for i, b in enumerate(clean_boxes, 1):
        print(f"  {i:<4} {b['text']:<40} {b['confidence']:>6.4f}")

    # Save cleaned JSON
    out_json = json_path.parent / f"{json_path.stem}_cleaned.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(clean_boxes, f, indent=2, ensure_ascii=False)
    print(f"\n  Cleaned JSON saved → {out_json}")
    print(f"{'='*65}\n")
