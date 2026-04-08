"""
preprocessing.py - Invoice Image Preprocessing Pipeline using OpenCV

Steps:
  1. Resize          - standardise dimensions
  2. Grayscale       - reduce to single channel
  3. CLAHE           - contrast-limited adaptive histogram equalisation
  4. Gaussian blur   - noise removal
  5. Skew correction - deskew via Hough / minAreaRect
Returns the processed image as a numpy ndarray.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)

# ─── Default hyper-parameters ────────────────────────────────────────────────
DEFAULT_TARGET_W   = 1024          # target width after resize
DEFAULT_TARGET_H   = 1408          # target height after resize
CLAHE_CLIP_LIMIT   = 2.0           # CLAHE clip limit
CLAHE_TILE_GRID    = (8, 8)        # CLAHE tile grid size
GAUSSIAN_KERNEL    = (5, 5)        # Gaussian blur kernel (must be odd)
GAUSSIAN_SIGMA     = 0             # 0 = auto-calculated by OpenCV
SKEW_ANGLE_THRESH  = 0.5           # skip deskew if |angle| < threshold (deg)


# ─── Core preprocessing functions ────────────────────────────────────────────

def load_image(source: Union[str, Path, bytes, np.ndarray]) -> np.ndarray:
    """Load image from filepath, raw bytes, or pass-through ndarray."""
    if isinstance(source, np.ndarray):
        return source.copy()
    if isinstance(source, (str, Path)):
        img = cv2.imread(str(source))
        if img is None:
            raise ValueError(f"Cannot load image: {source}")
        return img
    if isinstance(source, bytes):
        arr = np.frombuffer(source, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Cannot decode image from bytes.")
        return img
    raise TypeError(f"Unsupported type: {type(source)}")


def resize_image(
    img: np.ndarray,
    target_w: int = DEFAULT_TARGET_W,
    target_h: int = DEFAULT_TARGET_H,
    keep_aspect: bool = True,
) -> np.ndarray:
    """
    Resize the image to target dimensions.
    If keep_aspect=True the image is scaled uniformly so it fits
    within (target_w × target_h); the rest is NOT padded.
    """
    h, w = img.shape[:2]
    if keep_aspect:
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
    else:
        new_w, new_h = target_w, target_h

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    logger.debug(f"resize: ({w}×{h}) → ({new_w}×{new_h})")
    return resized


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert BGR/RGB image to single-channel grayscale."""
    if len(img.shape) == 2:
        return img          # already grayscale
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_clahe(
    gray: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid: tuple = CLAHE_TILE_GRID,
) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalisation (CLAHE).
    Enhances local contrast while limiting noise amplification.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    enhanced = clahe.apply(gray)
    logger.debug("CLAHE applied.")
    return enhanced


def remove_gaussian_noise(
    img: np.ndarray,
    kernel: tuple = GAUSSIAN_KERNEL,
    sigma: int = GAUSSIAN_SIGMA,
) -> np.ndarray:
    """
    Gaussian blur for noise removal.
    Works on both grayscale (2D) and BGR (3D) images.
    """
    blurred = cv2.GaussianBlur(img, kernel, sigma)
    logger.debug(f"Gaussian blur applied (kernel={kernel}, σ={sigma}).")
    return blurred


def correct_skew(img: np.ndarray, angle_thresh: float = SKEW_ANGLE_THRESH) -> np.ndarray:
    """
    Detect and correct image skew using minAreaRect on white-pixel coords.

    Steps:
      - Invert colours so text pixels are 'white'
      - Find coords of all foreground pixels
      - Compute minimum area bounding rectangle → extract rotation angle
      - Rotate the image to align text horizontally
    """
    # Work with a grayscale copy for angle detection
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Invert so text = white
    inverted = cv2.bitwise_not(gray)

    # Threshold to create binary mask
    _, thresh = cv2.threshold(inverted, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Coordinates of all foreground (text) pixels
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 10:
        logger.debug("Too few foreground pixels; skipping skew correction.")
        return img

    # minAreaRect returns angle in [-90, 0)
    angle = cv2.minAreaRect(coords)[-1]

    # Normalise angle to [-45, 45]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < angle_thresh:
        logger.debug(f"Skew angle {angle:.2f}° below threshold; no correction.")
        return img

    # Rotate
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    logger.info(f"Skew corrected by {angle:.2f}°.")
    return rotated


# ─── Full pipeline ────────────────────────────────────────────────────────────

def preprocess(
    source: Union[str, Path, bytes, np.ndarray],
    target_w: int = DEFAULT_TARGET_W,
    target_h: int = DEFAULT_TARGET_H,
    keep_aspect: bool = True,
    return_steps: bool = False,
) -> Union[np.ndarray, dict]:
    """
    Full invoice preprocessing pipeline.

    Args:
        source       : File path, bytes, or numpy array.
        target_w/h   : Resize target dimensions.
        keep_aspect  : Preserve aspect ratio during resize.
        return_steps : If True, return a dict with all intermediate images.

    Returns:
        Processed grayscale image (ndarray), or dict of all steps.
    """
    # Step 1 — Load
    original = load_image(source)

    # Step 2 — Resize
    resized = resize_image(original, target_w, target_h, keep_aspect)

    # Step 3 — Grayscale
    gray = to_grayscale(resized)

    # Step 4 — Gaussian noise removal (before CLAHE to avoid amplifying noise)
    denoised = remove_gaussian_noise(gray)

    # Step 5 — CLAHE contrast enhancement
    enhanced = apply_clahe(denoised)

    # Step 6 — Skew correction
    deskewed = correct_skew(enhanced)

    if return_steps:
        return {
            "original":  original,
            "resized":   resized,
            "gray":      gray,
            "denoised":  denoised,
            "enhanced":  enhanced,
            "deskewed":  deskewed,
            "final":     deskewed,     # alias
        }
    return deskewed


# ─── Batch processing from Invoice-Dataset ───────────────────────────────────

INVOICE_DATASET_ROOT = Path(r"C:\FINALYEAR_PROJECT\Invoice-Dataset")
SUPPORTED_EXTS       = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def get_dataset_images(dataset_root: Path = INVOICE_DATASET_ROOT) -> list[Path]:
    """Recursively collect all supported image paths from the dataset folders."""
    images = []
    for ext in SUPPORTED_EXTS:
        images.extend(dataset_root.rglob(f"*{ext}"))
        images.extend(dataset_root.rglob(f"*{ext.upper()}"))
    images = sorted(set(images))
    logger.info(f"Found {len(images)} images in {dataset_root}")
    return images


def preprocess_dataset(
    dataset_root: Path = INVOICE_DATASET_ROOT,
    output_root: Optional[Path] = None,
    target_w: int = DEFAULT_TARGET_W,
    target_h: int = DEFAULT_TARGET_H,
    save: bool = True,
    max_images: Optional[int] = None,
) -> list[dict]:
    """
    Batch-preprocess all images in the Invoice-Dataset folder.

    Args:
        dataset_root : Root path of the invoice dataset.
        output_root  : Where to save processed images.
                       Defaults to dataset_root/../preprocessed/
        target_w/h   : Resize dimensions.
        save         : Write processed images to output_root.
        max_images   : Limit to first N images (None = all).

    Returns:
        List of dicts: {path, processed_image, saved_to}
    """
    if output_root is None:
        output_root = dataset_root.parent / "preprocessed"

    image_paths = get_dataset_images(dataset_root)
    if max_images:
        image_paths = image_paths[:max_images]

    results = []
    total = len(image_paths)

    for idx, img_path in enumerate(image_paths, 1):
        try:
            processed = preprocess(img_path, target_w, target_h)

            saved_to = None
            if save:
                # Mirror folder structure under output_root
                rel = img_path.relative_to(dataset_root)
                out_path = output_root / rel.parent / img_path.name
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), processed)
                saved_to = out_path

            results.append({
                "path":            img_path,
                "processed_image": processed,
                "saved_to":        saved_to,
                "status":          "ok",
            })

            if idx % 100 == 0 or idx == total:
                logger.info(f"  [{idx}/{total}] processed.")

        except Exception as e:
            logger.error(f"SKIP {img_path.name}: {e}")
            results.append({
                "path":            img_path,
                "processed_image": None,
                "saved_to":        None,
                "status":          f"error: {e}",
            })

    ok_count  = sum(1 for r in results if r["status"] == "ok")
    err_count = total - ok_count
    logger.info(f"Batch done — {ok_count} OK, {err_count} errors.")
    return results


# ─── Quick-test entrypoint ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # ── Single image test ──────────────────────────────────────────────────
    if len(sys.argv) == 2:
        src = Path(sys.argv[1])
        steps = preprocess(src, return_steps=True)
        print(f"\nProcessed: {src.name}")
        for name, img in steps.items():
            if isinstance(img, np.ndarray):
                print(f"  {name:12s} → shape={img.shape}, dtype={img.dtype}")

        # Show result
        cv2.imshow("Original",  cv2.resize(steps["original"], (512, 700)))
        cv2.imshow("Processed", cv2.resize(steps["final"],    (512, 700)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ── Batch test (first 10 images) ──────────────────────────────────────
    else:
        print("Running batch preprocessing on first 10 invoice images…")
        results = preprocess_dataset(
            save=True,
            max_images=10,
        )
        for r in results:
            status  = r["status"]
            name    = r["path"].name
            saved   = r["saved_to"] or "not saved"
            shape   = r["processed_image"].shape if r["processed_image"] is not None else "N/A"
            print(f"  [{status}] {name:40s} shape={shape}  → {saved}")
