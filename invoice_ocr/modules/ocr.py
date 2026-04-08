"""
ocr.py - PaddleOCR integration for invoice text extraction
"""

import numpy as np
import logging
from typing import List, Tuple, Optional
from pathlib import Path

from paddleocr import PaddleOCR
from config import settings
from models.invoice_schema import OCRTextBlock, BoundingBox

logger = logging.getLogger(__name__)


class InvoiceOCR:
    """
    Wraps PaddleOCR to extract text blocks with bounding boxes
    and confidence scores from preprocessed invoice images.
    """

    _instance: Optional[PaddleOCR] = None  # Singleton OCR engine

    def __init__(self):
        self.engine = self._get_engine()

    def _get_engine(self) -> PaddleOCR:
        """Return a shared (singleton) PaddleOCR instance."""
        if InvoiceOCR._instance is None:
            logger.info("Initializing PaddleOCR engine...")
            InvoiceOCR._instance = PaddleOCR(
                use_angle_cls=settings.OCR_USE_ANGLE_CLS,
                lang=settings.OCR_LANG,
                use_gpu=settings.OCR_USE_GPU,
                det_db_thresh=settings.OCR_DET_DB_THRESH,
                det_db_box_thresh=settings.OCR_DET_DB_BOX_THRESH,
                rec_algorithm=settings.OCR_REC_ALGORITHM,
                show_log=settings.DEBUG,
            )
            logger.info("PaddleOCR engine ready.")
        return InvoiceOCR._instance

    def run(self, img: np.ndarray) -> List[OCRTextBlock]:
        """
        Run OCR on a preprocessed image.

        Args:
            img: Preprocessed numpy image (grayscale or BGR).

        Returns:
            List of OCRTextBlock with text, confidence, and bounding boxes.
        """
        # PaddleOCR expects BGR or grayscale; convert grayscale → BGR
        if len(img.shape) == 2:
            import cv2
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        raw_result = self.engine.ocr(img, cls=settings.OCR_USE_ANGLE_CLS)

        if not raw_result or raw_result[0] is None:
            logger.warning("PaddleOCR returned no results.")
            return []

        blocks: List[OCRTextBlock] = []
        for line in raw_result[0]:
            if line is None:
                continue
            box_points, (text, confidence) = line
            bbox = self._points_to_bbox(box_points)
            blocks.append(
                OCRTextBlock(
                    text=text.strip(),
                    confidence=round(float(confidence), 4),
                    bounding_box=bbox,
                )
            )

        logger.info(f"OCR extracted {len(blocks)} text blocks.")
        return blocks

    def run_from_path(self, image_path: Path) -> List[OCRTextBlock]:
        """Run OCR directly from an image file path."""
        raw_result = self.engine.ocr(str(image_path), cls=settings.OCR_USE_ANGLE_CLS)
        if not raw_result or raw_result[0] is None:
            return []

        blocks: List[OCRTextBlock] = []
        for line in raw_result[0]:
            if line is None:
                continue
            box_points, (text, confidence) = line
            bbox = self._points_to_bbox(box_points)
            blocks.append(
                OCRTextBlock(
                    text=text.strip(),
                    confidence=round(float(confidence), 4),
                    bounding_box=bbox,
                )
            )
        return blocks

    @staticmethod
    def _points_to_bbox(points: List[List[float]]) -> BoundingBox:
        """Convert PaddleOCR quad points to axis-aligned bounding box."""
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return BoundingBox(
            x_min=round(min(xs), 2),
            y_min=round(min(ys), 2),
            x_max=round(max(xs), 2),
            y_max=round(max(ys), 2),
        )

    @staticmethod
    def get_raw_text(blocks: List[OCRTextBlock]) -> str:
        """Concatenate all text blocks into a single string."""
        return "\n".join(b.text for b in blocks if b.text)

    @staticmethod
    def average_confidence(blocks: List[OCRTextBlock]) -> float:
        """Compute the average confidence across all text blocks."""
        if not blocks:
            return 0.0
        return round(sum(b.confidence for b in blocks) / len(blocks), 4)
