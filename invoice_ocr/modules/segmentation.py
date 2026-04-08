"""
segmentation.py - Invoice region segmentation
Divides the invoice image / text blocks into logical regions:
  - header    (vendor info, logo, invoice number, date)
  - bill_to   (customer/billing info)
  - line_items (table of goods/services)
  - summary   (subtotals, tax, totals)
  - footer    (payment terms, notes, signatures)
"""

import numpy as np
import logging
from typing import List, Dict, Tuple

from models.invoice_schema import OCRTextBlock, InvoiceSegment, BoundingBox

logger = logging.getLogger(__name__)

# Vertical split ratios (fraction of total image height)
REGION_SPLITS = {
    "header":     (0.00, 0.20),
    "bill_to":    (0.20, 0.38),
    "line_items": (0.38, 0.75),
    "summary":    (0.75, 0.90),
    "footer":     (0.90, 1.00),
}


class InvoiceSegmenter:
    """
    Segments OCR text blocks into invoice regions based on
    their vertical position relative to the image height.
    Column-based heuristics are used for line-item detection.
    """

    def __init__(self, image_height: int, image_width: int):
        self.image_height = image_height
        self.image_width = image_width

    def segment(self, blocks: List[OCRTextBlock]) -> List[InvoiceSegment]:
        """
        Assign each OCR text block to an invoice region.

        Returns:
            List of InvoiceSegment, one per region.
        """
        # Build bucket dict
        buckets: Dict[str, List[OCRTextBlock]] = {r: [] for r in REGION_SPLITS}

        for block in blocks:
            region = self._classify_block(block)
            buckets[region].append(block)

        segments: List[InvoiceSegment] = []
        for region_name, region_blocks in buckets.items():
            y_start_ratio, y_end_ratio = REGION_SPLITS[region_name]
            bbox = BoundingBox(
                x_min=0,
                y_min=round(y_start_ratio * self.image_height, 2),
                x_max=float(self.image_width),
                y_max=round(y_end_ratio * self.image_height, 2),
            )
            segments.append(
                InvoiceSegment(
                    region=region_name,
                    bounding_box=bbox,
                    raw_text_blocks=region_blocks,
                )
            )
            logger.debug(
                f"Segment '{region_name}': {len(region_blocks)} text blocks."
            )

        logger.info(f"Segmentation complete — {len(segments)} regions.")
        return segments

    def _classify_block(self, block: OCRTextBlock) -> str:
        """
        Classify a single text block into a region
        based on its vertical center position.
        """
        y_center = (block.bounding_box.y_min + block.bounding_box.y_max) / 2
        ratio = y_center / self.image_height

        for region, (start, end) in REGION_SPLITS.items():
            if start <= ratio < end:
                return region
        return "footer"  # Default for anything at very bottom

    @staticmethod
    def get_segment_text(segment: InvoiceSegment) -> str:
        """Concatenate all text from a segment's blocks."""
        return "\n".join(
            b.text for b in segment.raw_text_blocks if b.text
        )

    def detect_table_rows(
        self, line_item_blocks: List[OCRTextBlock], col_count: int = 4
    ) -> List[List[OCRTextBlock]]:
        """
        Group line-item blocks into rows by clustering on Y-coordinate.
        Uses a simple proximity threshold.
        """
        if not line_item_blocks:
            return []

        sorted_blocks = sorted(
            line_item_blocks,
            key=lambda b: (b.bounding_box.y_min, b.bounding_box.x_min)
        )

        rows: List[List[OCRTextBlock]] = []
        current_row: List[OCRTextBlock] = [sorted_blocks[0]]
        row_y = sorted_blocks[0].bounding_box.y_min

        for block in sorted_blocks[1:]:
            if abs(block.bounding_box.y_min - row_y) < 15:  # same row threshold
                current_row.append(block)
            else:
                rows.append(sorted(current_row, key=lambda b: b.bounding_box.x_min))
                current_row = [block]
                row_y = block.bounding_box.y_min

        if current_row:
            rows.append(sorted(current_row, key=lambda b: b.bounding_box.x_min))

        logger.debug(f"Detected {len(rows)} table rows in line items.")
        return rows
