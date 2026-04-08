"""
postprocessing.py - Post-processing and field extraction from OCR segments
"""

import re
import logging
from typing import List, Optional

from models.invoice_schema import (
    InvoiceSegment, InvoiceFields, InvoiceLineItem, OCRTextBlock
)
from modules.segmentation import InvoiceSegmenter

logger = logging.getLogger(__name__)


# ─── Regex Patterns ──────────────────────────────────────────────────────────

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
        r"\b((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2},?\s+\d{4})\b",
    ],
    "due_date": [
        r"(?:due\s*(?:date|by|on)[:\s]*)([\d\/\-\.]+)",
        r"(?:payment\s*due[:\s]*)([\d\/\-\.]+)",
    ],
    "total_amount": [
        r"(?:total\s*(?:amount|due|payable)?[:\s]*)[€$£₹]?\s*([\d,]+\.?\d*)",
        r"(?:amount\s*(?:due|payable)[:\s]*)[€$£₹]?\s*([\d,]+\.?\d*)",
        r"(?:grand\s*total[:\s]*)[€$£₹]?\s*([\d,]+\.?\d*)",
    ],
    "subtotal": [
        r"(?:subtotal|sub\s*total)[:\s]*[€$£₹]?\s*([\d,]+\.?\d*)",
    ],
    "tax": [
        r"(?:tax|vat|gst|cgst|sgst|igst)[:\s%0-9]*[€$£₹]?\s*([\d,]+\.?\d*)",
    ],
    "discount": [
        r"(?:discount)[:\s%0-9]*[€$£₹]?\s*([\d,]+\.?\d*)",
    ],
    "email": [
        r"[a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9.\-]+",
    ],
    "phone": [
        r"(?:phone|tel|mob|contact)[:\s]*([+\d\s\-().]{7,20})",
        r"\b(\+?\d[\d\s\-().]{6,18}\d)\b",
    ],
    "currency": [
        r"([€$£₹¥])",
        r"\b(USD|EUR|GBP|INR|JPY|CAD|AUD)\b",
    ],
    "payment_terms": [
        r"(?:payment\s*terms?|terms?)[:\s]*(net\s*\d+|due\s*on\s*receipt|immediate|[\w\s]{3,30})",
    ],
}


class PostProcessor:
    """
    Extracts structured invoice fields from OCR segments
    using regex patterns and heuristics.
    """

    def __init__(self, segmenter: Optional[InvoiceSegmenter] = None):
        self.segmenter = segmenter

    def extract_fields(self, segments: List[InvoiceSegment]) -> InvoiceFields:
        """Main entry: extract all invoice fields from all segments."""
        segment_map = {s.region: s for s in segments}
        full_text = self._get_full_text(segments)
        full_text_lower = full_text.lower()

        fields = InvoiceFields()

        # ── Invoice number ────────────────────────────────────────────────
        fields.invoice_number = self._find_first(
            full_text_lower, PATTERNS["invoice_number"]
        )

        # ── Dates ─────────────────────────────────────────────────────────
        dates = self._find_all(full_text, PATTERNS["date"])
        if dates:
            fields.invoice_date = dates[0]
        if len(dates) > 1:
            fields.due_date = dates[1]
        # Override with explicit due-date pattern
        due = self._find_first(full_text_lower, PATTERNS["due_date"])
        if due:
            fields.due_date = due

        # ── Financials ────────────────────────────────────────────────────
        fields.total_amount = self._find_first(full_text_lower, PATTERNS["total_amount"])
        fields.subtotal = self._find_first(full_text_lower, PATTERNS["subtotal"])
        fields.tax = self._find_first(full_text_lower, PATTERNS["tax"])
        fields.discount = self._find_first(full_text_lower, PATTERNS["discount"])
        fields.currency = self._find_first(full_text, PATTERNS["currency"])

        # ── Contact ───────────────────────────────────────────────────────
        fields.vendor_email = self._find_first(full_text, PATTERNS["email"])
        fields.vendor_phone = self._find_first(full_text, PATTERNS["phone"])
        fields.payment_terms = self._find_first(full_text_lower, PATTERNS["payment_terms"])

        # ── Vendor / Customer (header and bill_to region heuristics) ──────
        if "header" in segment_map:
            header_text = self._get_segment_text(segment_map["header"])
            lines = [l.strip() for l in header_text.split("\n") if l.strip()]
            if lines:
                fields.vendor_name = lines[0]
            if len(lines) > 1:
                fields.vendor_address = " ".join(lines[1:3])

        if "bill_to" in segment_map:
            bill_text = self._get_segment_text(segment_map["bill_to"])
            lines = [l.strip() for l in bill_text.split("\n") if l.strip()]
            for i, line in enumerate(lines):
                low = line.lower()
                if any(k in low for k in ["bill to", "ship to", "customer", "client"]):
                    if i + 1 < len(lines):
                        fields.customer_name = lines[i + 1]
                    if i + 2 < len(lines):
                        fields.customer_address = " ".join(lines[i + 2: i + 4])
                    break

        # ── Line Items ────────────────────────────────────────────────────
        if "line_items" in segment_map and self.segmenter:
            li_blocks = segment_map["line_items"].raw_text_blocks
            rows = self.segmenter.detect_table_rows(li_blocks)
            fields.line_items = self._parse_line_items(rows)

        logger.info("Post-processing complete.")
        return fields

    def _parse_line_items(
        self, rows: List[List[OCRTextBlock]]
    ) -> List[InvoiceLineItem]:
        """Parse detected table rows into InvoiceLineItem objects."""
        items: List[InvoiceLineItem] = []
        # Skip header row if first row looks like labels
        start_idx = 0
        if rows:
            header_texts = [b.text.lower() for b in rows[0]]
            header_kws = {"description", "item", "qty", "quantity", "price", "amount", "total"}
            if any(kw in " ".join(header_texts) for kw in header_kws):
                start_idx = 1

        for row in rows[start_idx:]:
            texts = [b.text for b in row]
            if not texts:
                continue
            item = InvoiceLineItem()
            if len(texts) >= 1:
                item.description = texts[0]
            if len(texts) >= 2:
                item.quantity = self._clean_number(texts[1])
            if len(texts) >= 3:
                item.unit_price = self._clean_number(texts[2])
            if len(texts) >= 4:
                item.total = self._clean_number(texts[3])
            elif len(texts) == 3:
                # 3-column: desc, qty, total
                item.total = self._clean_number(texts[2])
                item.unit_price = None
            items.append(item)
        return items

    # ─── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _find_first(text: str, patterns: List[str]) -> Optional[str]:
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1).strip() if m.lastindex else m.group(0).strip()
        return None

    @staticmethod
    def _find_all(text: str, patterns: List[str]) -> List[str]:
        results = []
        for pat in patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                val = m.group(1).strip() if m.lastindex else m.group(0).strip()
                if val and val not in results:
                    results.append(val)
        return results

    @staticmethod
    def _get_full_text(segments: List[InvoiceSegment]) -> str:
        return "\n".join(
            b.text
            for s in segments
            for b in s.raw_text_blocks
            if b.text
        )

    @staticmethod
    def _get_segment_text(segment: InvoiceSegment) -> str:
        return "\n".join(b.text for b in segment.raw_text_blocks if b.text)

    @staticmethod
    def _clean_number(text: str) -> Optional[str]:
        clean = re.sub(r"[^\d.,]", "", text)
        return clean if clean else None
