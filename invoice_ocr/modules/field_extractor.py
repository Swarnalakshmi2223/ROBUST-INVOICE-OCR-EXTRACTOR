"""
field_extractor.py - Extract key invoice fields from OCR text.

Extracts:
  - invoice_number : invoice/bill number
  - date           : invoice date
  - total          : total amount due
  - vendor         : vendor / company name

Strategy:
  1. Keyword-anchored regex  — look near labels like "Invoice No:", "Date:", "Total:"
  2. Fallback general regex  — find dates/amounts anywhere in the text
  3. Vendor heuristic        — first significant line of the document (header)

Output (JSON):
  {
    "invoice_number": "INV-2024-001",
    "date":           "15/03/2024",
    "total":          "1,250.00",
    "vendor":         "Acme Corp Ltd"
  }

Usage (as module):
    from modules.field_extractor import InvoiceFieldExtractor

    extractor = InvoiceFieldExtractor()
    result    = extractor.extract(ocr_text)          # from plain text
    result    = extractor.extract_from_boxes(boxes)  # from OCR box list
    print(result)   # dict
    print(extractor.to_json(result))  # JSON string

Standalone:
    py modules/field_extractor.py <ocr_result.json>
    py modules/field_extractor.py --text "Invoice No: 123 Date: 01/01/2024 ..."
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ─── Output schema ────────────────────────────────────────────────────────────

EMPTY_RESULT: Dict[str, str] = {
    "invoice_number": "",
    "date":           "",
    "total":          "",
    "vendor":         "",
}


# ─── Regex patterns ───────────────────────────────────────────────────────────

# ── Invoice number ─────────────────────────────────────────────────────────
_INV_NUM_PATTERNS = [
    # Keyword-anchored (highest priority)
    r"(?:invoice\s*(?:no|number|num|#|id)[:\s#\-]*)([\w\-/]{3,20})",
    r"(?:inv[:\s#\-]*no[:\s#\-]*)([\w\-/]{3,20})",
    r"(?:inv[oice]*[:\s#\-]+)([\w\-/]{3,20})",
    r"(?:bill\s*(?:no|number)[:\s#\-]*)([\w\-/]{3,20})",
    r"(?:receipt\s*(?:no|number|#)[:\s#\-]*)([\w\-/]{3,20})",
    # Fallback: common formatted invoice numbers
    r"\b(INV[-/]\w{3,15})\b",
    r"\b([A-Z]{2,5}[-/]\d{4,10})\b",
    r"\b(\d{4,10})\b",   # bare number last resort
]

# ── Date ───────────────────────────────────────────────────────────────────
_DATE_PATTERNS = [
    # Keyword-anchored
    r"(?:invoice\s*date|date\s*of\s*invoice|issued?(?:\s*on)?|bill\s*date)[:\s]*(\d{1,2}[\-/\.]\d{1,2}[\-/\.]\d{2,4})",
    r"(?:invoice\s*date|date\s*of\s*invoice|issued?(?:\s*on)?|bill\s*date)[:\s]*(\d{4}[\-/\.]\d{1,2}[\-/\.]\d{1,2})",
    r"(?:invoice\s*date|date\s*of\s*invoice|issued?(?:\s*on)?)[:\s]*(\d{1,2}\s+\w{3,9}\s+\d{4})",
    r"(?:date)[:\s]*(\d{1,2}[\-/\.]\d{1,2}[\-/\.]\d{2,4})",
    # General date formats (fallback)
    r"\b(\d{1,2}[\-/\.]\d{1,2}[\-/\.]\d{4})\b",
    r"\b(\d{4}[\-/\.]\d{1,2}[\-/\.]\d{1,2})\b",
    r"\b(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4})\b",
    r"\b((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2},?\s+\d{4})\b",
]

# ── Total amount ────────────────────────────────────────────────────────────
_TOTAL_PATTERNS = [
    # Keyword-anchored (highest priority first)
    r"(?:grand\s*total|total\s*amount\s*due|total\s*due|amount\s*due|amount\s*payable)[:\s]*[€$£₹¥]?\s*([\d,]+\.?\d*)",
    r"(?:total\s*amount|total\s*payable|total)[:\s]*[€$£₹¥]?\s*([\d,]+\.?\d{0,2})",
    r"(?:net\s*amount|net\s*total)[:\s]*[€$£₹¥]?\s*([\d,]+\.?\d{0,2})",
    r"(?:balance\s*due|balance)[:\s]*[€$£₹¥]?\s*([\d,]+\.?\d{0,2})",
    # With currency symbol in front of number
    r"[€$£₹¥]\s*([\d,]{1,12}\.?\d{0,2})\b",
]

# ── Vendor name ─────────────────────────────────────────────────────────────
_VENDOR_PATTERNS = [
    r"(?:from|vendor|supplier|company|billed?\s*by|service\s*provider)[:\s]+([A-Za-z0-9&,.\-\s]{3,60})",
    r"(?:sold\s*by|merchant)[:\s]+([A-Za-z0-9&,.\-\s]{3,60})",
]


# ─── Helper: single-match search ─────────────────────────────────────────────

def _find_first(text: str, patterns: List[str]) -> Optional[str]:
    """Return the first regex match (group 1 if present, else group 0)."""
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = (m.group(1) if m.lastindex else m.group(0)).strip()
            if val:
                return val
    return None


def _clean_vendor(raw: str) -> str:
    """Strip trailing noise from a vendor string."""
    # Remove anything after a newline, digit sequence, or common trailing chars
    raw = re.split(r"[\n\r|]", raw)[0]
    raw = re.sub(r"\s{2,}", " ", raw).strip()
    # Remove trailing punctuation
    raw = raw.rstrip(".,;:-")
    return raw


# ─── Vendor heuristic ─────────────────────────────────────────────────────────

_SKIP_LINES = re.compile(
    r"""^(
        invoice|bill|receipt|order|quotation|proforma|tax\s*invoice|
        date|no\.|number|\d+|page|http|www\.|ref
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)

_MIN_VENDOR_LEN = 3


def _extract_vendor_from_lines(lines: List[str]) -> Optional[str]:
    """
    Heuristic: vendor name is usually the first 'meaningful' line of the doc
    — not a label, not a number, not a URL.
    """
    for line in lines[:15]:  # look only in the first 15 lines
        line = line.strip()
        if len(line) < _MIN_VENDOR_LEN:
            continue
        if _SKIP_LINES.match(line):
            continue
        if re.match(r"^[\d\s\-/.,]+$", line):
            continue   # all numeric
        if "@" in line or "://" in line:
            continue   # email / URL
        return _clean_vendor(line)
    return None


# ─── Main extractor class ─────────────────────────────────────────────────────

class InvoiceFieldExtractor:
    """
    Extracts four key invoice fields from OCR text using regex and keyword matching.

    Returns:
        {
            "invoice_number": str | "",
            "date":           str | "",
            "total":          str | "",
            "vendor":         str | "",
        }
    """

    def extract(self, text: str) -> Dict[str, str]:
        """
        Extract fields from a plain text string.

        Args:
            text: Full OCR text (lines joined by \\n).

        Returns:
            Dict matching the invoice JSON schema.
        """
        result = dict(EMPTY_RESULT)
        lines  = [l.strip() for l in text.splitlines() if l.strip()]

        # ── Invoice number ────────────────────────────────────────────────
        result["invoice_number"] = _find_first(text, _INV_NUM_PATTERNS) or ""

        # ── Date ──────────────────────────────────────────────────────────
        result["date"] = _find_first(text, _DATE_PATTERNS) or ""

        # ── Total amount ──────────────────────────────────────────────────
        result["total"] = _find_first(text, _TOTAL_PATTERNS) or ""

        # ── Vendor name ───────────────────────────────────────────────────
        # Try keyword-anchored patterns first
        vendor = _find_first(text, _VENDOR_PATTERNS)
        if vendor:
            result["vendor"] = _clean_vendor(vendor)
        else:
            # Fallback: first meaningful header line
            result["vendor"] = _extract_vendor_from_lines(lines) or ""

        logger.info(
            f"Extracted → invoice_number='{result['invoice_number']}' | "
            f"date='{result['date']}' | total='{result['total']}' | "
            f"vendor='{result['vendor']}'"
        )
        return result

    def extract_from_boxes(self, boxes: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Extract fields directly from OCR box list (from InvoiceOCR.run()).

        Boxes are sorted top-to-bottom to preserve reading order before
        joining into text.

        Args:
            boxes: List of OCR result dicts with 'text', 'y_min', 'x_min'.

        Returns:
            Same dict as extract().
        """
        # Sort reading order: top → bottom, left → right
        sorted_boxes = sorted(
            boxes,
            key=lambda b: (round(b.get("y_min", 0) / 20) * 20, b.get("x_min", 0))
        )
        text = "\n".join(b.get("text", "") for b in sorted_boxes if b.get("text", "").strip())
        return self.extract(text)

    @staticmethod
    def to_json(result: Dict[str, str], indent: int = 2) -> str:
        """Serialize the result dict to a JSON string."""
        return json.dumps(result, indent=indent, ensure_ascii=False)

    @staticmethod
    def to_json_file(result: Dict[str, str], path: Path) -> None:
        """Save the result dict as a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Field extraction result saved → {path}")


# ─── Standalone entry-point ───────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    extractor = InvoiceFieldExtractor()

    # ── Mode 1: --text "raw text" ─────────────────────────────────────────
    if len(sys.argv) >= 2 and sys.argv[1] == "--text":
        raw_text = " ".join(sys.argv[2:])
        result = extractor.extract(raw_text)

    # ── Mode 2: ocr_result.json ───────────────────────────────────────────
    elif len(sys.argv) >= 2:
        json_path = Path(sys.argv[1])
        if not json_path.exists():
            print(f"[ERROR] File not found: {json_path}")
            sys.exit(1)

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        # Accept either a list of OCR boxes OR a plain text string
        if isinstance(data, list):
            result = extractor.extract_from_boxes(data)
        elif isinstance(data, str):
            result = extractor.extract(data)
        else:
            print("[ERROR] JSON must be a list of OCR boxes or a plain text string.")
            sys.exit(1)

    else:
        # Built-in demo
        demo_text = """
        Acme Corporation Ltd
        123 Business Park, Chennai - 600001
        Email: billing@acme.com

        INVOICE

        Invoice No: INV-2024-00842
        Date: 15/03/2024
        Due Date: 30/03/2024

        Bill To:
        John Doe
        456 Client Street, Mumbai

        Description         Qty   Unit Price   Total
        Web Development      1     50,000.00   50,000.00
        Server Hosting       12       500.00    6,000.00
        Support Package      1      5,000.00    5,000.00

        Subtotal                              61,000.00
        GST (18%)                             10,980.00
        Grand Total                    ₹    71,980.00
        """
        print("[INFO] No arguments provided — running with built-in demo text.\n")
        result = extractor.extract(demo_text)

    # ── Print result ──────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print( "  Invoice Field Extraction Result")
    print(f"{'='*55}")
    print(extractor.to_json(result))
    print(f"{'='*55}\n")

    # ── Save JSON if input was a file ─────────────────────────────────────
    if len(sys.argv) >= 2 and not sys.argv[1].startswith("--"):
        out_path = Path(sys.argv[1]).parent / f"{Path(sys.argv[1]).stem}_fields.json"
        extractor.to_json_file(result, out_path)
        print(f"  Saved → {out_path}")
