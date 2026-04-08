"""
output.py - JSON output formatting and file persistence
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from models.invoice_schema import InvoiceResult, OCRResponse

logger = logging.getLogger(__name__)


class OutputFormatter:
    """
    Converts InvoiceResult objects to clean JSON,
    and optionally saves them to disk.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir

    def to_dict(self, result: InvoiceResult) -> dict:
        """Serialize InvoiceResult to a plain Python dict."""
        return result.model_dump(mode="json")

    def to_json(self, result: InvoiceResult, indent: int = 2) -> str:
        """Serialize InvoiceResult to a pretty-printed JSON string."""
        data = self.to_dict(result)
        return json.dumps(data, indent=indent, ensure_ascii=False, default=str)

    def save(self, result: InvoiceResult, filename: Optional[str] = None) -> Path:
        """
        Save the JSON result to the output directory.

        Args:
            result:   Processed invoice result.
            filename: Optional custom filename (without extension).
                      Defaults to '<original_file_stem>_<timestamp>.json'.

        Returns:
            Path to the saved JSON file.
        """
        if self.output_dir is None:
            raise ValueError("output_dir must be set to save files.")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            stem = Path(result.file_name).stem
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"{stem}_{ts}"

        out_path = self.output_dir / f"{filename}.json"
        json_str = self.to_json(result)
        out_path.write_text(json_str, encoding="utf-8")
        logger.info(f"Result saved → {out_path}")
        return out_path

    def build_response(
        self,
        result: InvoiceResult,
        success: bool = True,
        message: str = "Invoice processed successfully.",
    ) -> OCRResponse:
        """Wrap an InvoiceResult in a standard OCRResponse envelope."""
        return OCRResponse(
            success=success,
            message=message,
            data=result,
        )

    def build_error_response(self, message: str) -> OCRResponse:
        """Build a failure OCRResponse with no data."""
        return OCRResponse(
            success=False,
            message=message,
            data=None,
        )

    @staticmethod
    def load_json(path: Path) -> dict:
        """Load a previously saved result JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def summarize(self, result: InvoiceResult) -> dict:
        """Return a compact summary dict of the most important fields."""
        fields = result.extracted_fields
        return {
            "file_name": result.file_name,
            "processed_at": result.processed_at.isoformat(),
            "invoice_number": fields.invoice_number,
            "invoice_date": fields.invoice_date,
            "due_date": fields.due_date,
            "vendor_name": fields.vendor_name,
            "customer_name": fields.customer_name,
            "total_amount": fields.total_amount,
            "currency": fields.currency,
            "line_items_count": len(fields.line_items),
            "confidence_avg": result.confidence_avg,
            "processing_time_seconds": result.processing_time_seconds,
            "status": result.status,
        }
