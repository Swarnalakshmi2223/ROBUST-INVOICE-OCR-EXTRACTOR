"""
invoice_schema.py - Pydantic models for invoice data structures
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class OCRTextBlock(BaseModel):
    text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bounding_box: BoundingBox


class InvoiceLineItem(BaseModel):
    description: Optional[str] = None
    quantity: Optional[str] = None
    unit_price: Optional[str] = None
    total: Optional[str] = None


class InvoiceFields(BaseModel):
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    vendor_name: Optional[str] = None
    vendor_address: Optional[str] = None
    vendor_email: Optional[str] = None
    vendor_phone: Optional[str] = None
    customer_name: Optional[str] = None
    customer_address: Optional[str] = None
    subtotal: Optional[str] = None
    tax: Optional[str] = None
    discount: Optional[str] = None
    total_amount: Optional[str] = None
    currency: Optional[str] = None
    payment_terms: Optional[str] = None
    line_items: List[InvoiceLineItem] = []


class InvoiceSegment(BaseModel):
    region: str  # e.g., "header", "body", "footer", "line_items"
    bounding_box: Optional[BoundingBox] = None
    raw_text_blocks: List[OCRTextBlock] = []


class InvoiceResult(BaseModel):
    file_name: str
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    image_width: int
    image_height: int
    total_text_blocks: int
    segments: List[InvoiceSegment] = []
    extracted_fields: InvoiceFields
    raw_text: str
    confidence_avg: float
    processing_time_seconds: float
    status: str = "success"
    errors: List[str] = []


class OCRResponse(BaseModel):
    success: bool
    message: str
    data: Optional[InvoiceResult] = None


class BatchOCRResponse(BaseModel):
    success: bool
    message: str
    total_files: int
    processed: int
    failed: int
    results: List[OCRResponse] = []
