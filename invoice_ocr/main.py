"""
main.py - FastAPI Application for Invoice OCR

Provides:
 - POST /upload endpoint to process invoice images.
 - Runs complete pipeline: Preprocessing → OCR → Post-processing → Field Extraction
 - Returns JSON matching the requested schema.
"""

import time
import uuid
import logging
from pathlib import Path

import aiofiles
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import cv2

# Project modules
from modules.preprocessing import preprocess
from modules.ocr import InvoiceOCR
from modules.postprocessing import OCRPostProcessor
from modules.field_extractor import InvoiceFieldExtractor

# ─── Setup ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Invoice OCR API", description="Extract invoice fields from images")

# Singleton initialisation
ocr_engine = InvoiceOCR()
post_processor = OCRPostProcessor()
field_extractor = InvoiceFieldExtractor()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ─── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"message": "Invoice OCR API is running. Use POST /upload to process an image."}


@app.post("/upload")
async def process_invoice(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Process an uploaded invoice image.
    Pipeline:
      1. Save uploaded file temporarily
      2. Preprocessing (grayscale, blur, clahe, deskew)
      3. OCR (PaddleOCR bounding boxes + text)
      4. Post-processing (Confidence filter, NMS overlap removal)
      5. Field Extraction (Regex/Keyword matching)
    Returns JSON with invoice_number, date, total, and vendor.
    """
    # 1. Validate and save file
    ext = Path(file.filename).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}:
        raise HTTPException(status_code=400, detail="Unsupported file format.")

    file_id = uuid.uuid4().hex[:8]
    temp_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    
    async with aiofiles.open(temp_path, "wb") as buffer:
        content = await file.read()
        await buffer.write(content)
        
    # Clean up file in background after response is sent
    background_tasks.add_task(temp_path.unlink, missing_ok=True)

    start_time = time.perf_counter()

    try:
        # 2. Preprocessing
        logger.info(f"[{file_id}] Starting preprocessing...")
        preprocessed_img = preprocess(str(temp_path))

        # 3. OCR (PaddleOCR)
        logger.info(f"[{file_id}] Running OCR...")
        raw_boxes = ocr_engine.run(preprocessed_img)

        if not raw_boxes:
            logger.warning(f"[{file_id}] OCR returned no text.")
            return JSONResponse(content={
                "invoice_number": "", "date": "", "total": "", "vendor": ""
            })

        # 4. Post-processing (clean boxes: conf > 0.5, remove overlap)
        logger.info(f"[{file_id}] Post-processing text blocks...")
        clean_boxes = post_processor.clean_boxes(raw_boxes)

        # 5. Field extraction
        logger.info(f"[{file_id}] Extracting invoice fields...")
        extracted_data = field_extractor.extract_from_boxes(clean_boxes)

        elapsed = round(time.perf_counter() - start_time, 2)
        logger.info(f"[{file_id}] Processing complete in {elapsed}s")

        return JSONResponse(content=extracted_data)

    except Exception as e:
        logger.exception(f"[{file_id}] Error processing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # Run server locally
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
