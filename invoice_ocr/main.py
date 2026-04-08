"""
main.py - FastAPI application for Invoice OCR
"""

import time
import logging
import uuid
from pathlib import Path
from typing import List

import aiofiles
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from models.invoice_schema import OCRResponse, BatchOCRResponse, InvoiceResult
from modules.preprocessing import ImagePreprocessor
from modules.ocr import InvoiceOCR
from modules.segmentation import InvoiceSegmenter
from modules.postprocessing import PostProcessor
from modules.output import OutputFormatter

# ─── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ─── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Invoice OCR API powered by PaddleOCR. "
        "Upload invoice images to extract structured JSON data."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Module Instances ───────────────────────────────────────────────────────
preprocessor = ImagePreprocessor()
ocr_engine = InvoiceOCR()
formatter = OutputFormatter(output_dir=settings.OUTPUT_DIR)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ─── Helpers ────────────────────────────────────────────────────────────────

def _validate_file(filename: str) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )


async def _save_upload(file: UploadFile) -> Path:
    """Save uploaded file to the uploads directory."""
    uid = uuid.uuid4().hex[:8]
    safe_name = f"{uid}_{file.filename}"
    dest = settings.UPLOAD_DIR / safe_name
    async with aiofiles.open(dest, "wb") as f:
        content = await file.read()
        await f.write(content)
    return dest


def _process_image(image_path: Path, original_filename: str) -> InvoiceResult:
    """Core processing pipeline — runs synchronously."""
    start = time.perf_counter()
    errors: List[str] = []

    # 1. Preprocess
    processed = preprocessor.preprocess(image_path)
    ocr_ready = processed["ocr_ready"]
    h, w = processed["processed_shape"][:2]

    # 2. OCR
    blocks = ocr_engine.run(ocr_ready)
    raw_text = InvoiceOCR.get_raw_text(blocks)
    confidence_avg = InvoiceOCR.average_confidence(blocks)

    # 3. Segment
    segmenter = InvoiceSegmenter(image_height=h, image_width=w)
    segments = segmenter.segment(blocks)

    # 4. Post-process
    post = PostProcessor(segmenter=segmenter)
    extracted_fields = post.extract_fields(segments)

    elapsed = round(time.perf_counter() - start, 3)

    return InvoiceResult(
        file_name=original_filename,
        image_width=w,
        image_height=h,
        total_text_blocks=len(blocks),
        segments=segments,
        extracted_fields=extracted_fields,
        raw_text=raw_text,
        confidence_avg=confidence_avg,
        processing_time_seconds=elapsed,
        status="success",
        errors=errors,
    )


# ─── Routes ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {"message": f"Welcome to {settings.APP_NAME} v{settings.APP_VERSION}"}


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok", "version": settings.APP_VERSION}


@app.post("/ocr/invoice", response_model=OCRResponse, tags=["OCR"])
async def process_invoice(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Invoice image to process"),
    save_output: bool = True,
):
    """
    Process a single invoice image and return extracted JSON data.
    - Preprocessing → OCR → Segmentation → Post-processing → JSON
    """
    _validate_file(file.filename)
    image_path = await _save_upload(file)

    try:
        result = _process_image(image_path, file.filename)
    except Exception as e:
        logger.exception("Processing failed.")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up upload in background
        background_tasks.add_task(image_path.unlink, missing_ok=True)

    if save_output:
        formatter.save(result)

    return formatter.build_response(result)


@app.post("/ocr/batch", response_model=BatchOCRResponse, tags=["OCR"])
async def process_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Multiple invoice images"),
    save_output: bool = True,
):
    """
    Process multiple invoice images in a single request.
    Returns aggregated results with per-file status.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    responses: List[OCRResponse] = []
    processed_count = 0
    failed_count = 0

    for file in files:
        try:
            _validate_file(file.filename)
            image_path = await _save_upload(file)
            result = _process_image(image_path, file.filename)
            background_tasks.add_task(image_path.unlink, missing_ok=True)
            if save_output:
                formatter.save(result)
            responses.append(formatter.build_response(result))
            processed_count += 1
        except Exception as e:
            logger.error(f"Failed to process {file.filename}: {e}")
            responses.append(formatter.build_error_response(
                f"Failed to process '{file.filename}': {str(e)}"
            ))
            failed_count += 1

    return BatchOCRResponse(
        success=failed_count == 0,
        message=f"Processed {processed_count}/{len(files)} files.",
        total_files=len(files),
        processed=processed_count,
        failed=failed_count,
        results=responses,
    )


@app.get("/outputs", tags=["Results"])
async def list_outputs():
    """List all saved OCR JSON output files."""
    files = sorted(settings.OUTPUT_DIR.glob("*.json"))
    return {
        "count": len(files),
        "files": [f.name for f in files],
    }


@app.get("/outputs/{filename}", tags=["Results"])
async def get_output(filename: str):
    """Download a specific JSON output file."""
    path = settings.OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path, media_type="application/json", filename=filename)


# ─── Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
