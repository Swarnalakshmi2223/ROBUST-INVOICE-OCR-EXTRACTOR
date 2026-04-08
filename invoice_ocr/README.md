# Invoice OCR API

A FastAPI + PaddleOCR based invoice OCR system that extracts structured data from invoice images.

## Project Structure

```
invoice_ocr/
├── main.py                     # FastAPI app & API routes
├── config.py                   # Settings (pydantic-settings)
├── requirements.txt            # Python dependencies
├── .env                        # Optional env overrides
├── uploads/                    # Temp uploaded files
├── outputs/                    # Saved JSON results
├── models/
│   └── invoice_schema.py       # Pydantic data models
└── modules/
    ├── preprocessing.py        # Image preprocessing pipeline
    ├── ocr.py                  # PaddleOCR wrapper
    ├── segmentation.py         # Invoice region segmentation
    ├── postprocessing.py       # Field extraction (regex)
    └── output.py               # JSON formatter & file saver
```

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the API server
python main.py
# or
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Welcome message |
| GET | `/health` | Health check |
| POST | `/ocr/invoice` | Process single invoice image |
| POST | `/ocr/batch` | Process multiple invoice images |
| GET | `/outputs` | List saved JSON results |
| GET | `/outputs/{filename}` | Download a JSON result |

## Interactive Docs

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Pipeline Flow

```
Uploaded Image
     │
     ▼
 Preprocessing          ← resize, denoise, deskew, binarize, CLAHE
     │
     ▼
 PaddleOCR              ← detect & recognize text blocks + bounding boxes
     │
     ▼
 Segmentation           ← divide into header / bill_to / line_items / summary / footer
     │
     ▼
 Post-processing        ← regex extraction of invoice fields & line items
     │
     ▼
 JSON Output            ← structured InvoiceResult → saved to outputs/
```

## Supported Image Formats

`.jpg` `.jpeg` `.png` `.bmp` `.tiff` `.tif` `.webp`

## Environment Variables (`.env`)

```env
DEBUG=false
OCR_LANG=en
OCR_USE_GPU=false
TARGET_WIDTH=1024
TARGET_HEIGHT=1408
```
