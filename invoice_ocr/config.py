"""
config.py - Application configuration and settings
"""

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Invoice OCR API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    OUTPUT_DIR: Path = BASE_DIR / "outputs"

    # Image preprocessing
    TARGET_WIDTH: int = 1024
    TARGET_HEIGHT: int = 1408
    DENOISE_STRENGTH: int = 10

    # PaddleOCR
    OCR_LANG: str = "en"
    OCR_USE_GPU: bool = False
    OCR_USE_ANGLE_CLS: bool = True
    OCR_DET_DB_THRESH: float = 0.3
    OCR_DET_DB_BOX_THRESH: float = 0.6
    OCR_REC_ALGORITHM: str = "CRNN"

    # Segmentation confidence threshold
    SEGMENTATION_CONFIDENCE: float = 0.5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Ensure directories exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
