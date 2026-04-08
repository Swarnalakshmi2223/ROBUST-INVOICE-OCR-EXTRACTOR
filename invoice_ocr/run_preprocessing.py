"""
run_preprocessing.py
Run full preprocessing on ALL images in the Invoice-Dataset folder.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

from modules.preprocessing import preprocess_dataset, INVOICE_DATASET_ROOT

if __name__ == "__main__":
    print("=" * 60)
    print("  Invoice Dataset - Full Preprocessing")
    print(f"  Dataset : {INVOICE_DATASET_ROOT}")
    print("=" * 60)

    results = preprocess_dataset(save=True, max_images=None)

    ok_count  = sum(1 for r in results if r["status"] == "ok")
    err_count = len(results) - ok_count

    print("\n" + "=" * 60)
    print(f"  DONE!")
    print(f"  Total images : {len(results)}")
    print(f"  Successful   : {ok_count}")
    print(f"  Errors       : {err_count}")
    print("=" * 60)

    if err_count:
        print("\nFailed images:")
        for r in results:
            if r["status"] != "ok":
                print(f"  [ERROR] {r['path'].name}  →  {r['status']}")

    print(f"\nProcessed images saved to:")
    print(f"  C:\\FINALYEAR_PROJECT\\preprocessed\\")
