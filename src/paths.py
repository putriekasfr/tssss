"""Lokasi proyek, dataset, dan konstanta yang dipakai semua notebook."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["WBC", "RBC", "Platelets"]          # urutan indeks YOLO 0, 1, 2
COCO_CAT_IDS = [1, 2, 3]                        # id kategori COCO = indeks YOLO + 1
SPLITS = ["train", "val", "test"]
SEED = 42

SPLIT_DIR = PROJECT_ROOT / "splits"
CONFIG_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
COCO_DIR = OUTPUT_DIR / "coco"                  # anotasi COCO hasil konversi (dibuat ulang, tidak di-commit)
PRED_DIR = OUTPUT_DIR / "preds"                 # prediksi tiap run dalam format COCO
LOG_PATH = PROJECT_ROOT / "experiments" / "log.csv"


def find_data_root() -> Path:
    """Cari folder TXL-PBC (berisi images/ dan labels/).

    Urutan: variabel lingkungan TXL_ROOT, folder data/ di proyek, lalu lokasi umum di Colab.
    """
    candidates = [
        os.environ.get("TXL_ROOT"),
        PROJECT_ROOT / "data" / "TXL-PBC_Dataset-master" / "TXL-PBC",
        PROJECT_ROOT / "data" / "TXL-PBC_Dataset" / "TXL-PBC",
        Path("/content/TXL-PBC_Dataset/TXL-PBC"),
        Path("/content/TXL-PBC_Dataset-master/TXL-PBC"),
    ]
    for c in candidates:
        if c and (Path(c) / "images" / "train").is_dir():
            return Path(c)
    raise FileNotFoundError(
        "Folder TXL-PBC tidak ditemukan. Jalankan sel unduh dataset atau set TXL_ROOT."
    )
