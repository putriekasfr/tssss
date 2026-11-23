"""YOLOv8 (ultralytics): data.yaml, pelatihan sesuai config, dan ekspor prediksi ke format COCO.

Dipakai oleh R01, R04, R07, R08 dan varian NMS R01-nms*. Angka yang dilaporkan TIDAK diambil dari validasi bawaan
ultralytics, tetapi dari prediksi yang diekspor lalu dinilai dengan pycocotools (src/evaluate.py),
sama seperti Faster R-CNN.
"""
import json
from pathlib import Path

import yaml

from .paths import CLASSES


def make_data_yaml(data_root, out_path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump({
        "path": str(Path(data_root).resolve()),
        "train": "images/train", "val": "images/val", "test": "images/test",
        "names": {i: c for i, c in enumerate(CLASSES)},
    }, sort_keys=False))
    return out_path


def _training_finished(run_dir, cfg) -> bool:
    """Pelatihan YOLO dianggap selesai bila semua epoch tercatat, ATAU berhenti dini (patience) dan ultralytics
    sudah 'strip optimizer' dari last.pt (tanda akhir pelatihan normal)."""
    import pandas as pd
    import torch
    run_dir = Path(run_dir)
    best, last, res = run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt", run_dir / "results.csv"
    if not (best.exists() and last.exists() and res.exists()):
        return False
    if len(pd.read_csv(res)) >= int(cfg["epochs"]):
        return True
    try:
        return torch.load(last, map_location="cpu", weights_only=False).get("optimizer") is None
    except Exception:
        return False


def train(cfg: dict, data_yaml, project_dir):
    """Latih YOLOv8 sesuai cfg. Bila last.pt sudah ada (sesi putus), pelatihan dilanjutkan."""
    from ultralytics import YOLO

    run_dir = Path(project_dir) / cfg["run_id"]
    last = run_dir / "weights" / "last.pt"
    done = run_dir / "DONE"                     # penanda dibuat setelah pelatihan selesai
    if done.exists():
        print("Run sudah selesai sebelumnya; hapus folder", run_dir, "bila ingin melatih ulang.")
        return None
    if _training_finished(run_dir, cfg):
        # pelatihan sudah tuntas (mis. proses berhenti saat validasi akhir) -> tidak perlu dilatih lagi
        print(f"Pelatihan {cfg['run_id']} sudah selesai; lanjut ke prediksi.")
        done.write_text("ok")
        return None
    if last.exists():
        print("Melanjutkan dari", last)
        res = YOLO(str(last)).train(resume=True, data=str(data_yaml))
        done.write_text("ok")
        return res

    a = cfg["augment"]
    model = YOLO(cfg["pretrained"])
    try:
        res = model.train(
            data=str(data_yaml), epochs=int(cfg["epochs"]), imgsz=int(cfg["imgsz"]), patience=int(cfg.get("patience", 100)),
            batch=int(cfg["batch"]), seed=int(cfg["seed"]), deterministic=True,
            optimizer=cfg["optimizer"], lr0=float(cfg["lr0"]), momentum=float(cfg["momentum"]),
            weight_decay=float(cfg["weight_decay"]), amp=bool(cfg["amp"]),
            fliplr=a["hflip"], flipud=a["vflip"], scale=a["scale"],
            hsv_h=a["hue"], hsv_s=a["saturation"], hsv_v=a["value"],
            mosaic=a["mosaic"], mixup=0.0, translate=0.0, degrees=0.0,
            project=str(project_dir), name=cfg["run_id"], exist_ok=True, plots=True,
            workers=2,          # Colab gratis: 2 CPU; lebih banyak worker hanya menambah RAM
            cache=False,        # citra tidak disimpan di RAM/disk cache (RAM sistem Colab gratis ±12,7 GB)
        )
    except Exception as e:
        # bila semua epoch & best.pt sudah tersimpan, kegagalan di validasi akhir ultralytics tidak membatalkan run
        if not _training_finished(run_dir, cfg):
            raise
        print(f"PERINGATAN: validasi akhir ultralytics gagal ({type(e).__name__}); bobot sudah lengkap, lanjut.")
        res = None
    done.write_text("ok")
    return res


def predict_coco(weights, img_dir, coco_json, cfg, chunk: int = 16) -> list:
    """Prediksi semua citra di coco_json -> list deteksi format COCO (piksel citra asli).

    Citra dikirim per kelompok `chunk` (bawaan 16). Mengirim seluruh daftar sekaligus membuat ultralytics
    memproses SEMUA citra dalam satu batch (252 citra validasi -> > 12 GB VRAM pada imgsz 832 -> CUDA OOM).
    `rect=False` mengunci letterbox persegi imgsz x imgsz sehingga hasil tidak bergantung pada ukuran kelompok
    (sama dengan cara prediksi baseline R01).
    """
    import gc
    import torch
    from ultralytics import YOLO

    model = YOLO(str(weights))
    coco = json.loads(Path(coco_json).read_text())
    name2id = {im["file_name"]: im["id"] for im in coco["images"]}
    paths = [str(Path(img_dir) / im["file_name"]) for im in coco["images"]]
    p = cfg["predict"]
    dets = []
    for i in range(0, len(paths), chunk):
        for r in model.predict(source=paths[i:i + chunk], imgsz=int(cfg["imgsz"]), conf=float(p["conf"]),
                               iou=float(p["iou_nms"]), max_det=int(p["max_det"]), rect=False,
                               stream=True, verbose=False):
            img_id = name2id[Path(r.path).name]
            for (x0, y0, x1, y1), s_, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                dets.append({"image_id": img_id, "category_id": int(c) + 1,
                             "bbox": [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)],
                             "score": round(float(s_), 5)})
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dets
