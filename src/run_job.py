"""Menjalankan SATU pekerjaan GPU di proses Python terpisah.

Dipanggil oleh notebook M3 lewat `src.pipeline.run_isolated(...)`. Karena proses ini berakhir setelah
pekerjaan selesai, seluruh memori GPU (VRAM) dan RAM yang dipakai pelatihan pasti dilepas sebelum
pekerjaan berikutnya dimulai — tidak menumpuk walaupun banyak run dijalankan dalam satu sesi Colab.

Pekerjaan:
    R04, R06, R07 ... R10     latih + evaluasi validasi (run_yolo / run_frcnn)
    nms:R01[:0.45,0.60]       ablasi ambang IoU NMS pada bobot R01 (tanpa latih ulang)
    test:R04                  prediksi set uji satu run final
Hasil (metrik) dicetak pada baris yang diawali '__HASIL__' dalam format JSON.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from src import pipeline as P                                  # noqa: E402
from src.exp_log import load_config                             # noqa: E402
from src.paths import PROJECT_ROOT, find_data_root              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    a = ap.parse_args()
    data_root, runs_dir = find_data_root(), PROJECT_ROOT / "runs"
    job = a.job
    if job == "selftest":                       # cek cepat mekanisme proses terpisah (tanpa GPU)
        print("baris log biasa")
        print("  12%|██        | 7/56 [00:01<00:07, 6.5it/s]\r 50%|█████     | 28/56 [00:04<00:04, 6.6it/s]")
        out = {"ok": 1}
        print("__HASIL__" + json.dumps(out), flush=True)
        return
    if job.startswith("nms:"):
        parts = job.split(":")
        ious = [float(x) for x in parts[2].split(",")] if len(parts) > 2 else P.NMS_IOUS
        res = P.run_nms_ablation(parts[1], data_root, runs_dir, ious=ious)
        out = {k: {c: float(v) for c, v in row.items() if isinstance(v, (int, float))} for k, row in res.iterrows()}
    elif job.startswith("test:"):
        dets = P.predict_split(job.split(":")[1], "test", data_root, runs_dir)
        out = {"deteksi": len(dets)}
    else:
        kind = "yolo" if load_config(job)["model"].startswith("yolo") else "frcnn"
        out = (P.run_yolo if kind == "yolo" else P.run_frcnn)(job, data_root, runs_dir)
    print("__HASIL__" + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
