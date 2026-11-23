"""Efisiensi: jumlah parameter dan latensi per citra (batch 1) sesuai kisi-kisi 3.3.

Latensi diukur ujung ke ujung (baca tensor citra -> pra-proses -> model -> NMS/pasca-proses) agar
YOLOv8 dan Faster R-CNN dibandingkan setara. Warm-up 10 citra, lalu rerata >= 50 citra, dengan
torch.cuda.synchronize() sebelum dan sesudah tiap citra.
"""
import time
from pathlib import Path

import numpy as np


def _timeit(fn, items, warmup=10, n=50):
    import torch
    for x in items[:warmup]:
        fn(x)
    torch.cuda.synchronize()
    ts = []
    for x in items[warmup:warmup + n]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(x)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return dict(latensi_ms_mean=float(np.mean(ts)), latensi_ms_std=float(np.std(ts)),
                latensi_ms_median=float(np.median(ts)), n_citra=len(ts), warmup=warmup)


def yolo_efficiency(weights, image_paths, cfg, warmup=10, n=50) -> dict:
    from ultralytics import YOLO
    model = YOLO(str(weights))
    p = cfg["predict"]
    fn = lambda path: model.predict(path, imgsz=int(cfg["imgsz"]), conf=float(p["conf"]), iou=float(p["iou_nms"]), rect=False,
                                    max_det=int(p["max_det"]), verbose=False)
    out = _timeit(fn, [str(x) for x in image_paths], warmup, n)
    out["parameter_juta"] = sum(q.numel() for q in model.model.parameters()) / 1e6
    return out


def frcnn_efficiency(model, image_paths, warmup=10, n=50) -> dict:
    import torch
    from PIL import Image
    from torchvision.transforms import functional as TF
    model.eval()

    @torch.no_grad()
    def fn(path):
        img = TF.to_tensor(Image.open(path).convert("RGB")).to("cuda")
        return model([img])

    out = _timeit(fn, [str(x) for x in image_paths], warmup, n)
    out["parameter_juta"] = sum(q.numel() for q in model.parameters()) / 1e6
    return out
