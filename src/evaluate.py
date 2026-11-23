"""Evaluasi terpadu untuk SEMUA model (satu alat yang sama, sesuai kisi-kisi 3.3).

Masukan setiap model: list deteksi format COCO
    {"image_id": int, "category_id": 1..3, "bbox": [x, y, w, h] (piksel citra asli), "score": float}
yang disimpan di outputs/preds/<RUN>_<split>.json.

- Deteksi : pycocotools COCOeval -> mAP@0.5, mAP@[0.5:0.95], AP per kelas, AP_small.
- Hitung  : jumlah deteksi dengan skor >= tau per kelas per citra vs jumlah ground truth
            -> MAE dan MAPE per kelas. tau dipilih per kelas di set VALIDASI.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import CLASSES, COCO_CAT_IDS

TAU_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)


def load_json(p):
    return json.loads(Path(p).read_text())


# ----------------------------------------------------------------------------- deteksi
def coco_metrics(gt_json: Path, dets: list, verbose: bool = True) -> dict:
    """mAP dan AP per kelas dengan pycocotools (konvensi COCO, maxDets=100)."""
    import contextlib
    import io
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if len(dets) == 0:
        raise ValueError("Tidak ada deteksi untuk dievaluasi.")
    with contextlib.redirect_stdout(io.StringIO()) if not verbose else contextlib.nullcontext():
        gt = COCO(str(gt_json))
        dt = gt.loadRes(dets)
        E = COCOeval(gt, dt, iouType="bbox")
        E.evaluate()
        E.accumulate()
        E.summarize()
    s = E.stats
    out = {"mAP50_95": s[0], "mAP50": s[1], "mAP75": s[2],
           "AP_small": s[3], "AP_medium": s[4], "AP_large": s[5], "AR100": s[8]}
    # precision: [T(iou), R(recall), K(kelas), A(area: all, small, medium, large), M(maxDets)]
    P = E.eval["precision"]
    iou_thr = E.params.iouThrs
    t50 = int(np.argmin(np.abs(iou_thr - 0.5)))
    for k, cat in enumerate(E.params.catIds):
        name = CLASSES[COCO_CAT_IDS.index(cat)]
        p_all = P[:, :, k, 0, -1]
        p50 = P[t50, :, k, 0, -1]
        p_small = P[:, :, k, 1, -1]
        out[f"AP50_95_{name}"] = float(np.mean(p_all[p_all > -1])) if (p_all > -1).any() else float("nan")
        out[f"AP50_{name}"] = float(np.mean(p50[p50 > -1])) if (p50 > -1).any() else float("nan")
        out[f"APsmall_{name}"] = float(np.mean(p_small[p_small > -1])) if (p_small > -1).any() else float("nan")
    return {k: float(v) for k, v in out.items()}


# ----------------------------------------------------------------------------- hitung sel
def gt_counts(gt_json: Path) -> pd.DataFrame:
    """Jumlah ground truth per citra per kelas (baris = image_id, kolom = kelas)."""
    gt = load_json(gt_json)
    ids = [im["id"] for im in gt["images"]]
    df = pd.DataFrame(0, index=ids, columns=CLASSES)
    for a in gt["annotations"]:
        df.loc[a["image_id"], CLASSES[COCO_CAT_IDS.index(a["category_id"])]] += 1
    df.index.name = "image_id"
    return df


def pred_counts(dets: list, image_ids: list, tau) -> pd.DataFrame:
    """Jumlah deteksi dengan skor >= tau. tau boleh float atau dict {kelas: tau}."""
    if not isinstance(tau, dict):
        tau = {c: float(tau) for c in CLASSES}
    D = dets if isinstance(dets, pd.DataFrame) else pd.DataFrame(dets)
    if len(D) == 0:
        df = pd.DataFrame(0, index=image_ids, columns=CLASSES)
    else:
        kelas = D["category_id"].map(dict(zip(COCO_CAT_IDS, CLASSES)))
        keep = D["score"].values >= kelas.map(tau).values
        df = (pd.DataFrame({"image_id": D["image_id"].values[keep], "kelas": kelas.values[keep]})
              .groupby(["image_id", "kelas"]).size().unstack(fill_value=0)
              .reindex(index=image_ids, columns=CLASSES, fill_value=0))
    df.index.name = "image_id"
    return df.astype(int)


def count_errors(gt: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """MAE, MAPE (%), dan bias (rerata pred - gt) per kelas.

    MAPE hanya dihitung pada citra yang memiliki kelas tsb (gt > 0) agar tidak membagi nol;
    jumlah citra yang dipakai ditampilkan di kolom n_mape.
    """
    rows = []
    for c in CLASSES:
        err = pred[c] - gt[c]
        mask = gt[c] > 0
        mape = (err[mask].abs() / gt[c][mask]).mean() * 100 if mask.any() else float("nan")
        rows.append(dict(kelas=c, MAE=err.abs().mean(), MAPE=mape, bias=err.mean(),
                         n_img=len(err), n_mape=int(mask.sum())))
    return pd.DataFrame(rows).set_index("kelas")


def select_tau(gt_json_val: Path, dets_val: list, grid=TAU_GRID) -> tuple:
    """Pilih tau per kelas yang meminimalkan MAE di set VALIDASI. Kembalikan (tau_dict, tabel_kurva)."""
    gt = gt_counts(gt_json_val)
    D = pd.DataFrame(dets_val)
    curve = []
    for t in grid:
        e = count_errors(gt, pred_counts(D, list(gt.index), float(t)))
        for c in CLASSES:
            curve.append(dict(tau=float(t), kelas=c, MAE=e.loc[c, "MAE"]))
    curve = pd.DataFrame(curve)
    best = curve.loc[curve.groupby("kelas")["MAE"].idxmin()]
    return {r.kelas: float(r.tau) for r in best.itertuples()}, curve


def evaluate_run(gt_json: Path, dets: list, tau, verbose: bool = True) -> dict:
    """Ringkasan satu run pada satu split: metrik COCO + galat hitung dengan tau tertentu."""
    m = coco_metrics(gt_json, dets, verbose=verbose)
    gt = gt_counts(gt_json)
    e = count_errors(gt, pred_counts(dets, list(gt.index), tau))
    for c in CLASSES:
        m[f"MAE_{c}"] = float(e.loc[c, "MAE"])
        m[f"MAPE_{c}"] = float(e.loc[c, "MAPE"])
    return m
