"""Analisis kesalahan: cocokkan deteksi (skor >= tau) dengan ground truth, lalu beri kategori.

Pencocokan per citra per kelas, serakah berdasarkan skor, IoU >= 0,5 (sama dengan AP@0.5).
Kategori diberi berurutan; kategori pertama yang terpenuhi dipakai.

False negative (sel terlewat):
  salah kelas     -> ada deteksi kelas lain dengan IoU >= 0,5 pada sel ini
  objek kecil     -> luas < 32^2 px
  terpotong tepi  -> kotak menyentuh tepi citra (<= 2 px)
  bertumpuk       -> IoU > 0,3 dengan sel lain
  terlewat lain   -> selain di atas (sering: tepi buram / kontras rendah)
False positive (deteksi berlebih):
  salah kelas     -> IoU >= 0,5 dengan sel kelas lain
  deteksi ganda   -> IoU >= 0,5 dengan sel sekelas yang sudah dicocokkan deteksi lain
  lokalisasi buruk-> 0,1 <= IoU < 0,5 dengan sel sekelas
  partikel kecil  -> luas deteksi < 32^2 px dan tidak menempel sel (artefak/debris)
  latar/artefak   -> selain di atas
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import CLASSES, COCO_CAT_IDS

FN_CATS = ["salah kelas", "objek kecil", "terpotong tepi", "bertumpuk", "terlewat lain"]
FP_CATS = ["salah kelas", "deteksi ganda", "lokalisasi buruk", "partikel kecil", "latar/artefak"]


def _xyxy(b):
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    return np.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], 1)


def iou(a, b):
    """IoU antara kotak xywh a (N) dan b (M) -> (N, M)."""
    A, B = _xyxy(a), _xyxy(b)
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)))
    iw = np.clip(np.minimum(A[:, None, 2], B[:, 2]) - np.maximum(A[:, None, 0], B[:, 0]), 0, None)
    ih = np.clip(np.minimum(A[:, None, 3], B[:, 3]) - np.maximum(A[:, None, 1], B[:, 1]), 0, None)
    inter = iw * ih
    aa = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])
    bb = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return inter / (aa[:, None] + bb - inter + 1e-9)


def match(gt_json, dets, tau, iou_thr=0.5) -> pd.DataFrame:
    """Satu baris per sel GT dan per deteksi terpakai: jenis TP/FP/FN + kategori kesalahan."""
    gt = json.loads(Path(gt_json).read_text()) if not isinstance(gt_json, dict) else gt_json
    if not isinstance(tau, dict):
        tau = {c: float(tau) for c in CLASSES}
    id2cls = dict(zip(COCO_CAT_IDS, CLASSES))
    imgs = {im["id"]: im for im in gt["images"]}
    G = pd.DataFrame(gt["annotations"])
    D = pd.DataFrame(dets)
    D = D[D.score.values >= D.category_id.map(id2cls).map(tau).values] if len(D) else D
    rows = []
    for img_id, im in imgs.items():
        W, H = im["width"], im["height"]
        g = G[G.image_id == img_id] if len(G) else G
        d = D[D.image_id == img_id].sort_values("score", ascending=False) if len(D) else D
        gb, gc = np.array(g.bbox.tolist()).reshape(-1, 4), g.category_id.values
        db, dc, ds = np.array(d.bbox.tolist()).reshape(-1, 4), d.category_id.values, d.score.values
        I = iou(db, gb)                       # deteksi x GT
        G_I = iou(gb, gb)
        np.fill_diagonal(G_I, 0)
        g_match = np.full(len(gb), -1)
        d_type = [""] * len(db)
        for i in range(len(db)):              # serakah berdasarkan skor, sekelas
            cand = np.where((gc == dc[i]) & (g_match < 0))[0]
            if len(cand):
                j = cand[np.argmax(I[i, cand])]
                if I[i, j] >= iou_thr:
                    g_match[j] = i
                    d_type[i] = "TP"
        for i in range(len(db)):
            if d_type[i] == "TP":
                rows.append(dict(image_id=img_id, jenis="TP", kelas=id2cls[dc[i]], kategori="",
                                 bbox=db[i].tolist(), score=ds[i]))
                continue
            same, other = gc == dc[i], gc != dc[i]
            if other.any() and I[i, other].max() >= iou_thr:
                cat = "salah kelas"
            elif same.any() and I[i, same].max() >= iou_thr:
                cat = "deteksi ganda"
            elif same.any() and I[i, same].max() >= 0.1:
                cat = "lokalisasi buruk"
            elif db[i, 2] * db[i, 3] < 32 ** 2 and (len(gb) == 0 or I[i].max() < 0.1):
                cat = "partikel kecil"
            else:
                cat = "latar/artefak"
            rows.append(dict(image_id=img_id, jenis="FP", kelas=id2cls[dc[i]], kategori=cat,
                             bbox=db[i].tolist(), score=ds[i]))
        for j in np.where(g_match < 0)[0]:
            x, y, w, h = gb[j]
            other_det = (dc != gc[j]) if len(dc) else np.array([], bool)
            if other_det.any() and I[other_det, j].max() >= iou_thr:
                cat = "salah kelas"
            elif w * h < 32 ** 2:
                cat = "objek kecil"
            elif x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2:
                cat = "terpotong tepi"
            elif len(gb) > 1 and G_I[j].max() > 0.3:
                cat = "bertumpuk"
            else:
                cat = "terlewat lain"
            rows.append(dict(image_id=img_id, jenis="FN", kelas=id2cls[gc[j]], kategori=cat,
                             bbox=gb[j].tolist(), score=np.nan))
    return pd.DataFrame(rows)


def category_table(m: pd.DataFrame) -> pd.DataFrame:
    """Frekuensi kategori kesalahan per jenis (FP/FN) dan kelas, plus persentase dalam jenisnya."""
    e = m[m.jenis != "TP"]
    t = e.groupby(["jenis", "kategori", "kelas"]).size().unstack(fill_value=0).reindex(columns=CLASSES, fill_value=0)
    t["total"] = t.sum(1)
    t["% dalam jenis"] = (t["total"] / t.groupby(level="jenis")["total"].transform("sum") * 100).round(1)
    return t


def prf(m: pd.DataFrame) -> pd.DataFrame:
    """Presisi, recall, F1 per kelas pada ambang tau."""
    out = []
    for c in CLASSES:
        x = m[m.kelas == c]
        tp, fp, fn = (x.jenis == "TP").sum(), (x.jenis == "FP").sum(), (x.jenis == "FN").sum()
        p = tp / (tp + fp) if tp + fp else np.nan
        r = tp / (tp + fn) if tp + fn else np.nan
        out.append(dict(kelas=c, TP=tp, FP=fp, FN=fn, presisi=p, recall=r, F1=2 * p * r / (p + r) if p + r else np.nan))
    return pd.DataFrame(out).set_index("kelas")
