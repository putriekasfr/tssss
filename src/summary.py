"""Ringkasan hasil M3: evaluasi banyak run, rerata ± SD antar seed, galat per citra per kelompok."""
import numpy as np
import pandas as pd

from .data_utils import load_source_map
from .evaluate import evaluate_run, gt_counts, load_json, pred_counts, select_tau
from .exp_log import read_log
from .paths import CLASSES, COCO_DIR, PRED_DIR

RUN_LABEL = {
    "R01": "YOLOv8s · 640 · NMS 0,70", "R04": "YOLOv8s · 832 · NMS 0,70",
    "R02": "Faster R-CNN · penuh · NMS 0,50", "R06": "Faster R-CNN · backbone beku · NMS 0,50",
}


def label(run_id: str) -> str:
    """Nama konfigurasi yang mudah dibaca, termasuk varian NMS ('R01-nms0.45') dan run seed."""
    if run_id in RUN_LABEL:
        return RUN_LABEL[run_id]
    if "-nms" in run_id:
        base, iou = run_id.split("-nms")
        return RUN_LABEL.get(base, base).rsplit(" · NMS", 1)[0] + f" · NMS {float(iou):.2f}".replace(".", ",")
    log = read_log().set_index("run_id")
    return f"{model_of(run_id)} · {log.loc[run_id, 'value']}" if run_id in log.index else run_id


def model_of(run_id: str) -> str:
    log = read_log().set_index("run_id")
    return "YOLOv8s" if str(log.loc[run_id, "model"]).startswith("yolo") else "Faster R-CNN"


def runs_done():
    log = read_log()
    return list(log.loc[log.status == "done", "run_id"])


def tau_from_val(run_id) -> dict:
    """tau per kelas yang dipilih di VALIDASI untuk run ini (dipakai juga untuk set uji)."""
    return select_tau(COCO_DIR / "val.json", load_json(PRED_DIR / f"{run_id}_val.json"))[0]


def eval_runs(run_ids, split="val") -> pd.DataFrame:
    rows = {}
    for r in run_ids:
        dets = load_json(PRED_DIR / f"{r}_{split}.json")
        tau = tau_from_val(r)
        m = evaluate_run(COCO_DIR / f"{split}.json", dets, tau, verbose=False)
        m["tau"] = str(tau)
        rows[r] = m
    return pd.DataFrame(rows).T


def mean_sd(df: pd.DataFrame, groups: dict, cols) -> tuple:
    """groups: {nama: [run...]} -> (tabel numerik rerata/SD, tabel teks 'rerata ± SD')."""
    num, txt = {}, {}
    for g, runs in groups.items():
        x = df.loc[runs, cols].astype(float)
        num[(g, "mean")], num[(g, "sd")] = x.mean(), x.std(ddof=1)
        txt[g] = [f"{a:.4f} ± {b:.4f}" if abs(a) < 2 else f"{a:.3f} ± {b:.3f}"
                  for a, b in zip(x.mean(), x.std(ddof=1))]
    return pd.DataFrame(num).T, pd.DataFrame(txt, index=cols).T


def per_image_errors(run_ids, split, data_root) -> pd.DataFrame:
    """Galat hitung per citra per kelas (pred - gt) beserta sumber dan kepadatan RBC."""
    gt_json = COCO_DIR / f"{split}.json"
    gt = gt_counts(gt_json)
    imgs = load_json(gt_json)["images"]
    src = load_source_map(data_root)
    meta = pd.DataFrame({"image_id": [i["id"] for i in imgs],
                         "file": [i["file_name"] for i in imgs],
                         "source": [src[i["file_name"][:-4]] for i in imgs]}).set_index("image_id")
    meta["padat"] = np.where(gt["RBC"] > 15, "RBC>15", "RBC≤15")
    out = []
    for r in run_ids:
        pc = pred_counts(load_json(PRED_DIR / f"{r}_{split}.json"), list(gt.index), tau_from_val(r))
        e = (pc - gt).stack().rename("err").reset_index().rename(columns={"level_1": "kelas"})
        e["run"] = r
        out.append(e)
    df = pd.concat(out).join(meta, on="image_id")
    df["abs_err"] = df.err.abs()
    return df


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Korelasi Pearson & Spearman antar-run antara mAP@[.5:.95] (dan AP per kelas) dengan MAE hitung per kelas.

    df: satu baris per run/varian (keluaran eval_runs) dengan kolom 'model'. Dihitung untuk gabungan run
    dan per paradigma. Dengan n kecil, p-value hanya indikatif; n dilaporkan.
    """
    from scipy.stats import pearsonr, spearmanr
    out = []
    groups = [("semua run", df)] + [(g, d) for g, d in df.groupby("model")]
    for gname, d in groups:
        for c in CLASSES:
            for xname in ["mAP50_95", f"AP50_95_{c}"]:
                x, y = d[xname].astype(float), d[f"MAE_{c}"].astype(float)
                if len(d) < 3 or x.nunique() < 2 or y.nunique() < 2:
                    continue
                r, pr = pearsonr(x, y)
                rho, ps = spearmanr(x, y)
                out.append(dict(kelompok=gname, kelas=c, x=xname, n=len(d), pearson_r=r, p_pearson=pr,
                                spearman_rho=rho, p_spearman=ps))
    return pd.DataFrame(out, columns=["kelompok", "kelas", "x", "n", "pearson_r", "p_pearson",
                                      "spearman_rho", "p_spearman"])
