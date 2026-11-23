"""Satu fungsi per jenis model: latih -> prediksi validasi -> evaluasi -> catat log.

Dipakai notebook M2 (baseline R01, R02) dan notebook 03/04 (run ablasi M3) agar semua run
diproses dengan langkah yang identik. Set uji TIDAK disentuh di sini.
"""
import tempfile
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd

from .data_utils import yolo_to_coco
from .evaluate import evaluate_run, select_tau
from .exp_log import gpu_name, load_config, read_log, set_seed, update_log
from .paths import COCO_DIR, OUTPUT_DIR, PRED_DIR, PROJECT_ROOT, SPLITS


def ensure_coco(data_root):
    """Buat outputs/coco/<split>.json bila belum ada."""
    for s in SPLITS:
        if not (COCO_DIR / f"{s}.json").exists():
            yolo_to_coco(data_root, s, COCO_DIR / f"{s}.json")


def free_gpu(verbose=True):
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    if verbose:
        print(f"VRAM terpakai sekarang: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")


def _prev_minutes(run_id) -> float:
    log = read_log()
    if "gpu_minutes" not in log or not (log.run_id == run_id).any():
        return 0.0
    v = log.loc[log.run_id == run_id, "gpu_minutes"].iloc[0]
    return 0.0 if pd.isna(v) else float(v)


def _save_env(run_id):
    env_dir = PROJECT_ROOT / "experiments" / "env"
    env_dir.mkdir(parents=True, exist_ok=True)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (env_dir / f"{run_id}.txt").write_text(f"GPU: {gpu_name()}\nPython: {sys.version.split()[0]}\n" + freeze)


def _train_logged(cfg, train_fn):
    run_id = cfg["run_id"]
    prev = _prev_minutes(run_id)
    update_log(run_id, owner=cfg["owner"], model=cfg["model"], factor=cfg["factor"], value=cfg["value"],
               seed=cfg["seed"], status="running", gpu=gpu_name())
    t0 = time.time()
    try:
        out = train_fn()
    except Exception as e:
        update_log(run_id, status="failed", gpu_minutes=round(prev + (time.time() - t0) / 60, 1),
                   notes=f"{type(e).__name__}: {str(e)[:150]}")
        raise
    minutes = round(prev + (time.time() - t0) / 60, 1)
    update_log(run_id, status="trained", gpu_minutes=minutes)
    print(f"[{run_id}] pelatihan selesai · total waktu GPU {minutes} menit")
    return out


def _finish(cfg, dets_val, note):
    run_id = cfg["run_id"]
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    (PRED_DIR / f"{run_id}_val.json").write_text(json.dumps(dets_val))
    tau, _ = select_tau(COCO_DIR / "val.json", dets_val)
    m = evaluate_run(COCO_DIR / "val.json", dets_val, tau)
    update_log(run_id, status="done", val_mAP50=round(m["mAP50"], 4), val_mAP50_95=round(m["mAP50_95"], 4),
               val_AP_small=round(m["AP_small"], 4), val_MAE_WBC=round(m["MAE_WBC"], 3),
               val_MAE_RBC=round(m["MAE_RBC"], 3), val_MAE_Platelets=round(m["MAE_Platelets"], 3),
               notes=f"tau={tau}; {note}")
    print(f"[{run_id}] prediksi validasi -> {PRED_DIR / f'{run_id}_val.json'}")
    return m


def run_yolo(run_id, data_root, runs_dir) -> dict:
    """Latih YOLOv8 sesuai configs/<run_id>.yaml lalu evaluasi di validasi."""
    from . import yolo_utils
    from ultralytics import YOLO

    cfg = load_config(run_id)
    assert cfg["model"].startswith("yolo"), f"{run_id} bukan run YOLO"
    ensure_coco(data_root)
    _save_env(run_id)
    set_seed(int(cfg["seed"]))
    # yaml data disimpan di disk lokal Colab (bukan Drive): file kecil di Drive bisa "hilang" sesaat saat sinkronisasi
    local = Path("/content") if Path("/content").exists() else Path(tempfile.gettempdir())
    data_yaml = yolo_utils.make_data_yaml(data_root, local / "data_txl.yaml")
    _train_logged(cfg, lambda: yolo_utils.train(cfg, data_yaml, runs_dir))
    free_gpu()
    best = runs_dir / run_id / "weights" / "best.pt"
    dets = yolo_utils.predict_coco(best, data_root / "images" / "val", COCO_DIR / "val.json", cfg)
    n_param = sum(p.numel() for p in YOLO(str(best)).model.parameters()) / 1e6
    free_gpu(verbose=False)
    return _finish(cfg, dets, f"params={n_param:.2f}M")


def run_frcnn(run_id, data_root, runs_dir) -> dict:
    """Latih Faster R-CNN sesuai configs/<run_id>.yaml lalu evaluasi di validasi."""
    from . import frcnn

    cfg = load_config(run_id)
    assert cfg["model"].startswith("fasterrcnn"), f"{run_id} bukan run Faster R-CNN"
    ensure_coco(data_root)
    _save_env(run_id)
    out = runs_dir / run_id
    hist = _train_logged(cfg, lambda: frcnn.train(cfg, data_root, COCO_DIR, out, device="cuda"))
    print(f"[{run_id}] puncak VRAM selama pelatihan: {hist.gpu_peak_gb.max():.1f} GB")
    free_gpu()
    model = frcnn.load_best(cfg, out, device="cuda")
    dets = frcnn.predict_coco(model, data_root / "images" / "val", COCO_DIR / "val.json", cfg, device="cuda")
    n_param = frcnn.count_params(model)
    bobot = getattr(model, "bobot_dipakai", "best")
    del model
    free_gpu(verbose=False)
    best_ep = int(hist.loc[hist.fitness.idxmax(), "epoch"])
    catatan = f"params={n_param:.2f}M; best_epoch={best_ep}"
    if bobot != "best":
        catatan += f"; BOBOT DIPAKAI: {bobot}"
    return _finish(cfg, dets, catatan)


# ----------------------------------------------------------------------------- M3
YOLO_CANDIDATES = ["R01", "R04"]                 # baseline 640 + ablasi A1 resolusi 832
FRCNN_CANDIDATES = ["R02", "R06"]                # baseline fine-tune penuh + backbone beku
NMS_IOUS = [0.45, 0.60, 0.75]                    # ablasi A2 (menu T1-A): ambang IoU NMS
SEED_RUNS = {"yolo": ["R07", "R08"], "frcnn": ["R09", "R10"]}


def choose_best(candidates) -> str:
    """Konfigurasi terbaik = val mAP@[.5:.95] tertinggi di antara run yang sudah selesai (aturan tetap)."""
    log = read_log().set_index("run_id")
    done = [r for r in candidates if r in log.index and log.loc[r, "status"] == "done"]
    missing = [r for r in candidates if r not in done]
    if missing:
        raise RuntimeError(f"Run berikut belum selesai: {missing}. Selesaikan dulu sebelum memilih.")
    return max(done, key=lambda r: float(log.loc[r, "val_mAP50_95"]))


def prepare_seed_configs(kind: str, best: str):
    """Tulis configs R07/R08 (YOLO) atau R09/R10 (Faster R-CNN) agar mewarisi konfigurasi terbaik.

    Config yang run-nya sudah dimulai tidak diubah lagi (menjaga jejak eksperimen).
    """
    from .paths import CONFIG_DIR
    import yaml
    log = read_log().set_index("run_id")
    for run_id, seed in zip(SEED_RUNS[kind], [123, 2024]):
        status = log.loc[run_id, "status"] if run_id in log.index else "planned"
        if status not in ("planned",) and not pd.isna(status):
            print(f"{run_id} sudah {status}; config tidak diubah")
            continue
        cfg = {"inherits": best, "run_id": run_id, "owner": "Tsabita" if kind == "yolo" else "Putri",
               "factor": "multi-seed", "value": f"{best} seed {seed}", "seed": seed}
        path = CONFIG_DIR / f"{run_id}.yaml"
        path.write_text(f"# {run_id} - perbandingan utama 3 seed: konfigurasi terbaik di validasi = {best}\n"
                        + yaml.safe_dump(cfg, sort_keys=False))
        update_log(run_id, value=cfg["value"], factor="multi-seed", seed=seed, owner=cfg["owner"])
        print(f"{run_id}: mewarisi {best}, seed {seed}")


def load_model(run_id, runs_dir):
    """Kembalikan (jenis, objek model/bobot, cfg) untuk prediksi, efisiensi, dan XAI."""
    cfg = load_config(run_id)
    if cfg["model"].startswith("yolo"):
        return "yolo", runs_dir / run_id / "weights" / "best.pt", cfg
    from . import frcnn
    return "frcnn", frcnn.load_best(cfg, runs_dir / run_id, device="cuda"), cfg


def predict_split(run_id, split, data_root, runs_dir) -> list:
    """Prediksi satu split -> outputs/preds/<run>_<split>.json."""
    kind, obj, cfg = load_model(run_id, runs_dir)
    ensure_coco(data_root)
    gt = COCO_DIR / f"{split}.json"
    img_dir = data_root / "images" / split
    if kind == "yolo":
        from . import yolo_utils
        dets = yolo_utils.predict_coco(obj, img_dir, gt, cfg)
    else:
        from . import frcnn
        dets = frcnn.predict_coco(obj, img_dir, gt, cfg, device="cuda")
        del obj
    free_gpu(verbose=False)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    (PRED_DIR / f"{run_id}_{split}.json").write_text(json.dumps(dets))
    return dets


# ----------------------------------------------------------------------------- ablasi A2: NMS (tanpa latih ulang)
def nms_variant_id(base: str, iou: float) -> str:
    return f"{base}-nms{iou:.2f}"


def run_nms_ablation(base, data_root, runs_dir, ious=NMS_IOUS, owner="Putri") -> pd.DataFrame:
    """Prediksi ulang set validasi dari bobot `base` dengan ambang IoU NMS lain (tanpa pelatihan).

    Hanya satu faktor yang berubah: predict.iou_nms. Bobot, ambang skor (0,001), dan max_det sama.
    Setiap varian dicatat di log sebagai '<base>-nms<iou>' dan dinilai dengan langkah yang sama
    (pycocotools + galat hitung, tau dipilih di validasi untuk varian itu).
    """
    import copy
    cfg0 = load_config(base)
    kind = "yolo" if cfg0["model"].startswith("yolo") else "frcnn"
    ensure_coco(data_root)
    rows = {}
    for iou in ious:
        vid = nms_variant_id(base, iou)
        cfg = copy.deepcopy(cfg0)
        cfg["run_id"], cfg["predict"]["iou_nms"] = vid, float(iou)
        update_log(vid, owner=owner, model=cfg0["model"], factor="A2 NMS IoU", value=f"{base} · NMS IoU {iou:.2f}",
                   seed=cfg0["seed"], status="running", gpu=gpu_name())
        t0 = time.time()
        if kind == "yolo":
            from . import yolo_utils
            dets = yolo_utils.predict_coco(runs_dir / base / "weights" / "best.pt", data_root / "images" / "val",
                                           COCO_DIR / "val.json", cfg)
        else:
            from . import frcnn
            model = frcnn.load_best(cfg, runs_dir / base, device="cuda")     # box_nms_thresh dari cfg
            dets = frcnn.predict_coco(model, data_root / "images" / "val", COCO_DIR / "val.json", cfg, device="cuda")
            del model
        free_gpu(verbose=False)
        update_log(vid, gpu_minutes=round((time.time() - t0) / 60, 1))
        rows[vid] = _finish(cfg, dets, f"bobot {base}; prediksi ulang tanpa latih")
    return pd.DataFrame(rows).T


# ----------------------------------------------------------------------------- proses terpisah (hemat VRAM)
def gpu_used_gb() -> float:
    """VRAM yang sedang terpakai di seluruh GPU (semua proses), dalam GB."""
    try:
        import torch
        free, total = torch.cuda.mem_get_info()
        return (total - free) / 1024 ** 3
    except Exception:
        return float("nan")


def run_isolated(job: str, data_root, log_file=None) -> dict:
    """Jalankan satu pekerjaan GPU (lihat src/run_job.py) di proses Python terpisah.

    Begitu proses selesai, semua VRAM dan RAM-nya dilepas oleh sistem operasi, sehingga memori tidak
    menumpuk antar run. Log lengkap disimpan ke `log_file`; di notebook hanya baris ringkas yang dicetak
    (bilah progres tqdm disaring agar keluaran tidak membanjir).
    """
    import os
    import re
    import subprocess
    env = os.environ.copy()
    env["TXL_ROOT"] = str(data_root)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTHONUNBUFFERED"] = "1"
    log_file = Path(log_file) if log_file else PROJECT_ROOT / "experiments" / "proses" / f"{job.replace(':', '_')}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    hasil = None
    with open(log_file, "a") as lf:
        p = subprocess.Popen([sys.executable, "-u", "-m", "src.run_job", job], cwd=str(PROJECT_ROOT), env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for raw in p.stdout:
            for line in raw.replace("\r", "\n").splitlines():
                lf.write(line + "\n")
                if line.startswith("__HASIL__"):
                    hasil = json.loads(line[len("__HASIL__"):])
                elif line.strip() and not re.search(r"\d+%\|| it/s|s/it|━", line):
                    print(line, flush=True)
        rc = p.wait()
    if rc != 0 or hasil is None:
        raise RuntimeError(f"Pekerjaan {job} gagal (kode {rc}). Lihat log lengkap: {log_file}")
    return hasil


# ----------------------------------------------------------------------------- pemulihan log dari artefak
def rebuild_log(data_root, runs_dir) -> pd.DataFrame:
    """Bangun ulang experiments/log.csv dari artefak di Drive bila log hilang/kurang.

    Sumber kebenaran: configs/, runs/<RUN>/ (bobot & riwayat pelatihan) dan outputs/preds/<RUN>_val.json.
    - prediksi validasi ada        -> status 'done', metrik dihitung ulang (pycocotools + galat hitung)
    - pelatihan selesai, belum ada prediksi -> status 'trained' (notebook akan memprediksi saja, tanpa latih ulang)
    - pelatihan belum lengkap      -> status 'running' (notebook melanjutkan dari last.pt)
    Waktu GPU diambil dari riwayat pelatihan (YOLO: kolom time di results.csv; Faster R-CNN: menit per epoch).
    """
    from .paths import CONFIG_DIR
    ensure_coco(data_root)
    log = read_log().set_index("run_id") if len(read_log()) else pd.DataFrame()
    run_ids = sorted(p.stem for p in CONFIG_DIR.glob("R*.yaml"))
    variants = [nms_variant_id(b, i) for b in ["R01", "R02"] for i in NMS_IOUS]
    for rid in run_ids + variants:
        base = rid.split("-nms")[0]
        cfg = load_config(base)
        pred = PRED_DIR / f"{rid}_val.json"
        rd = runs_dir / base
        minutes = None
        if cfg["model"].startswith("yolo"):
            done_train = (rd / "DONE").exists() or (rd / "weights" / "best.pt").exists()
            f = rd / "results.csv"
            if f.exists():
                r = pd.read_csv(f); r.columns = [c.strip() for c in r.columns]
                if "time" in r.columns:
                    minutes = round(float(r["time"].iloc[-1]) / 60, 1)
                done_train = done_train and len(r) >= int(cfg["epochs"])
        else:
            f = rd / "history.csv"
            h = pd.read_csv(f) if f.exists() else pd.DataFrame()
            done_train = len(h) >= int(cfg["epochs"])
            if len(h) and "minutes" in h.columns:
                minutes = round(float(h["minutes"].sum()), 1)
        if "-nms" in rid:
            minutes = None
        sudah = rid in log.index and log.loc[rid, "status"] == "done"
        if sudah:
            continue
        info = dict(owner=("Putri" if "-nms" in rid else cfg.get("owner")), model=cfg["model"],
                    factor=("A2 NMS IoU" if "-nms" in rid else cfg.get("factor")),
                    value=(f"{base} · NMS IoU {float(rid.split('-nms')[1]):.2f}" if "-nms" in rid else cfg.get("value")),
                    seed=cfg["seed"])
        if minutes is not None:
            info["gpu_minutes"] = minutes
        if pred.exists():
            dets = json.loads(pred.read_text())
            tau, _ = select_tau(COCO_DIR / "val.json", dets)
            m = evaluate_run(COCO_DIR / "val.json", dets, tau, verbose=False)
            update_log(rid, status="done", val_mAP50=round(m["mAP50"], 4), val_mAP50_95=round(m["mAP50_95"], 4),
                       val_AP_small=round(m["AP_small"], 4), val_MAE_WBC=round(m["MAE_WBC"], 3),
                       val_MAE_RBC=round(m["MAE_RBC"], 3), val_MAE_Platelets=round(m["MAE_Platelets"], 3),
                       notes=f"tau={tau}; log dipulihkan dari artefak", **info)
            print(f"  {rid}: done (dipulihkan dari prediksi validasi)")
        elif "-nms" not in rid and done_train:
            update_log(rid, status="trained", notes="log dipulihkan: pelatihan selesai, prediksi validasi belum ada", **info)
            print(f"  {rid}: trained (tinggal prediksi validasi)")
        elif "-nms" not in rid and ((rd / "last.pt").exists() or (rd / "weights" / "last.pt").exists()):
            update_log(rid, status="running", notes="log dipulihkan: pelatihan belum lengkap", **info)
            print(f"  {rid}: running (dilanjutkan dari checkpoint)")
        else:
            update_log(rid, status="planned", **info)
    return read_log()
