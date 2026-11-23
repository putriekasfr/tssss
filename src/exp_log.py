"""Konfigurasi run (configs/*.yaml) dan log eksperimen (experiments/log_runs/*.json -> log.csv).

Setiap run, termasuk yang gagal, dicatat. Kolom status: planned / running / done / failed.
"""
import datetime as dt
import os
import random

import pandas as pd
import yaml

from .paths import CONFIG_DIR, LOG_PATH

LOG_COLUMNS = ["run_id", "owner", "model", "factor", "value", "seed", "status", "date",
               "gpu", "gpu_minutes", "val_mAP50", "val_mAP50_95", "val_AP_small",
               "val_MAE_WBC", "val_MAE_RBC", "val_MAE_Platelets", "notes"]


def load_config(run_id: str) -> dict:
    """Baca configs/<run_id>.yaml. Kunci `inherits: Rxx` menyalin konfigurasi run lain lalu menimpanya."""
    cfg = yaml.safe_load((CONFIG_DIR / f"{run_id}.yaml").read_text())
    base = cfg.pop("inherits", None)
    if base:
        merged = load_config(base)
        for k, v in cfg.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k] = {**merged[k], **v}
            else:
                merged[k] = v
        cfg = merged
    return cfg


def set_seed(seed: int, deterministic: bool = True):
    """Seed untuk random, numpy, dan torch (bila terpasang)."""
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# Log disimpan SATU BERKAS PER RUN (experiments/log_runs/<RUN>.json). Dua akun Colab yang berjalan bersamaan
# tidak pernah menulis berkas yang sama, jadi log tidak bisa saling menimpa. experiments/log.csv hanyalah
# ringkasan yang dibuat ulang dari berkas-berkas itu (untuk dibuka di Excel/laporan).
RUNLOG_DIR = LOG_PATH.parent / "log_runs"


def _run_file(run_id):
    return RUNLOG_DIR / f"{run_id}.json"


def _read_one(p):
    import json
    for _ in range(3):
        try:
            return json.loads(p.read_text())
        except Exception:
            import time as _t
            _t.sleep(1)
    return None


def read_log() -> pd.DataFrame:
    """Gabungkan semua experiments/log_runs/*.json menjadi satu tabel (urut run_id)."""
    rows = [r for r in (_read_one(p) for p in sorted(RUNLOG_DIR.glob("*.json"))) if r]
    if not rows:
        return pd.DataFrame(columns=LOG_COLUMNS)
    df = pd.DataFrame(rows)
    for c in LOG_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[LOG_COLUMNS + [c for c in df.columns if c not in LOG_COLUMNS]]
    return df.sort_values("run_id", key=lambda s: s.str.replace("-nms", "~", regex=False)).reset_index(drop=True)


def update_log(run_id: str, **fields):
    """Perbarui catatan satu run (hanya berkas milik run itu yang ditulis), lalu segarkan log.csv."""
    import json
    RUNLOG_DIR.mkdir(parents=True, exist_ok=True)
    f = _run_file(run_id)
    rec = (_read_one(f) if f.exists() else None) or {c: None for c in LOG_COLUMNS}
    rec["run_id"] = run_id
    rec["date"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    for k, v in fields.items():
        try:
            v = v.item()                     # numpy -> tipe Python biasa
        except AttributeError:
            pass
        rec[k] = None if (isinstance(v, float) and v != v) else v
    tmp = f.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    tmp.replace(f)
    df = read_log()
    try:                                     # ringkasan; gagal menulis tidak memengaruhi catatan per run
        df.to_csv(LOG_PATH, index=False)
    except Exception:
        pass
    return df


def gpu_name() -> str:
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    except ImportError:
        return "CPU"
