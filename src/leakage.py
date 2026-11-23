"""Pemeriksaan duplikat/kebocoran: MD5 (identik byte) dan perceptual hash (pHash).

pHash ditulis sendiri mengikuti algoritma imagehash.phash (grayscale 32x32 -> DCT 2D ->
blok frekuensi rendah 8x8 -> bandingkan dengan median) agar hasilnya sama di lokal dan Colab
tanpa bergantung versi pustaka.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dct


def phash(path: Path, hash_size: int = 8, highfreq_factor: int = 4) -> np.ndarray:
    """Kembalikan pHash 64-bit sebagai array boolean (hash_size*hash_size,)."""
    n = hash_size * highfreq_factor
    with Image.open(path) as im:
        g = im.convert("L").resize((n, n), Image.LANCZOS)
    px = np.asarray(g, dtype=np.float64)
    d = dct(dct(px, axis=0), axis=1)
    low = d[:hash_size, :hash_size]
    return (low > np.median(low)).flatten()


def md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def hash_split(data_root: Path, split: str, stems: list) -> pd.DataFrame:
    rows = []
    for s in stems:
        p = Path(data_root) / "images" / split / f"{s}.png"
        rows.append(dict(split=split, stem=s, md5=md5(p), phash=phash(p)))
    return pd.DataFrame(rows)


def cross_pairs(a: pd.DataFrame, b: pd.DataFrame, max_dist: int = 8, same: bool = False) -> pd.DataFrame:
    """Semua pasangan (a_i, b_j) dengan jarak Hamming pHash <= max_dist.

    same=True dipakai untuk pasangan di dalam satu split (hanya i < j).
    """
    A = np.stack(a["phash"].values)
    B = np.stack(b["phash"].values)
    dist = (A[:, None, :] != B[None, :, :]).sum(-1)
    ii, jj = np.where(dist <= max_dist)
    rows = []
    for i, j in zip(ii, jj):
        if same and i >= j:
            continue
        rows.append(dict(split_a=a.split.iloc[i], stem_a=a.stem.iloc[i],
                         split_b=b.split.iloc[j], stem_b=b.stem.iloc[j],
                         hamming=int(dist[i, j]),
                         identical_md5=a.md5.iloc[i] == b.md5.iloc[j]))
    cols = ["split_a", "stem_a", "split_b", "stem_b", "hamming", "identical_md5"]
    return pd.DataFrame(rows, columns=cols).sort_values("hamming").reset_index(drop=True)
