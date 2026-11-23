"""Membaca label YOLO, menyimpan split, dan konversi YOLO -> COCO."""
import json
from pathlib import Path

import pandas as pd
from PIL import Image

from .paths import CLASSES, COCO_CAT_IDS, SPLITS, SPLIT_DIR


def list_split(data_root: Path, split: str) -> list:
    """Nama berkas citra (tanpa ekstensi) pada satu split, urut abjad agar deterministik."""
    return sorted(p.stem for p in (Path(data_root) / "images" / split).glob("*.png"))


def read_yolo(label_path: Path, dedup: bool = True) -> list:
    """Baca satu berkas label YOLO -> list (cls, xc, yc, w, h) dalam koordinat ternormalisasi.

    dedup=True membuang baris yang identik persis. Audit M2 menemukan 47 baris duplikat di
    5 berkas label TXL-PBC; ultralytics juga membuangnya otomatis, jadi ground truth COCO
    dibuat sama agar kedua model dinilai terhadap label yang identik.
    """
    rows, seen = [], set()
    p = Path(label_path)
    if not p.exists():
        return rows
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        key = tuple(parts)
        if dedup and key in seen:
            continue
        seen.add(key)
        c, xc, yc, w, h = parts
        rows.append((int(c), float(xc), float(yc), float(w), float(h)))
    return rows


def count_duplicate_lines(data_root: Path) -> "pd.DataFrame":
    """Berkas label yang memuat baris identik lebih dari sekali."""
    rows = []
    for split in SPLITS:
        for p in sorted((Path(data_root) / "labels" / split).glob("*.txt")):
            lines = [l.strip() for l in p.read_text().splitlines() if l.strip()]
            raw = read_yolo(p, dedup=False)
            clean = read_yolo(p, dedup=True)
            if len(raw) != len(clean):
                dup_cls = pd.Series([CLASSES[r[0]] for r in raw]).value_counts() - \
                    pd.Series([CLASSES[r[0]] for r in clean]).value_counts()
                rows.append(dict(split=split, stem=p.stem, baris_asli=len(lines), baris_unik=len(clean),
                                 duplikat=len(raw) - len(clean),
                                 **{f"dup_{c}": int(dup_cls.get(c, 0)) for c in CLASSES}))
    return pd.DataFrame(rows)


def load_source_map(data_root: Path) -> dict:
    """stem citra -> dataset asal (BCCD, BCDD, PBC, Raabin-WBC) dari metadata_file.xlsx."""
    meta = pd.read_excel(Path(data_root) / "metadata_file.xlsx")
    return {Path(n).stem: s for n, s in zip(meta["New_Image_Name"], meta["Source_Dataset"])}


def load_boxes(data_root: Path, dedup: bool = True) -> tuple:
    """Kembalikan (df_images, df_boxes) untuk seluruh split.

    df_images: satu baris per citra (split, stem, width, height, source, jumlah tiap kelas).
    df_boxes : satu baris per kotak dengan koordinat piksel dan luas.
    """
    data_root = Path(data_root)
    src = load_source_map(data_root)
    img_rows, box_rows = [], []
    for split in SPLITS:
        for stem in list_split(data_root, split):
            with Image.open(data_root / "images" / split / f"{stem}.png") as im:
                W, H = im.size
            labels = read_yolo(data_root / "labels" / split / f"{stem}.txt", dedup=dedup)
            counts = {c: 0 for c in CLASSES}
            for c, xc, yc, w, h in labels:
                counts[CLASSES[c]] += 1
                bw, bh = w * W, h * H
                box_rows.append(dict(split=split, stem=stem, source=src.get(stem, "?"),
                                     cls=CLASSES[c], x=(xc - w / 2) * W, y=(yc - h / 2) * H,
                                     w=bw, h=bh, area=bw * bh,
                                     img_w=W, img_h=H))
            img_rows.append(dict(split=split, stem=stem, width=W, height=H,
                                 source=src.get(stem, "?"), n_boxes=len(labels), **counts))
    return pd.DataFrame(img_rows), pd.DataFrame(box_rows)


def write_split_files(data_root: Path, out_dir: Path = SPLIT_DIR) -> dict:
    """Simpan daftar nama berkas tiap split resmi ke splits/{train,val,test}.txt."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for split in SPLITS:
        stems = list_split(data_root, split)
        (out_dir / f"{split}.txt").write_text("\n".join(f"{s}.png" for s in stems) + "\n")
        sizes[split] = len(stems)
    return sizes


def read_split_file(split: str, split_dir: Path = SPLIT_DIR) -> list:
    """Baca daftar stem dari splits/<split>.txt (sumber kebenaran untuk semua notebook)."""
    names = (Path(split_dir) / f"{split}.txt").read_text().split()
    return [Path(n).stem for n in names]


def yolo_to_coco(data_root: Path, split: str, out_json: Path) -> dict:
    """Konversi label YOLO satu split ke COCO JSON.

    - image_id = urutan pada splits/<split>.txt mulai 1 (sama untuk semua model).
    - bbox = [x_min, y_min, w, h] dalam piksel citra asli; area = w*h (dipakai AP_small).
    - category_id = indeks YOLO + 1.
    """
    data_root = Path(data_root)
    stems = read_split_file(split)
    images, anns = [], []
    ann_id = 1
    for img_id, stem in enumerate(stems, start=1):
        with Image.open(data_root / "images" / split / f"{stem}.png") as im:
            W, H = im.size
        images.append(dict(id=img_id, file_name=f"{stem}.png", width=W, height=H))
        for c, xc, yc, w, h in read_yolo(data_root / "labels" / split / f"{stem}.txt"):
            # potong kotak yang sedikit keluar dari tepi citra
            x0 = max(0.0, (xc - w / 2) * W)
            y0 = max(0.0, (yc - h / 2) * H)
            x1 = min(float(W), (xc + w / 2) * W)
            y1 = min(float(H), (yc + h / 2) * H)
            bw, bh = x1 - x0, y1 - y0
            anns.append(dict(id=ann_id, image_id=img_id, category_id=COCO_CAT_IDS[c],
                             bbox=[round(x0, 2), round(y0, 2), round(bw, 2), round(bh, 2)],
                             area=round(bw * bh, 2), iscrowd=0))
            ann_id += 1
    coco = dict(info=dict(description=f"TXL-PBC {split} (dikonversi dari YOLO)"),
                licenses=[], images=images, annotations=anns,
                categories=[dict(id=i, name=n) for i, n in zip(COCO_CAT_IDS, CLASSES)])
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(coco))
    return coco
