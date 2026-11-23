"""Faster R-CNN ResNet-50-FPN (torchvision): dataset, model, pelatihan dengan checkpoint, prediksi.

Dipakai oleh R02, R06, R09, R10 dan varian NMS R02-nms*. Semua pengaturan dibaca dari configs/<RUN>.yaml.
Checkpoint disimpan tiap epoch (last.pt) sehingga pelatihan bisa dilanjutkan bila sesi Colab putus.
"""
import contextlib
import io
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as TF

from .evaluate import coco_metrics
from .exp_log import set_seed


# ----------------------------------------------------------------------------- data
class CocoDetDataset(Dataset):
    """Membaca citra + anotasi COCO hasil konversi (outputs/coco/<split>.json)."""

    def __init__(self, img_dir, coco_json, train: bool, augment: dict = None):
        coco = json.loads(Path(coco_json).read_text())
        self.img_dir = Path(img_dir)
        self.images = coco["images"]
        self.anns = defaultdict(list)
        for a in coco["annotations"]:
            self.anns[a["image_id"]].append(a)
        self.train = train
        self.aug = augment or {}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        info = self.images[i]
        img = Image.open(self.img_dir / info["file_name"]).convert("RGB")
        W, H = img.size
        anns = self.anns[info["id"]]
        boxes = torch.tensor([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2],
                               a["bbox"][1] + a["bbox"][3]] for a in anns],
                             dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor([a["category_id"] for a in anns], dtype=torch.int64)

        if self.train and self.aug.get("hue", 0) > 0:
            img = TF.adjust_hue(img, random.uniform(-self.aug["hue"], self.aug["hue"]))
        img = TF.to_tensor(img)
        if self.train:
            if random.random() < self.aug.get("hflip", 0):
                img = img.flip(-1)
                boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
            if random.random() < self.aug.get("vflip", 0):
                img = img.flip(-2)
                boxes[:, [1, 3]] = H - boxes[:, [3, 1]]
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor(info["id"])}
        return img, target


def collate(batch):
    return tuple(zip(*batch))


def _seed_worker(worker_id):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


# ----------------------------------------------------------------------------- model
def build_model(cfg: dict, num_classes: int = 4):
    """Faster R-CNN pralatih COCO; kepala klasifikasi diganti 3 kelas + latar."""
    sizes = tuple((int(s),) for s in cfg["anchor_sizes"])          # satu ukuran per level FPN
    anchor = AnchorGenerator(sizes=sizes, aspect_ratios=((0.5, 1.0, 2.0),) * len(sizes))
    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
        trainable_backbone_layers=int(cfg["trainable_backbone_layers"]),
        rpn_anchor_generator=anchor,
        min_size=int(cfg["min_size"]), max_size=int(cfg["max_size"]),
        box_score_thresh=float(cfg["predict"]["conf"]),
        box_nms_thresh=float(cfg["predict"]["iou_nms"]),
        box_detections_per_img=int(cfg["predict"]["max_det"]),
    )
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    return model


def _set_train_sizes(model, cfg):
    model.transform.min_size = tuple(int(s) for s in cfg["train_min_sizes"])   # skala acak saat latih


def _set_eval_size(model, cfg):
    model.transform.min_size = (int(cfg["min_size"]),)                       # ukuran tetap saat inferensi


def _save_verified(obj, path, tries=3):
    """torch.save lalu pastikan berkas benar-benar ada dan tidak kosong (Drive bersama kadang gagal diam-diam)."""
    path = Path(path)
    for k in range(tries):
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(obj, tmp)
        tmp.replace(path)
        time.sleep(0.5)
        if path.exists() and path.stat().st_size > 1024:
            return True
        print(f"  PERINGATAN: {path.name} belum tersimpan (percobaan {k + 1}/{tries}), diulang...")
        time.sleep(2)
    print(f"  PERINGATAN: {path.name} gagal tersimpan setelah {tries} percobaan. Cek sisa penyimpanan Google Drive.")
    return False


def count_params(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


# ----------------------------------------------------------------------------- prediksi
@torch.no_grad()
def predict_coco(model, img_dir, coco_json, cfg, device="cuda") -> list:
    """Prediksi semua citra di coco_json -> list deteksi format COCO (piksel citra asli)."""
    model.eval()
    _set_eval_size(model, cfg)
    coco = json.loads(Path(coco_json).read_text())
    dets = []
    for info in coco["images"]:
        img = TF.to_tensor(Image.open(Path(img_dir) / info["file_name"]).convert("RGB")).to(device)
        out = model([img])[0]
        for (x0, y0, x1, y1), s, l in zip(out["boxes"].tolist(), out["scores"].tolist(),
                                          out["labels"].tolist()):
            dets.append({"image_id": info["id"], "category_id": int(l),
                         "bbox": [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)],
                         "score": round(float(s), 5)})
    return dets


def quiet_metrics(gt_json, dets):
    """coco_metrics tanpa mencetak tabel panjang pycocotools (dipakai tiap epoch)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return coco_metrics(gt_json, dets)


# ----------------------------------------------------------------------------- pelatihan
def train(cfg: dict, data_root, coco_dir, out_dir, device="cuda", log_every=50):
    """Latih Faster R-CNN sesuai cfg. Bisa dilanjutkan dari out_dir/last.pt.

    Model terbaik dipilih dengan fitness = 0.1*mAP50 + 0.9*mAP50-95 di set VALIDASI
    (rumus yang sama dengan pemilihan best.pt di ultralytics).
    Kembalikan DataFrame riwayat per epoch.
    """
    data_root, coco_dir, out_dir = Path(data_root), Path(coco_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(cfg["seed"]))

    ds = CocoDetDataset(data_root / "images" / "train", coco_dir / "train.json",
                        train=True, augment=cfg["augment"])
    g = torch.Generator()
    g.manual_seed(int(cfg["seed"]))
    dl = DataLoader(ds, batch_size=int(cfg["batch"]), shuffle=True, num_workers=2,
                    collate_fn=collate, worker_init_fn=_seed_worker, generator=g, drop_last=True)

    model = build_model(cfg).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=float(cfg["lr"]), momentum=float(cfg["momentum"]),
                          weight_decay=float(cfg["weight_decay"]))
    use_amp = bool(cfg.get("amp", True)) and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best, history = 0, -1.0, []
    last_path = out_dir / "last.pt"
    if last_path.exists():                                   # lanjutkan run yang terputus
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best, history = ck["epoch"] + 1, ck["best"], ck["history"]
        print(f"Melanjutkan dari epoch {start_epoch}")

    epochs, warm = int(cfg["epochs"]), int(cfg.get("warmup_iters", 0))
    steps = [int(s) for s in cfg.get("lr_steps", [])]
    base_lr = float(cfg["lr"])
    it_per_epoch = len(dl)
    gt_val = coco_dir / "val.json"

    for epoch in range(start_epoch, epochs):
        model.train()
        _set_train_sizes(model, cfg)
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        t0, run_loss = time.time(), 0.0
        for i, (imgs, targets) in enumerate(dl):
            it = epoch * it_per_epoch + i
            lr = base_lr * (0.1 ** sum(epoch >= s for s in steps))
            if it < warm:
                lr *= 0.001 + (1 - 0.001) * it / warm
            for gp in opt.param_groups:
                gp["lr"] = lr
            imgs = [im.to(device) for im in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast("cuda", enabled=use_amp):
                losses = model(imgs, targets)
                loss = sum(losses.values())
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"Loss tidak hingga di epoch {epoch}: {losses}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item()
            if (i + 1) % log_every == 0:
                print(f"  epoch {epoch + 1} iter {i + 1}/{it_per_epoch} loss {run_loss / (i + 1):.4f} lr {lr:.5f}")

        peak_gb = torch.cuda.max_memory_reserved() / 1024 ** 3 if device.startswith("cuda") else 0.0
        del imgs, targets, losses, loss                       # lepas tensor batch terakhir sebelum evaluasi
        if epoch + 1 >= int(cfg.get("eval_from", 1)) or epoch + 1 == epochs:
            dets = predict_coco(model, data_root / "images" / "val", gt_val, cfg, device)
            m = quiet_metrics(gt_val, dets) if dets else {"mAP50": 0.0, "mAP50_95": 0.0}
            del dets
        else:                                                 # HEMAT: epoch awal tidak dievaluasi (bukan kandidat best.pt)
            m = {"mAP50": float("nan"), "mAP50_95": float("nan")}
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        fit = 0.1 * m["mAP50"] + 0.9 * m["mAP50_95"]
        row = dict(epoch=epoch + 1, train_loss=run_loss / it_per_epoch, val_mAP50=m["mAP50"],
                   val_mAP50_95=m["mAP50_95"], fitness=fit, lr=lr, minutes=(time.time() - t0) / 60,
                   gpu_peak_gb=round(peak_gb, 2))
        history.append(row)
        print(f"epoch {epoch + 1}/{epochs}  loss {row['train_loss']:.4f}  "
              f"val mAP50 {m['mAP50']:.4f}  mAP50-95 {m['mAP50_95']:.4f}  "
              f"GPU puncak {peak_gb:.1f} GB  ({row['minutes']:.1f} mnt)")
        if fit == fit and fit > best:                          # fit NaN (epoch tidak dievaluasi) dilewati
            best = fit
            _save_verified(model.state_dict(), out_dir / "best.pt")
        _save_verified(dict(model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(),
                            epoch=epoch, best=best, history=history), last_path)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    return pd.DataFrame(history)


def load_best(cfg, out_dir, device="cuda"):
    """Muat bobot epoch terbaik (best.pt). Bila best.pt hilang (gagal tersimpan di Drive), pakai bobot epoch
    terakhir dari last.pt dan tandai `model.bobot_dipakai = "last"` agar dicatat di log dan laporan."""
    model = build_model(cfg).to(device)
    best_p, last_p = Path(out_dir) / "best.pt", Path(out_dir) / "last.pt"
    model.bobot_dipakai = "best"
    try:
        model.load_state_dict(torch.load(best_p, map_location=device))
    except (FileNotFoundError, RuntimeError, EOFError) as e:
        if not last_p.exists():
            raise
        ck = torch.load(last_p, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        model.bobot_dipakai = f"last (epoch {ck['epoch'] + 1}; best.pt tidak terbaca: {type(e).__name__})"
        print(f"PERINGATAN: {best_p} tidak terbaca → memakai bobot epoch terakhir (epoch {ck['epoch'] + 1}) dari last.pt.")
        _save_verified(ck["model"], best_p)              # pulihkan best.pt agar langkah berikutnya konsisten
    model.eval()
    _set_eval_size(model, cfg)
    return model
