"""Interpretabilitas dengan EigenCAM (Muhammad & Yeasin, 2020) untuk YOLOv8 dan Faster R-CNN.

EigenCAM tidak memakai gradien: peta panas = proyeksi aktivasi satu lapisan ke komponen utama
pertamanya. Karena bebas gradien, metode yang sama bisa dipakai untuk kedua detektor (perbandingan adil).
Lapisan yang dipakai memiliki stride 8 pada keduanya:
  - YOLOv8s      : keluaran neck P3 (masukan pertama kepala Detect)
  - Faster R-CNN : keluaran FPN level '1' (P3)

Selain gambar, dihitung porsi energi peta panas di dalam kotak tiap kelas vs latar, agar tafsiran
("model memperhatikan sel atau latar?") didukung angka.
"""
import numpy as np
from PIL import Image


def eigen_projection(act: np.ndarray) -> np.ndarray:
    """act: (C, h, w) -> peta (h, w) dari komponen utama pertama, tanda disamakan dengan rerata aktivasi."""
    C, h, w = act.shape
    X = act.reshape(C, -1).T.astype(np.float64)          # (hw, C)
    X = X - X.mean(0, keepdims=True)
    _, _, VT = np.linalg.svd(X, full_matrices=False)
    proj = (X @ VT[0]).reshape(h, w)
    ref = np.abs(act).mean(0)
    if np.corrcoef(proj.ravel(), ref.ravel())[0, 1] < 0:  # SVD punya ambiguitas tanda
        proj = -proj
    proj = np.maximum(proj, 0)
    return proj / (proj.max() + 1e-9)


def _resize(cam, W, H):
    return np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR), dtype=np.float32) / 255


class _Hook:
    def __init__(self, module):
        self.out = None
        self.h = module.register_forward_hook(self._f)

    def _f(self, m, i, o):
        self.out = o

    def remove(self):
        self.h.remove()


_NET_CACHE = {}


def clear_cache():
    """Lepas model YOLO yang di-cache EigenCAM dari VRAM."""
    import gc
    _NET_CACHE.clear()
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass


def _yolo_net(weights, device):
    """Muat model YOLO sekali saja per bobot (bukan sekali per gambar) agar VRAM tidak menumpuk."""
    key = (str(weights), device)
    if key not in _NET_CACHE:
        from ultralytics import YOLO
        _NET_CACHE.clear()                                    # hanya satu model di VRAM pada satu waktu
        _NET_CACHE[key] = YOLO(str(weights)).model.to(device).float().eval()
    return _NET_CACHE[key]


def yolo_eigencam(weights, img_path, imgsz=640, device="cuda"):
    """EigenCAM YOLOv8 pada citra asli. Letterbox dibuat sendiri agar pemetaan balik tepat."""
    import torch
    net = _yolo_net(weights, device)
    det = net.model[-1]
    hook = _Hook(net.model[det.f[0]])                       # P3
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    r = imgsz / max(W, H)
    nw, nh = round(W * r), round(H * r)
    pw, ph = int(np.ceil(nw / 32) * 32), int(np.ceil(nh / 32) * 32)
    canvas = Image.new("RGB", (pw, ph), (114, 114, 114))
    ox, oy = (pw - nw) // 2, (ph - nh) // 2
    canvas.paste(im.resize((nw, nh), Image.BILINEAR), (ox, oy))
    x = torch.from_numpy(np.asarray(canvas, dtype=np.float32) / 255).permute(2, 0, 1)[None].to(device)
    with torch.no_grad():
        net(x)
    act = hook.out[0] if isinstance(hook.out, (list, tuple)) else hook.out
    hook.remove()
    act = act[0].float().cpu().numpy()                       # pindah ke CPU, lepas tensor GPU
    del x, hook
    cam = eigen_projection(act)
    cam = _resize(cam, pw, ph)[oy:oy + nh, ox:ox + nw]       # buang padding letterbox
    return _resize(cam, W, H)


def frcnn_eigencam(model, img_path, level="1", device="cuda"):
    """EigenCAM Faster R-CNN dari keluaran FPN (level '1' = stride 8)."""
    import torch
    from torchvision.transforms import functional as TF
    model.eval()
    hook = _Hook(model.backbone)
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    with torch.no_grad():
        model([TF.to_tensor(im).to(device)])
    feats = hook.out
    hook.remove()
    act = feats[level][0].float().cpu().numpy()             # (C, h, w) untuk citra yang sudah di-resize & dipad
    del feats, hook
    t = model.transform
    s = min(float(t.min_size[-1]) / min(W, H), float(t.max_size) / max(W, H))
    rh, rw = int(np.floor(H * s)), int(np.floor(W * s))       # ukuran sebelum padding (kiri-atas)
    stride = 2 ** (int(level) + 2)
    cam = eigen_projection(act)
    full = _resize(cam, act.shape[2] * stride, act.shape[1] * stride)[:rh, :rw]
    return _resize(full, W, H)


def energy_share(cam: np.ndarray, boxes_by_class: dict) -> dict:
    """Porsi energi peta panas di dalam kotak tiap kelas dan di latar (jumlah = 1)."""
    H, W = cam.shape
    total = cam.sum() + 1e-9
    owner = np.zeros((H, W), dtype=np.int8)                  # 0 = latar; kelas kecil menimpa kelas besar
    names = list(boxes_by_class)
    for k, c in enumerate(names, 1):
        for x, y, w, h in boxes_by_class[c]:
            owner[int(max(0, y)):int(min(H, y + h)), int(max(0, x)):int(min(W, x + w))] = k
    out = {c: float(cam[owner == k].sum() / total) for k, c in enumerate(names, 1)}
    out["latar"] = float(cam[owner == 0].sum() / total)
    out["luas_latar"] = float((owner == 0).mean())
    return out


def overlay(img_path, cam, alpha=0.45):
    import matplotlib.cm as cm
    im = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255
    heat = cm.jet(cam)[..., :3]
    return np.clip((1 - alpha) * im + alpha * heat, 0, 1)
