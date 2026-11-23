"""Gaya grafik bersama: warna tetap per entitas (kelas, sumber, split, model) di semua gambar.

Palet kategorikal divalidasi untuk buta warna (dataviz validator, mode terang):
kelas = 3 slot pertama (lolos semua pasangan); sumber = 4 slot berikutnya (lolos pasangan bersebelahan).
"""
import matplotlib.pyplot as plt

CLASS_COLORS = {"WBC": "#2a78d6", "RBC": "#eb6834", "Platelets": "#1baf7a"}
SOURCE_COLORS = {"BCCD": "#eda100", "BCDD": "#e87ba4", "PBC": "#008300", "Raabin-WBC": "#6250d6"}
SPLIT_COLORS = {"train": "#0e8aa0", "val": "#e34948", "test": "#7a4fd0"}       # teal, merah, ungu (lolos cek buta warna)
MODEL_COLORS = {"YOLOv8s": "#6250d6", "Faster R-CNN": "#eda100"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
BAR = dict(edgecolor="white", linewidth=1.5)          # celah 2 px antar batang


def style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlepad": 8,
        "axes.labelcolor": INK_2, "axes.edgecolor": "#b5b4ad", "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 9,
    })


def label_bars(ax, bars, fmt="{:.1f}", pad=2, fontsize=8):
    """Label nilai di ujung batang (warna teks netral, bukan warna seri)."""
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h), xytext=(0, pad),
                    textcoords="offset points", ha="center", va="bottom", fontsize=fontsize, color=INK_2)
