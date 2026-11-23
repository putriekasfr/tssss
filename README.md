# T1-A · mAP vs Galat Hitung Sel: YOLOv8s vs Faster R-CNN pada TXL-PBC

Tugas Besar Pengganti UAS · Deep Learning for Image – Computer Vision Lanjutan · Kelas DL-A
Tsabita Shafa Aurellia (240441100052) · Putri Eka Safira (240441100087)

**Pertanyaan riset (Daftar Topik T1-A):** seberapa kuat korelasi antara mAP@[0.5:0.95] dan galat hitung sel per citra
(MAE/MAPE per kelas) pada detektor one-stage (YOLOv8s) dan two-stage (Faster R-CNN ResNet-50-FPN)? Keduanya pralatih
COCO dan dinilai dengan satu alat evaluasi (pycocotools); korelasi Pearson/Spearman dihitung antar run.

> Model ini adalah prototipe riset, **bukan alat diagnosis**.

## Struktur repositori

```
configs/        konfigurasi tiap run (R01–R10); `inherits` menyalin run lain lalu mengubah SATU faktor
splits/         daftar berkas split resmi TXL-PBC (train 882 / val 252 / test 126)
src/            kode bersama: data_utils, leakage (pHash), evaluate, errors, xai (EigenCAM), efficiency, summary,
                yolo_utils, frcnn, pipeline, exp_log, viz (palet grafik)
notebooks/      01_M2_Audit_EDA_Baseline · 02_M3_Main_Experiments_Ablations_XAI
experiments/    log.csv (semua run, termasuk yang gagal) dan env/<RUN>.txt (versi pustaka + GPU)
outputs/        figures/ (semua grafik), eda/ & leakage/ (tabel), preds/, eval/
reports/        laporan kemajuan M2, laporan final M3
```

## Menjalankan di Google Colab (gratis, GPU T4)

1. Ikuti **`SKENARIO.md`**: unggah isi zip ke satu folder di My Drive (mis. `uas`), bagikan ke anggota lain (shortcut).
   Notebook hanya memakai folder yang berkas `VERSION`-nya `HEMAT-2026-11-23`; folder versi lama diabaikan.
2. Buka notebook dari Drive di Colab (klik dua kali → *Open with → Google Colaboratory*). Untuk bagian yang
   melatih model: **Runtime → Change runtime type → T4 GPU**.
3. Sel *Setup* di setiap notebook akan: me-mount Drive, menemukan folder proyek, mengunduh dataset dari GitHub
   resmi ke `/content`, dan memasang paket yang belum ada di Colab (`openpyxl`, `ultralytics`, `pycocotools`).
4. Notebook yang dibuka (hanya dua, satu per milestone):

| Notebook | Isi per bagian | GPU |
|---|---|---|
| `notebooks/01_M2_Audit_EDA_Baseline.ipynb` | A audit & EDA (Tsabita) · B split, COCO, kebocoran (Putri) · C baseline R01 YOLOv8s (Tsabita) · D baseline R02 Faster R-CNN (Putri) · E evaluasi terpadu (bersama) · F log | T4 untuk C, D |
| `notebooks/02_M3_Main_Experiments_Ablations_XAI.ipynb` | 0 setup & panel kendali · 1 persiapan + cek kebocoran (Putri) · 2.1 resolusi 640 vs 832 (Tsabita) · 2.2 ambang IoU NMS tanpa latih ulang (Putri) · 2.3 backbone beku (Putri) · 2.4 tabel & grafik ablasi (Tsabita) · 3.1 YOLOv8s 3 seed (Tsabita) · 3.2 Faster R-CNN 3 seed (Putri) · 3.3 korelasi mAP–MAE (Tsabita) · 3.4 set uji sekali + uji statistik + per ukuran (Putri) · 4 efisiensi (Tsabita) · 5.1 analisis kesalahan (Putri) · 5.2 EigenCAM (Tsabita) · 6 ringkasan, hipotesis, arsip zip (bersama) | T4 untuk 2.1–2.3, 3.1, 3.2, 3.4, 4, 5.2 |

Di notebook M2, saklar `JALANKAN_R01` / `JALANKAN_R02` menentukan baseline yang dilatih di sesi itu. Notebook M3
berjalan **otomatis seperti notebook referensi**: Run all menjalankan semua pekerjaan berurutan, melewati yang sudah
selesai (`SKIP`), dan melanjutkan dari checkpoint bila sesi putus; `PELAKSANA = "Tsabita"` / `"Putri"` membatasi
pekerjaan ke milik satu anggota agar dapat dikerjakan bergantian dari akun masing-masing. Setiap pekerjaan GPU M3
dijalankan di proses Python terpisah (`src/run_job.py`) agar VRAM dan RAM kosong kembali setelah tiap run; log
lengkapnya ada di `experiments/proses/`. Semua grafik tersimpan di
`outputs/figures/` (Gambar 1–8 dari M2, Gambar 9–19 dari M3); arsip ringkas M3 di `outputs/m3_results_bundle.zip`.

Semua run memakai langkah yang sama (`src/pipeline.py`). Bila sesi Colab putus,
jalankan ulang sel yang sama: pelatihan dilanjutkan dari `runs/<RUN>/` di Drive.

## Rencana run (menu ablasi T1-A)

| Run | Model | Faktor yang diubah | Seed | Pemilik |
|---|---|---|---|---|
| R01 | YOLOv8s | baseline, imgsz 640, NMS 0,70 | 42 | Tsabita |
| R02 | Faster R-CNN R50-FPN | baseline, fine-tune penuh, NMS 0,50 | 42 | Putri |
| R04 | YOLOv8s | **A1** resolusi 640 → 832 | 42 | Tsabita |
| R01-nms0.45 / 0.60 / 0.75, R02-nms0.45 / 0.60 / 0.75 | bobot R01 & R02 | **A2** ambang IoU NMS, tanpa latih ulang | 42 | Putri |
| R06 | Faster R-CNN | backbone beku | 42 | Putri |
| R07, R08 | YOLOv8s terbaik (R01/R04, validasi) | — | 123, 2024 | Tsabita |
| R09, R10 | Faster R-CNN terbaik (R02/R06, validasi) | — | 123, 2024 | Putri |

Aturan: satu faktor per perbandingan; konfigurasi terbaik = val mAP@[.5:.95] tertinggi; set uji dibuka satu kali di M3
(`BUKA_SET_UJI`) untuk 6 run final.

## Keputusan data penting (hasil audit M2)

- **Label duplikat:** 47 baris identik di 5 berkas label dibuang (18.143 → 18.096 kotak). Ultralytics membuangnya
  otomatis; ground truth COCO dibuat sama agar kedua model dinilai terhadap label yang identik.
- **Kebocoran:** 0 pasangan mirip (pHash Hamming ≤ 8) antara latih dan uji; 2 pasangan latih–validasi (Raabin-WBC)
  dicatat. Split resmi tidak diubah.
- **Augmentasi sama untuk kedua model:** flip horizontal & vertikal, skala ±20%, hue ±0,01; mosaic YOLO dimatikan.

## Dataset dan sitasi

TXL-PBC (lisensi MIT) tidak disertakan di repo ini; notebook mengunduhnya dari
<https://github.com/lugan113/TXL-PBC_Dataset>.

> Gan, L., Li, X., & Wang, X. (2025). A curated and re-annotated peripheral blood cell dataset integrating four public
> resources. *Scientific Data*, 12, 1694. https://doi.org/10.1038/s41597-025-05980-z

Pustaka: ultralytics (YOLOv8), torchvision (Faster R-CNN), pycocotools. Versi persis tiap run tercatat di
`experiments/env/`.

## Penggunaan AI generatif

Claude (Anthropic) dipakai untuk membantu menulis kerangka kode dan notebook. Desain eksperimen, analisis hasil,
dan penafsiran kesalahan dikerjakan tim. *(Lengkapi di laporan final sesuai kisi-kisi 6.2.)*
