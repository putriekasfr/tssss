# Skenario Eksperimen T1-A (versi HEMAT)

**Pertanyaan riset:** apakah mAP tinggi menjamin hitungan sel yang akurat? Bandingkan YOLOv8s (one-stage) dengan
Faster R-CNN ResNet-50-FPN (two-stage) pada TXL-PBC (split resmi 882/252/126, seed 42).

## Daftar run

| Run | Model | Yang diubah | Pemilik | Milestone | GPU T4 (perkiraan) |
|---|---|---|---|---|---|
| R01 | YOLOv8s | baseline, imgsz 640 | Tsabita | M2 | ± 17 menit |
| R02 | Faster R-CNN | baseline, fine-tune penuh | Putri | M2 | ± 25 menit |
| R04 | YOLOv8s | ablasi A1: imgsz 832 | Tsabita | M3 | ± 25 menit |
| R01/R02-nms | keduanya | ablasi A2: IoU NMS 0,45 / 0,60 / 0,75, tanpa latih ulang | Putri | M3 | ± 10 menit |
| R06 | Faster R-CNN | backbone beku | Putri | M3 | ± 20 menit |
| R07, R08 | YOLOv8s terbaik (R01/R04) | seed 123, 2024 | Tsabita | M3 | ± 35 menit |
| R09, R10 | Faster R-CNN terbaik (R02/R06) | seed 123, 2024 | Putri | M3 | ± 50 menit |
| Uji, efisiensi, error, XAI | 6 run final | set uji dibuka sekali | Bersama | M3 | ± 15 menit |

Total GPU ± 3,3 jam, dibagi dua akun (masing-masing ± 1,5 jam).

## Penghematan dibanding proposal M1 (tulis di laporan sebagai perubahan dari rencana)

- YOLOv8s: 60 epoch dengan *early stopping* (patience 15). Proposal awal: 100 epoch.
- Faster R-CNN: 12 epoch, jadwal standar 1× (lr turun di epoch 8 dan 11), val dievaluasi mulai epoch 6. Proposal awal: 24 epoch.
- Setiap run berjalan di proses terpisah, sehingga VRAM dan RAM kosong lagi setelah run selesai.
- Log dicatat satu berkas per run (`experiments/log_runs/`), sehingga dua akun aman berjalan bersamaan.

## Cara menjalankan

**Persiapan (sekali):**
1. Ekstrak zip, lalu unggah **isi**nya ke folder `uas` di My Drive akun Tsabita. Isinya: `src/`, `configs/`,
   `notebooks/`, `VERSION`, dst.
2. Bagikan folder `uas` ke akun Putri sebagai **Editor**.
3. Di akun Putri, buka *Shared with me*, klik kanan `uas`, lalu pilih **Organize → Add shortcut → My Drive**.

**Tahap 1 – M2** (`notebooks/01_M2_Audit_EDA_Baseline.ipynb`):
1. Akun Tsabita: Runtime → T4 GPU. Di sel awal Bagian C isi `PELAKSANA = "Tsabita"`, lalu **Run all**.
   Bagian A, B dan R01 jalan.
2. Akun Putri: tunggu sampai Bagian B di akun Tsabita selesai (± 5 menit). Lalu T4 GPU,
   `PELAKSANA = "Putri"`, dan **Run all**. R02 jalan; R01 tidak ikut dijalankan.
3. Akun yang selesai duluan akan menampilkan "TUNDA: Bagian E menunggu …". Itu normal. Setelah kedua run selesai,
   salah satu akun menjalankan **Run all** sekali lagi: R01 dan R02 di-SKIP, lalu Bagian E dan F terisi.

**Tahap 2 – M3** (unggah `02_M3_...ipynb` ke `notebooks/` setelah M2 selesai): dijelaskan bersama berkas M3.

## Aturan aman

- Sesi Colab putus? **Restart session → Run all.** Run yang selesai di-SKIP, yang setengah jalan dilanjutkan dari checkpoint.
- Jangan menghapus atau memindahkan isi folder `uas` selama ada notebook yang berjalan.
- Hanya boleh ada **satu** folder versi HEMAT di Drive. Folder versi lama (FINAL) otomatis diabaikan.
