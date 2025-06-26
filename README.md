# Back-Banjay-AI

Back-Banjay-AI adalah backend analisis risiko banjir berbasis data spasial, citra satelit, dan literatur ilmiah. Sistem ini menggabungkan computer vision, retrieval augmented generation (RAG), dan Large Language Model (LLM) untuk memberikan analisis risiko banjir yang komprehensif di wilayah Indonesia.

## Fitur Utama

- **Ekstraksi Lokasi Otomatis:** Mengidentifikasi lokasi dari input skenario pengguna menggunakan LLM.
- **Analisis Citra Satelit:** Mengunduh dan menganalisis gambar satelit Himawari-9 untuk mendeteksi jenis dan cakupan awan di wilayah tertentu.
- **Retrieval Augmented Generation:** Mengambil konteks ilmiah dari dokumen PDF dan data historis untuk memperkuat analisis.
- **Analisis Ahli Otomatis:** Menghasilkan laporan risiko banjir terstruktur berdasarkan data cuaca, historis, dan kondisi lahan.
- **Caching & Indexing:** Mendukung cache embedding dan dokumen untuk efisiensi proses.

## Struktur Direktori

```
.
├── app/
│   ├── main.py           # Alur utama backend dan endpoint analisis
│   ├── app_citra.py      # Modul analisis citra satelit & RAG cuaca
│   ├── app_tanah.py      # Modul analisis risiko banjir berbasis lahan
│   └── ...
├── data/
│   ├── raw/              # Data mentah (PDF, JSON, Excel)
│   └── processed/        # Cache embedding & dokumen
├── requirements.txt      # Daftar dependensi Python
├── Dockerfile            # Konfigurasi Docker (opsional)
├── README.md
└── ...
```

## Cara Menjalankan

1. **Instalasi Dependensi**
   ```
   pip install -r requirements.txt
   ```

2. **Struktur Data**
   - Letakkan file PDF (misal: `cuaca.pdf`, `tanah.pdf`) di `data/raw/`
   - Data historis banjir: `jumlah_kejadians.json` di `data/raw/`

3. **Menjalankan Analisis**
   Jalankan backend dari `app/main.py`:
   ```
   python app/main.py
   ```
   Ikuti instruksi untuk memasukkan skenario/pertanyaan.

4. **Output**
   Hasil analisis akan disimpan dalam file JSON (misal: `output1.json`) dengan struktur:
   - `location_info`
   - `cloud_coverage_summary`
   - `satellite_analysis`
   - `expert_context`
   - `final_report`

## Dependensi Utama

- Python 3.8+
- [sentence-transformers](https://www.sbert.net/)
- [faiss](https://github.com/facebookresearch/faiss)
- [transformers](https://huggingface.co/transformers/)
- [PyPDF2](https://pypi.org/project/PyPDF2/)
- [Pillow](https://pillow.readthedocs.io/)
- [requests](https://docs.python-requests.org/)

## Catatan

- API key Mistral dan endpoint sudah dikonfigurasi di kode (`main.py`, `app_citra.py`, `app_tanah.py`).
- Untuk penggunaan GPU, pastikan PyTorch dan CUDA sudah terpasang.

## Lisensi

Proyek ini dikembangkan untuk riset dan edukasi. Silakan modifikasi sesuai kebutuhan.

---

**Back-Banjay-AI** – Analisis risiko banjir berbasis AI, citra satelit, dan data