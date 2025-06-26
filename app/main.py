# app_analisis_computer_vision.py

import os
import re
import json
import faiss
import numpy as np
import PyPDF2
import requests
import pickle
import torch
from tqdm import tqdm
from PIL import Image
from sentence_transformers import SentenceTransformer

# ==============================================================================
# === KONFIGURASI UTAMA (DENGAN STRUKTUR FOLDER TERORGANISIR) ===
# ==============================================================================

MISTRAL_API_KEY = "ArKWofu383ebzKYXakZB8RF8Clgx5hv6" 
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MODEL_EKSTRAKSI = "mistral-small-2503"
MODEL_CITRA = "mistral-small-2503"
MODEL_AHLI = "mistral-large-latest"

# === Path file input/output yang telah diorganisir ===
DATA_RAW = "data/raw"
DATA_PROCESSED = "data/processed"

PDF_CUACA = os.path.join(DATA_RAW, "cuaca.pdf")
PDF_TANAH = os.path.join(DATA_RAW, "tanah.pdf")
JSON_DATA_LATIH = os.path.join(DATA_RAW, "merged_training_data.json")
JSON_RIWAYAT_BANJIR = os.path.join(DATA_RAW, "jumlah_kejadian.json")

CACHE_DOKUMEN_AHLI = os.path.join(DATA_PROCESSED, "cached_docs_ahli.pkl")
CACHE_EMBED_AHLI = os.path.join(DATA_PROCESSED, "cached_embeddings_ahli.npy")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


IMAGE_URL = "https://inderaja.bmkg.go.id/IMAGE/HCAI/CLC/HCAI_CLC_Indonesia.png"

def download_image(url, save_path):
    """Download gambar dari URL dan simpan ke save_path."""
    try:
        # Pastikan folder tujuan ada
        folder = os.path.dirname(os.path.abspath(save_path))
        os.makedirs(folder, exist_ok=True)
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        return save_path
    except Exception as e:
        print(f"❌ Error download gambar: {e}")
        return None
# ==============================================================================
# === BAGIAN COMPUTER VISION & ANALISIS CITRA (PERUBAHAN BESAR) ===
# ==============================================================================

# Definisikan batas geografis dari gambar satelit Anda
# Berdasarkan gambar: Longitude 90E - 145E, Latitude 10N - 15S
# ==============================================================================
# === BAGIAN COMPUTER VISION & ANALISIS CITRA (DISempurnakan) ===
# ==============================================================================

# Definisikan batas geografis dari gambar satelit Anda
# Berdasarkan gambar: Longitude 90E - 145E, Latitude 10N - 15S
IMAGE_BOUNDS = {
    "lat_max": 10.0,      # Batas atas peta (10° Lintang Utara)
    "lat_min": -15.0,     # Batas bawah peta (15° Lintang Selatan)
    "lon_max": 145.0,     # Batas kanan peta (145° Bujur Timur)
    "lon_min": 90.0       # Batas kiri peta (90° Bujur Timur)
}

# Definisikan peta warna dari legenda di gambar ke jenis awan
# Warna ini diambil secara manual dari legenda gambar Anda
COLOR_TO_CLOUD_TYPE = {
    (255, 0, 0): "Cumulonimbus",
    (87, 87, 87): "Dense",
    (132, 212, 255): "High Cloud",
    (96, 153, 198): "Middle Cloud",
    (255, 178, 79): "Cumulus",
    (170, 255, 0): "Stratocumulus",
    (0, 255, 0): "Stf/Fog",
    (0, 0, 0): "Clear"
}

def latlon_to_pixel(lat, lon, image_width, image_height, bounds):
    """Mengubah koordinat geografis (lat, lon) menjadi koordinat piksel (x, y)."""
    if not (bounds["lon_min"] <= lon <= bounds["lon_max"] and bounds["lat_min"] <= lat <= bounds["lat_max"]):
        return None, None # Koordinat di luar jangkauan peta
        
    lon_frac = (lon - bounds["lon_min"]) / (bounds["lon_max"] - bounds["lon_min"])
    lat_frac = (bounds["lat_max"] - lat) / (bounds["lat_max"] - bounds["lat_min"])
    
    x = int(lon_frac * image_width)
    y = int(lat_frac * image_height)
    
    x = max(0, min(x, image_width - 1))
    y = max(0, min(y, image_height - 1))
    
    return x, y

def find_closest_color(rgb_tuple, color_map):
    """Mencari warna terdekat dari peta warna menggunakan jarak Euclidean."""
    r1, g1, b1 = rgb_tuple
    # Abaikan piksel putih (garis pantai/batas negara) untuk menghindari misinterpretasi
    if r1 > 220 and g1 > 220 and b1 > 220:
        return "Coastline/Border"
        
    min_dist = float('inf')
    closest_type = "Unknown"
    
    for color, cloud_type in color_map.items():
        r2, g2, b2 = color
        dist = np.sqrt((r1 - r2)**2 + (g1 - g2)**2 + (b1 - b2)**2)
        if dist < min_dist:
            min_dist = dist
            closest_type = cloud_type
            
    return closest_type

def analyze_cloud_coverage_in_region(image_path, lat, lon, region_radius_deg=0.5):
    """
    METODE BARU: Menganalisis cakupan awan dalam sebuah kotak di sekitar koordinat.
    Mengembalikan ringkasan persentase jenis awan di wilayah tersebut.
    """
    try:
        with Image.open(image_path) as img:
            img_rgb = img.convert("RGB")
            width, height = img.size
            
            # Tentukan Bounding Box (kotak area) untuk dianalisis
            lat_max_box, lat_min_box = lat + region_radius_deg, lat - region_radius_deg
            lon_max_box, lon_min_box = lon + region_radius_deg, lon - region_radius_deg

            # Konversi sudut-sudut kotak geografis ke koordinat piksel
            x_start, y_start = latlon_to_pixel(lat_max_box, lon_min_box, width, height, IMAGE_BOUNDS)
            x_end, y_end = latlon_to_pixel(lat_min_box, lon_max_box, width, height, IMAGE_BOUNDS)

            # Cek jika koordinat berada di luar peta
            if x_start is None:
                print(f"    -> Peringatan: Koordinat ({lat}, {lon}) berada di luar jangkauan peta satelit.")
                return {"Error": "Koordinat di luar jangkauan citra satelit."}

            print(f"    -> Menganalisis wilayah di sekitar ({lat}, {lon})")
            print(f"    -> Kotak Piksel yang Dianalisis: ({x_start},{y_start}) hingga ({x_end},{y_end})")

            cloud_counts = {}
            total_pixels = 0
            
            # Ambil sampel piksel di dalam kotak (sampling setiap 3 piksel agar lebih cepat)
            for x in range(x_start, x_end, 3):
                for y in range(y_start, y_end, 3):
                    pixel_color = img_rgb.getpixel((x, y))
                    cloud_type = find_closest_color(pixel_color, COLOR_TO_CLOUD_TYPE)
                    
                    if cloud_type not in ["Unknown", "Coastline/Border"]:
                        cloud_counts[cloud_type] = cloud_counts.get(cloud_type, 0) + 1
                        total_pixels += 1
            
            if total_pixels == 0:
                return {"Analysis": "Tidak ada data awan valid di wilayah tersebut (kemungkinan besar lautan atau daratan tanpa awan)."}

            # Konversi jumlah hitungan ke persentase
            cloud_percentages = {k: round((v / total_pixels) * 100, 1) for k, v in cloud_counts.items()}
            
            # Urutkan berdasarkan persentase tertinggi
            sorted_coverage = dict(sorted(cloud_percentages.items(), key=lambda item: item[1], reverse=True))
            return sorted_coverage
            
    except FileNotFoundError:
        print(f"❌ Error: File gambar satelit tidak ditemukan di '{image_path}'")
        return {"Error": "File gambar tidak ditemukan."}
    except Exception as e:
        print(f"❌ Error saat memproses analisis regional gambar: {e}")
        return {"Error": f"Terjadi kesalahan saat analisis gambar: {e}"}

def analyze_cloud_type_with_llm(cloud_coverage_data, location_info):
    """
    Fungsi yang Diperbarui: Menganalisis data cakupan awan (bukan hanya 1 jenis) dengan LLM.
    """
    if "Error" in cloud_coverage_data or not cloud_coverage_data:
        return f"Tidak dapat melakukan analisis meteorologi: {cloud_coverage_data.get('Error', 'Data tidak tersedia.')}"
    
    # Ubah data dictionary menjadi string yang mudah dibaca untuk prompt
    coverage_summary = ", ".join([f"{cloud}: {percent}%" for cloud, percent in cloud_coverage_data.items()])

    prompt = f"""
    **Peran:** Anda adalah seorang ahli meteorologi BMKG.
    
    **Tugas:** Berikan analisis cuaca singkat dan jelas berdasarkan data persentase tutupan awan yang teridentifikasi di suatu wilayah.
    
    **Data Input:**
    - **Lokasi:** {location_info['name']} (Lat: {location_info['lat']}, Lon: {location_info['lon']})
    - **Data Cakupan Awan Terdeteksi:** {coverage_summary}

    **Instruksi:**
    1.  Identifikasi jenis awan yang paling dominan dan jelaskan karakteristik utamanya (1-2 kalimat).
    2.  Berdasarkan komposisi awan yang ada, jelaskan potensi cuaca signifikan yang paling mungkin terjadi di lokasi tersebut (misalnya: hujan lebat, badai guntur, cuaca cerah, dll.).
    3.  Berikan kesimpulan dalam format ringkas.

    **Analisis Meteorologis Anda:**
    """
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL_CITRA, "messages": [{"role": "user", "content": prompt.strip()}], "temperature": 0.3}
    try:
        response = requests.post(MISTRAL_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Gagal menganalisis data awan dengan LLM: {e}"

# ==============================================================================
# === FUNGSI LAINNYA (Ekstraksi LLM dan Analisis Ahli) ===
# ==============================================================================

def extract_location_with_llm(text: str) -> dict:
    default_info = {"name": "Tidak Diketahui", "lat": "Tidak Tersedia", "lon": "Tidak Tersedia", "subdistrict": None}
    prompt = (
        "Anda adalah alat ekstraksi data JSON. Dari teks berikut, ekstrak 'location_name', 'subdistrict', 'latitude', dan 'longitude'. "
        "Jika tidak ada, gunakan null.\nTeks: \"{text}\"\nJSON Output Anda:"
    ).format(text=text)
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL_EKSTRAKSI,
        "messages": [{"role": "user", "content": prompt.strip()}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }
    try:
        response = requests.post(MISTRAL_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        json_content_str = response.json()["choices"][0]["message"]["content"]
        extracted_data = json.loads(json_content_str)
        location_info = {
            "name": extracted_data.get("location_name") or extracted_data.get("name") or default_info["name"],
            "lat": str(extracted_data.get("latitude")) if extracted_data.get("latitude") is not None else default_info["lat"],
            "lon": str(extracted_data.get("longitude")) if extracted_data.get("longitude") is not None else default_info["lon"],
            "subdistrict": extracted_data.get("subdistrict") or extracted_data.get("kecamatan") or None
        }
        return location_info
    except Exception as e:
        print(f"❌ Error saat ekstraksi lokasi dengan LLM: {e}")
        return default_info

def load_and_process_documents(pdf_files, json_path, history_path, location_name):
    # Fungsi ini tidak diubah
    all_docs = []
    try:
        with open(json_path, "r", encoding="utf-8") as f: all_docs.extend([item["response"] for item in json.load(f)])
    except FileNotFoundError: pass
    for file_path in pdf_files:
        try:
            pdf = PyPDF2.PdfReader(file_path)
            content = " ".join([p.extract_text() for p in pdf.pages if p.extract_text()])
            chunks = re.split(r'\. ', content.replace('\n', ' '))
            all_docs.extend([c.strip() for c in chunks if 50 < len(c.strip()) < 500])
        except FileNotFoundError: pass
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            for row in json.load(f):
                nama_kab = row.get("Nama Kabupaten/Kota", "")
                if location_name and location_name.lower() in nama_kab.lower():
                    riwayat = ", ".join([f"{thn}: {row[str(thn)]} kejadian" for thn in range(2010, 2026) if str(thn) in row and row[str(thn)] is not None])
                    if riwayat: all_docs.append(f"Data historis untuk {nama_kab} menunjukkan riwayat banjir: {riwayat}.")
    except FileNotFoundError: pass
    return all_docs

def get_final_analysis(main_scenario, supporting_context, satellite_analysis, final_question):
    # Fungsi ini tidak diubah
    prompt = f"""
    **Peran:** Anda adalah seorang ahli hidrologi. Berikan ringkasan analisis risiko (Executive Summary).
    **Instruksi:** Tulis kesimpulan ringkas dengan format terstruktur berikut.
    **1. Kesimpulan Risiko:** (Nyatakan tingkat risiko: Rendah, Sedang, Tinggi, Sangat Tinggi).
    **2. Skor Risiko:** (Skor 0-100).
    **3. Faktor Peningkat Risiko:**
    - (Sebutkan 1-3 faktor utama dengan data dalam kurung).
    **4. Faktor Pereda Risiko:**
    - (Sebutkan 1-2 faktor utama dengan data dalam kurung).
    **5. Analisis Sintesis (Interaksi Cuaca dan Lahan):**
    - (Jelaskan bagaimana interaksi antara **Ringkasan Cuaca** dan **kondisi lahan** menentukan potensi banjir. Apakah tanah mampu menyerap potensi hujan?).
    **6. Rekomendasi Akhir:** (Satu kalimat jawaban untuk Pertanyaan Kunci).
    
    ---
    **Sumber Informasi:**
    1. Skenario: {main_scenario}
    2. Cuaca: {satellite_analysis}
    3. Konteks: {supporting_context}
    4. Pertanyaan: {final_question}
    ---
    **KESIMPULAN ANALISIS AHLI:**
    """
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL_AHLI, "messages": [{"role": "user", "content": prompt.strip()}], "temperature": 0.4, "max_tokens": 1024}
    try:
        response = requests.post(MISTRAL_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Gagal menghasilkan analisis akhir: {e}"

# ==============================================================================
# === ALUR KERJA UTAMA (MAIN) - Dengan Computer Vision ===
# ==============================================================================

def run_full_analysis(user_provided_scenario):
    """
    Fungsi utama untuk backend: menerima input teks skenario,
    mengembalikan hasil analisis sebagai dictionary.
    Gambar satelit otomatis diunduh dari BMKG.
    """
    final_question = "Apakah lokasi ini berisiko tinggi terhadap kejadian banjir genangan?"

    # TAHAP 0: Ekstraksi Lokasi dengan LLM
    location_info = extract_location_with_llm(user_provided_scenario)

    # TAHAP 1: Download gambar satelit terbaru dari BMKG
    temp_image_path = os.path.join(DATA_RAW, "satelit_auto.png")
    img_path = download_image(IMAGE_URL, temp_image_path)
    if not img_path:
        summary = ""
        satellite_analysis_result = "Gagal mengunduh gambar satelit dari BMKG."
    elif location_info["lat"] != "Tidak Tersedia" and location_info["lon"] != "Tidak Tersedia":
        lat = float(location_info["lat"])
        lon = float(location_info["lon"])
        cloud_coverage_data = analyze_cloud_coverage_in_region(img_path, lat, lon, region_radius_deg=0.5)
        if "Error" not in cloud_coverage_data:
            summary = ", ".join([f"{cloud}: {percent}%" for cloud, percent in cloud_coverage_data.items()])
        else:
            summary = cloud_coverage_data.get("Error", "")
        satellite_analysis_result = analyze_cloud_type_with_llm(cloud_coverage_data, location_info)
    else:
        summary = ""
        satellite_analysis_result = "Analisis citra tidak dapat dilakukan karena koordinat tidak lengkap."

    # TAHAP 2: Kesimpulan Akhir oleh Ahli
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
    all_expert_docs = load_and_process_documents(
        pdf_files=[PDF_CUACA, PDF_TANAH],
        json_path=JSON_DATA_LATIH,
        history_path=JSON_RIWAYAT_BANJIR,
        location_name=location_info['name']
    )

    if all_expert_docs:
        if os.path.exists(CACHE_EMBED_AHLI) and os.path.exists(CACHE_DOKUMEN_AHLI):
            with open(CACHE_DOKUMEN_AHLI, "rb") as f:
                all_expert_docs = pickle.load(f)
            expert_embeddings = np.load(CACHE_EMBED_AHLI)
        else:
            expert_embeddings = embedder.encode(all_expert_docs, convert_to_numpy=True, show_progress_bar=False)
            with open(CACHE_DOKUMEN_AHLI, "wb") as f:
                pickle.dump(all_expert_docs, f)
            np.save(CACHE_EMBED_AHLI, expert_embeddings)

        index = faiss.IndexFlatL2(expert_embeddings.shape[1])
        index.add(expert_embeddings)
        query_vector = embedder.encode([user_provided_scenario], convert_to_numpy=True)
        D, I = index.search(query_vector, k=7)
        context_for_expert = "\n\n".join([all_expert_docs[i] for i in I[0]])
    else:
        context_for_expert = "Tidak ada dokumen pendukung yang ditemukan."

    final_report = get_final_analysis(
        user_provided_scenario, 
        context_for_expert, 
        satellite_analysis_result, 
        final_question
    )

    # Kembalikan hasil sebagai dictionary
    return {
        "location_info": location_info,
        "cloud_coverage_summary": summary,
        "satellite_analysis": satellite_analysis_result,
        "expert_context": context_for_expert,
        "final_report": final_report
    }