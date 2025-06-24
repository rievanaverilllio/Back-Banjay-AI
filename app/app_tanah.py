import os
import re
import json
import faiss
import numpy as np
import PyPDF2
import requests
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# === CONFIG ===
API_KEY = "ArKWofu383ebzKYXakZB8RF8Clgx5hv6"
API_URL = "https://api.mistral.ai/v1/chat/completions"
MODEL_ID = "mistral-large-latest"
JSON_PATH = "merged_training_data.json"
PDF_FILES = ["cuaca.pdf", "tanah.pdf"]
HISTORY_JSON = "jumlah_kejadians.json"

EMBED_CACHE = "cached_embeddings.npy"
DOC_CACHE = "cached_docs.pkl"

# === Preprocessing Text ===
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Hapus whitespace ganda
    text = re.sub(r'(\n|\r)', ' ', text)  # Ganti newline
    text = re.sub(r'Page \d+ of \d+', '', text)  # Hapus footer halaman
    text = re.sub(r'Referensi|DAFTAR PUSTAKA', '', text, flags=re.IGNORECASE)
    return text.strip()

# === Load JSON dataset ===
print("📄 Memuat data JSON...")
with open(JSON_PATH, "r", encoding="utf-8") as f:
    json_data = json.load(f)
json_docs = [item["response"] for item in json_data]

# === Load and extract PDF text ===
print("📄 Mengekstrak teks dari jurnal PDF...")
def extract_text_from_pdf(file_path):
    pdf = PyPDF2.PdfReader(file_path)
    texts = []
    for page in pdf.pages:
        raw = page.extract_text()
        if raw:
            cleaned = clean_text(raw)
            texts.append(cleaned)
    return "\n".join(texts)

pdf_docs = []
for file in PDF_FILES:
    content = extract_text_from_pdf(file)
    chunks = content.split(". ")
    pdf_docs.extend([c.strip() for c in chunks if len(c.strip()) > 50 and len(c) < 500])

# === Load historical data from JSON ===
print("📊 Memuat data sejarah banjir dari JSON...")
with open(HISTORY_JSON, "r", encoding="utf-8") as f:
    history_data = json.load(f)

history_docs = []
for row in history_data:
    nama = row.get("Nama Kabupaten/Kota", "Tidak diketahui")
    lat = row.get("Latitude", "Tidak tersedia")
    lon = row.get("Longitude", "Tidak tersedia")
    riwayat = ", ".join([
        f"{tahun}: {row[str(tahun)]} kejadian"
        for tahun in range(2010, 2025)
        if str(tahun) in row and row[str(tahun)] is not None
    ])
    paragraph = (
        f"Wilayah {nama} (lat: {lat}, lon: {lon}) memiliki riwayat bencana banjir: {riwayat}. "
        f"Ini mencerminkan tren bencana pada periode 2010–2024."
    )
    history_docs.append(paragraph)

# === Gabungkan semua dokumen ===
all_docs = json_docs + pdf_docs + history_docs
print(f"✅ Total dokumen gabungan: {len(all_docs)}")

# === Embedding dan Index FAISS ===
if os.path.exists(EMBED_CACHE) and os.path.exists(DOC_CACHE):
    print("📦 Memuat embedding dari cache...")
    doc_embeddings = np.load(EMBED_CACHE)
    with open(DOC_CACHE, "rb") as f:
        all_docs = pickle.load(f)
else:
    print("🔄 Membuat embedding dan index (pertama kali)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    doc_embeddings = []
    for doc in tqdm(all_docs, desc="🔁 Encoding teks"):
        emb = embedder.encode(doc, convert_to_numpy=True)
        doc_embeddings.append(emb)
    doc_embeddings = np.array(doc_embeddings)

    # Simpan cache
    np.save(EMBED_CACHE, doc_embeddings)
    with open(DOC_CACHE, "wb") as f:
        pickle.dump(all_docs, f)

# === Buat index FAISS ===
dimension = len(doc_embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

# === Input pengguna ===
question = input("❓ Masukkan pertanyaanmu: ")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
query_vector = embedder.encode([question])
D, I = index.search(np.array(query_vector), k=5)

retrieved_docs = [all_docs[i] for i in I[0]]
context = "\n\n".join(retrieved_docs)

# === Prompt ke Mistral ===
prompt = f"""
Anda adalah seorang ahli hidrologi dan geoteknik berpengalaman, dengan spesialisasi dalam analisis risiko banjir berbasis data spasial, meteorologi, dan geoteknologi.

Tugas Anda adalah:
- Menjawab secara sistematis, teknis, dan logis berdasarkan pengetahuan ilmiah.
- Mengaitkan informasi yang diberikan dengan data historis kejadian banjir (jika tersedia).
- Memberikan analisis yang dapat dipertanggungjawabkan secara ilmiah.
- Menyimpulkan tingkat risiko banjir secara kuantitatif (skor 0–100).
- Gunakan pengetahuan profesional dan data berikut untuk menilai risiko banjir di wilayah ini.
- Jelaskan secara mengalir seperti laporan naratif yang logis dan teknis.

---

### Konteks Teknis dan Historis:
{context}

---

### Pertanyaan Pengguna:
{question}

Tulislah dengan bahasa ilmiah dan profesional, hindari pengulangan, dan gunakan data yang tersedia untuk mendukung jawaban.
"""

# === Kirim ke Mistral API ===
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "model": MODEL_ID,
    "messages": [
        {"role": "user", "content": prompt}
    ],
    "temperature": 0.5,
    "max_tokens": 2048
}

response = requests.post(API_URL, headers=headers, json=payload)

print("\n=== Jawaban Mistral ===\n")
if response.status_code == 200:
    print(response.json()["choices"][0]["message"]["content"])
else:
    print(f"❌ Error {response.status_code}: {response.text}")
