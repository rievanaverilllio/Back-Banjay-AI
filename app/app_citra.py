import os
import json
import faiss
import numpy as np
import PyPDF2
import requests
from tqdm import tqdm
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import re
import pickle

# === CONFIG ===
API_KEY = "ArKWofu383ebzKYXakZB8RF8Clgx5hv6"
API_URL = "https://api.mistral.ai/v1/chat/completions"
MODEL_ID = "mistral-small-2503"
PDF_PATH = "cuaca.pdf"
IMAGE_PATH = "Satelit.png"

PDF_CACHE = "cached_pdf_chunks.pkl"
EMBEDDING_CACHE = "cached_pdf_embeddings.npy"

device = "cuda" if torch.cuda.is_available() else "cpu"

# === LOAD MODEL SEKALI SAJA ===
print("🔧 Loading BLIP & Embedder...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# === STEP 1: Generate Caption ===
def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(image, return_tensors="pt").to(device)
    outputs = blip_model.generate(**inputs)
    return blip_processor.decode(outputs[0], skip_special_tokens=True)

# === STEP 2: Extract & Clean PDF ===
def extract_clean_text_from_pdf(file_path):
    reader = PyPDF2.PdfReader(file_path)
    all_text = []
    for page in reader.pages:
        raw = page.extract_text()
        if raw:
            text = re.sub(r'\s+', ' ', raw)
            text = re.sub(r'\n+', ' ', text)
            text = re.sub(r'[^a-zA-Z0-9À-ž.,:%\-\(\)\[\] ]', '', text)
            all_text.append(text.strip())
    return " ".join(all_text)

def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# === STEP 3: Retrieve Context ===
def retrieve_relevant_context(query, docs, k=5):
    if os.path.exists(EMBEDDING_CACHE):
        print("📦 Memuat embedding dari cache...")
        doc_embeddings = np.load(EMBEDDING_CACHE)
    else:
        print("🔁 Membuat embedding PDF...")
        doc_embeddings = embedder.encode(docs, convert_to_numpy=True, show_progress_bar=True)
        np.save(EMBEDDING_CACHE, doc_embeddings)

    query_vector = embedder.encode([query])[0]
    index = faiss.IndexFlatL2(len(query_vector))
    index.add(doc_embeddings)
    D, I = index.search(np.array([query_vector]), k)
    return [docs[i] for i in I[0]]

# === STEP 4: Prompting ===
def ask_mistral_with_rag(caption, retrieved_context):
    context_block = "\n\n".join(retrieved_context)

    prompt = f"""
Citra Himawari-9 Cloud Type  
Citra ini adalah representasi terbaru dari data jenis awan (Cloud Type) yang bersumber dari satelit cuaca Himawari-9, dengan fokus cakupan wilayah Indonesia. Citra ini menyediakan visualisasi detail mengenai distribusi dan jenis awan yang menutupi kepulauan Indonesia dan sekitarnya.

Berikut hasil caption otomatis gambar:
"{caption}"

Berikut informasi tambahan dari literatur ilmiah terkait cuaca dan klasifikasi awan:
{context_block}

Tugas Anda:
- Lakukan analisis cuaca secara teknis dan mendalam berdasarkan informasi yang diberikan.
- Fokus pada penjelasan teknis tentang jenis awan, potensi hujan, serta penyebarannya.
- Kaitkan hanya dengan informasi ilmiah yang relevan tanpa menambahkan contoh kasus atau skenario lain di luar gambar dan teks jurnal.
- Gunakan bahasa profesional dan ilmiah.
- Jangan menambahkan studi kasus lain atau ilustrasi di luar konteks data.

Berikan simpulan meteorologis singkat dan tepat.
"""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt.strip()}],
        "temperature": 0.5
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# === MAIN ===
if __name__ == "__main__":
    print("[1] Captioning Gambar...")
    caption = generate_caption(IMAGE_PATH)
    print("🖼️ Caption:", caption)

    if os.path.exists(PDF_CACHE):
        print("📁 Memuat chunk teks dari cache...")
        with open(PDF_CACHE, "rb") as f:
            chunks = pickle.load(f)
    else:
        print("[2] Ekstrak & Preprocessing PDF...")
        raw_text = extract_clean_text_from_pdf(PDF_PATH)
        chunks = chunk_text(raw_text)
        with open(PDF_CACHE, "wb") as f:
            pickle.dump(chunks, f)

    print("[3] Retrieve dokumen relevan...")
    retrieved = retrieve_relevant_context(caption, chunks)

    print("[4] Mempersiapkan prompt dan mengirim ke Mistral...")
    result = ask_mistral_with_rag(caption, retrieved)

    print("\n=== Jawaban Mistral ===\n")
    if result.get("choices"):
        print(result["choices"][0]["message"]["content"])
    else:
        print(f"❌ Error: {result}")
