from flask import Flask, request, jsonify
from flask_cors import CORS
from main import run_full_analysis
import json
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Pastikan folder 'hasil' ada
os.makedirs('hasil', exist_ok=True)

@app.route('/api/analisis', methods=['POST'])
def analisis_wilayah():
    data = request.get_json()
    input_text = data.get("text")

    if not input_text:
        return jsonify({"error": "Teks tidak ditemukan"}), 400

    # Jalankan analisis
    result = run_full_analysis(input_text)

    # Gabungkan input + hasil
    full_data = {
        "input": input_text,
        "hasil": result
    }

    # Simpan hasil sebagai file JSON lokal
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # filename = f"hasil/analisis_{timestamp}.json"
    # with open(filename, 'w', encoding='utf-8') as f:
    #     json.dump(full_data, f, ensure_ascii=False, indent=2)

    # return jsonify({"hasil": result, "saved_to": filename})

if __name__ == "__main__":
    app.run(debug=True)
