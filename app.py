# app.py
from flask import Flask, request, jsonify
import os
import threading
import json
from utils import *

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# In-memory status tracker
processing_status = {}

@app.route('/blueprints/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    pdf_name = file.filename
    project_id = request.form.get('project_id', 'default')

    file_path = os.path.join(UPLOAD_FOLDER, pdf_name)
    file.save(file_path)

    if pdf_name not in processing_status:
        processing_status[pdf_name] = {"status": "in_progress"}

    thread = threading.Thread(target=background_process, args=(file_path, pdf_name))
    thread.start()

    return jsonify({
        "status": "uploaded",
        "pdf_name": pdf_name,
        "message": "Processing started in background."
    })

@app.route('/blueprints/result', methods=['GET'])
def get_result():
    pdf_name = request.args.get('pdf_name')
    if not pdf_name:
        return jsonify({"error": "pdf_name is required"}), 400

    if pdf_name not in processing_status:
        return jsonify({"error": "PDF not found or not processed"}), 404

    status = processing_status[pdf_name]["status"]
    if status == "in_progress":
        return jsonify({
            "pdf_name": pdf_name,
            "status": "in_progress",
            "message": "Processing is still in progress. Please try again later."
        })

    if status == "complete":
        return jsonify({
            "pdf_name": pdf_name,
            "status": "complete",
            "result": processing_status[pdf_name]["result"]
        })

    return jsonify({
        "error": "Processing failed",
        "details": processing_status[pdf_name]["message"]
    }), 500

def background_process(file_path, pdf_name):
    try:
        # Step 1: Extract rulebook
        rulebook = extract_notes_and_table(file_path)
        rulebook_file = os.path.join(RESULTS_FOLDER, f"rulebook_{pdf_name}.json")
        with open(rulebook_file, 'w') as f:
            json.dump({"rulebook": rulebook}, f, indent=2)

        # Step 2: Convert PDF to images and detect
        images = pdf_to_images(file_path)
        all_detections = []

        # Build symbol → description map from rulebook
        symbol_desc = {}
        for item in rulebook:
            if item["type"] == "table_row":
                symbol = item["symbol"].strip()
                # Fix common OCR errors
                symbol = symbol.replace('O', '0').replace('l', '1').replace('I', '1')
                if len(symbol) <= 5 and symbol.isalnum():
                    symbol_desc[symbol] = item["description"]

        # Fallback descriptions for common emergency lights
        fallback = {
            "A1": "2x4 LED Emergency Fixture",
            "A1E": "Exit/Emergency Combo Unit",
            "W": "Wall-Mounted Emergency LED",
            "P": "Parking Lot Light",
            "EL": "Emergency Light",
            "EX": "Exit Sign",
            "EM": "Emergency Light"
        }

        for img_data in images:
            detections = detect_shaded_rectangles(img_data["image"])
            for det in detections:
                bbox = det["bounding_box"]
                nearby_text = extract_nearby_text(img_data["image"], bbox)

                # 🔍 Step: Extract symbol from nearby text
                symbol = "UNKNOWN"
                for word in nearby_text:
                    word = word.strip('.,;:()')
                    if word.isalnum() and word.isupper() and 2 <= len(word) <= 6:
                        symbol = word
                        break

                # 🧹 Clean up common OCR mistakes
                clean_map = {
                    'O': '0', 'l': '1', 'I': '1', 'B': '8', 'S': '5',
                    'AIE': 'A1E', 'AlE': 'A1E', 'AE': 'A1E', 'A1': 'A1',
                    'BLOG': 'A1E', 'JOT': 'A1E', 'TA': 'A1E', 'T': 'W'
                }
                for bad, good in clean_map.items():
                    symbol = symbol.replace(bad, good)

                # Only keep valid symbols
                if len(symbol) > 6 or not any(c.isalpha() for c in symbol):
                    symbol = "UNKNOWN"

                all_detections.append({
                    "symbol": symbol,
                    "bounding_box": bbox,
                    "text_nearby": nearby_text,
                    "source_sheet": img_data["name"]
                })

        # Save raw detections
        det_file = os.path.join(RESULTS_FOLDER, f"detections_{pdf_name}.json")
        with open(det_file, 'w') as f:
            json.dump(all_detections, f, indent=2)

        # Step 3: Group using rulebook + fallback
        summary = {}
        for det in all_detections:
            sym = det["symbol"]

            # Priority 1: Rulebook (from Lighting Schedule)
            if sym in symbol_desc:
                desc = symbol_desc[sym]
            # Priority 2: Fallback (common symbols)
            elif sym in fallback:
                desc = fallback[sym]
            # Priority 3: Generic
            else:
                desc = "Generic Emergency Light"

            key = f"Light_{sym}"
            if key not in summary:
                summary[key] = {"count": 0, "description": desc}
            summary[key]["count"] += 1

        # Save final result
        final_result = {}
        for k, v in summary.items():
            tag = k.replace("Light_", "")
            final_result[tag] = {"count": v["count"], "description": v["description"]}

        final_file = os.path.join(RESULTS_FOLDER, f"result_{pdf_name}.json")
        with open(final_file, 'w') as f:
            json.dump(final_result, f, indent=2)

        processing_status[pdf_name] = {
            "status": "complete",
            "result": final_result
        }

    except Exception as e:
        print(f"❌ ERROR: {e}")
        processing_status[pdf_name] = {
            "status": "error",
            "message": str(e)
        }

# ✅ Moved outside of background_process()
if __name__ == '__main__':
    print("✅ Starting Flask server on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))