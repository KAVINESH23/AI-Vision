# app.py
from flask import Flask, request, jsonify
import os
import threading
import json
from utils import *

app = Flask(__name__)

# Define folders
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# In-memory processing status tracker
processing_status = {}

@app.route('/')
def home():
    return jsonify({
        "status": "alive",
        "service": "Emergency Lighting Detection API",
        "endpoints": [
            "POST /blueprints/upload",
            "GET /blueprints/result?pdf_name=..."
        ]
    })

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

    # Start background processing
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

    status_info = processing_status[pdf_name]
    if status_info["status"] == "in_progress":
        return jsonify({
            "pdf_name": pdf_name,
            "status": "in_progress",
            "message": "Processing is still in progress. Please try again later."
        })

    if status_info["status"] == "complete":
        return jsonify({
            "pdf_name": pdf_name,
            "status": "complete",
            "result": status_info["result"]
        })

    return jsonify({
        "error": "Processing failed",
        "details": status_info.get("message", "Unknown error")
    }), 500

def background_process(file_path, pdf_name):
    try:
        print(f"🚀 Starting processing for {pdf_name}")

        # Step 1: Extract rulebook (General Notes + Lighting Schedule Table)
        rulebook = extract_notes_and_table(file_path)
        rulebook_file = os.path.join(RESULTS_FOLDER, f"rulebook_{pdf_name}.json")
        with open(rulebook_file, 'w') as f:
            json.dump({"rulebook": rulebook}, f, indent=2)
        print(f"✅ Rulebook saved: {len(rulebook)} items")

        # Step 2: Convert PDF to images and detect emergency lights
        images = pdf_to_images(file_path)
        all_detections = []

        for img_data in images:
            detections = detect_shaded_rectangles(img_data["image"])
            for det in detections:
                bbox = det["bounding_box"]
                nearby_text = extract_nearby_text(img_data["image"], bbox)

                # Extract symbol from nearby text
                symbol = "UNKNOWN"
                for word in nearby_text.split():
                    word = word.strip('.,;:()')
                    if word.isalnum() and word.isupper() and 2 <= len(word) <= 6:
                        symbol = word
                        break

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
        print(f"✅ Found {len(all_detections)} detections")

        # Generate annotation screenshot
        if images and all_detections:
            draw_detections(images[0]["image"], all_detections[:10], "annotation_example.png")
            print("🖼️ Annotation screenshot saved: annotation_example.png")

        # Step 3: Group lights using rulebook + fallback logic
        summary = {}
        fallback = {
            "A1": "2x4 LED Emergency Fixture",
            "A1E": "Exit/Emergency Combo Unit",
            "W": "Wall-Mounted Emergency LED",
            "P": "Parking Lot Light",
            "EL": "Emergency Light",
            "EX": "Exit Sign"
        }

        for det in all_detections:
            sym = det["symbol"]

            # Clean up common OCR mistakes
            clean_map = {
                'AIE': 'A1E', 'AlE': 'A1E', 'AE': 'A1E', 'A1': 'A1',
                'BLOG': 'A1E', 'JOT': 'A1E', 'TA': 'A1E', 'T': 'W',
                'I': 'W', 'O': '0', 'S': '5'
            }
            sym = clean_map.get(sym, sym)

            # Priority 1: Rulebook
            desc = None
            for item in rulebook:
                if item["type"] == "table_row" and item["symbol"].strip() == sym:
                    desc = item["description"]
                    break

            # Priority 2: Fallback
            if not desc:
                desc = fallback.get(sym, "Generic Emergency Light")

            key = f"Light_{sym}"
            if key not in summary:
                summary[key] = {"count": 0, "description": desc}
            summary[key]["count"] += 1

        # Final result
        final_result = {}
        for k, v in summary.items():
            tag = k.replace("Light_", "")
            final_result[tag] = {"count": v["count"], "description": v["description"]}

        # Save result
        result_file = os.path.join(RESULTS_FOLDER, f"result_{pdf_name}.json")
        with open(result_file, 'w') as f:
            json.dump(final_result, f, indent=2)

        # Update status
        processing_status[pdf_name] = {
            "status": "complete",
            "result": final_result
        }
        print("🎉 Processing complete!")

    except Exception as e:
        print(f"❌ ERROR: {e}")
        processing_status[pdf_name] = {
            "status": "error",
            "message": str(e)
        }

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"✅ Starting Flask server on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port)
