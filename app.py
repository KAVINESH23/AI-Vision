# utils.py
import fitz  # PyMuPDF
import cv2
import numpy as np
import pytesseract
import os
import json
import re

# Set Tesseract path (Windows only)
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except Exception:
    pass

def pdf_to_images(pdf_path, dpi=200):
    """Convert PDF to list of OpenCV images."""
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img_data = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img_data.reshape(pix.height, pix.width, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append({
            "image": img,
            "page_num": page_num + 1,
            "name": os.path.basename(pdf_path).replace(".pdf", "") + f"_page_{page_num+1}"
        })
    doc.close()
    return images

def detect_shaded_rectangles(image, min_area=500, max_area=5000):
    """Detect shaded rectangles (emergency lights)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h
            if 1.5 < aspect_ratio < 4.0:
                detections.append({
                    "bounding_box": [x, y, x+w, y+h],
                    "area": area,
                    "aspect_ratio": round(aspect_ratio, 2)
                })
    return detections

def extract_nearby_text(image, bbox, padding=50):
    """Extract text near bounding box"""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    h, w = image.shape[:2]
    roi_x1 = max(0, cx - padding)
    roi_y1 = max(0, cy - padding)
    roi_x2 = min(w, cx + padding)
    roi_y2 = min(h, cy + padding)
    roi = image[roi_y1:roi_y2, roi_x1:roi_x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(gray, config='--psm 6').strip()

def extract_notes_and_table(pdf_path):
    """Extract General Notes and Lighting Schedule Table using OCR on image regions"""
    doc = fitz.open(pdf_path)
    rulebook = []

    for page_num in range(len(doc)):
        sheet_name = f"{os.path.basename(pdf_path)}_page_{page_num+1}"
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=200)
        img_data = np.frombuffer(pix.samples, dtype=np.uint8)
        image = img_data.reshape(pix.height, pix.width, 3)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        h, w = image.shape[:2]

        # --- Region 1: Top-Left (General Notes)
        roi_notes = image[50:400, 50:700]
        notes_text = pytesseract.image_to_string(roi_notes, config='--psm 6').strip()
        if len(notes_text) > 10:
            for line in notes_text.split('\n'):
                line = re.sub(r'\s+', ' ', line.strip())
                if len(line) > 5 and any(kw in line.lower() for kw in ["emergency", "unswitched", "power", "note"]):
                    rulebook.append({
                        "type": "note",
                        "text": line,
                        "source_sheet": sheet_name
                    })

        # --- Region 2: Bottom-Right (Lighting Schedule Table)
        roi_table = image[h-600:h-100, w-800:w-100]
        table_text = pytesseract.image_to_string(roi_table, config='--psm 6').strip()

        if len(table_text) > 20:
            lines = table_text.split('\n')
            for line in lines:
                line = re.sub(r'\s+', ' ', line.strip())
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue

                symbol = parts[0]
                if len(symbol) <= 6 and symbol.isalnum() and symbol.isupper():
                    description = " ".join(parts[1:5])
                    rulebook.append({
                        "type": "table_row",
                        "symbol": symbol,
                        "description": description,
                        "source_sheet": sheet_name
                    })

    doc.close()
    return rulebook

def group_lights(detections, rulebook):
    """Group lights using rulebook"""
    symbol_desc = {}
    for item in rulebook:
        if item["type"] == "table_row":
            symbol = item["symbol"].strip()
            symbol = symbol.replace('O', '0').replace('l', '1').replace('I', '1')
            if len(symbol) <= 5 and symbol.isalnum():
                symbol_desc[symbol] = item["description"]

    fallback = {
        "A1": "2x4 LED Emergency Fixture",
        "A1E": "Exit/Emergency Combo Unit",
        "W": "Wall-Mounted Emergency LED"
    }

    summary = {}
    for det in detections:
        sym = det["symbol"]

        # Clean up OCR garbage
        clean_map = {
            'AIE': 'A1E', 'AlE': 'A1E', 'AE': 'A1E', 'A1': 'A1',
            'BLOG': 'A1E', 'JOT': 'A1E', 'TA': 'A1E', 'T': 'W'
        }
        sym = clean_map.get(sym, sym)

        desc = symbol_desc.get(sym, fallback.get(sym, "Generic Emergency Light"))
        key = f"Light_{sym}"

        if key not in summary:
            summary[key] = {"count": 0, "description": desc}
        summary[key]["count"] += 1

    return {"summary": summary}

def draw_detections(image, detections, output_path):
    """Draw bounding boxes and labels"""
    img_copy = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bounding_box"]
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_copy, det["symbol"], (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(output_path, img_copy)
