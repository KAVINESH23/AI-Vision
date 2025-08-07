# test_extract.py
from utils import extract_notes_and_table
import json
import os

print("🔍 Extracting rulebook from PDF.pdf...")
rulebook = extract_notes_and_table("data/PDF.pdf")

print(f"✅ Found {len(rulebook)} items")
for item in rulebook[:10]:
    print(item)

os.makedirs("results", exist_ok=True)
with open("results/rulebook_PDF.pdf.json", "w") as f:
    json.dump({"rulebook": rulebook}, f, indent=2)

print("📄 Saved to results/rulebook_PDF.pdf.json")