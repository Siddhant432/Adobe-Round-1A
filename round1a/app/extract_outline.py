import os
import json
import fitz  # PyMuPDF
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the model
model = joblib.load("/app/model/heading_classifier.pkl")
scaler = joblib.load("/app/model/scaler.pkl")

# Heuristic to guess the Title (largest font on page 1)
def extract_title(doc):
    page = doc[0]
    blocks = page.get_text("dict")['blocks']
    max_size = 0
    title = ""
    for b in blocks:
        for l in b.get("lines", []):
            for s in l.get("spans", []):
                if s["size"] > max_size:
                    max_size = s["size"]
                    title = s["text"].strip()
    return title

# Feature extraction
def extract_features(span):
    text = span["text"].strip()
    size = span["size"]
    flags = span["flags"]  # bold, italic etc.
    font = span["font"]
    n_words = len(text.split())
    n_chars = len(text)
    is_bold = int("Bold" in font or flags in [1, 2, 17])
    return [size, is_bold, n_words, n_chars]

# Map model labels
label_map = {0: "None", 1: "H3", 2: "H2", 3: "H1"}

# Main processing function
def process_pdf(filepath):
    doc = fitz.open(filepath)
    title = extract_title(doc)
    outlines = []
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")['blocks']
        for b in blocks:
            for l in b.get("lines", []):
                for s in l.get("spans", []):
                    text = s.get("text", "").strip()
                    if not text or len(text) < 4:
                        continue
                    feats = extract_features(s)
                    X = scaler.transform([feats])
                    pred = model.predict(X)[0]
                    label = label_map.get(pred, "None")
                    if label != "None":
                        outlines.append({
                            "level": label,
                            "text": text,
                            "page": page_num
                        })
    return {
        "title": title,
        "outline": outlines
    }

# I/O handler
def main():
    input_dir = "/app/input"
    output_dir = "/app/output"
    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(input_dir, filename)
            print(f"Processing {filename}...")
            result = process_pdf(filepath)
            outname = filename.replace(".pdf", ".json")
            with open(os.path.join(output_dir, outname), "w") as f:
                json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
