import os, json, fitz, joblib, re
from collections import defaultdict

base_dir = "/app"
model = joblib.load(os.path.join(base_dir, "model", "heading_classifier.pkl"))
scaler = joblib.load(os.path.join(base_dir, "model", "scaler.pkl"))

label_map = {0: "None", 1: "H4", 2: "H3", 3: "H2", 4: "H1"}

def clean_text(text):
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def is_garbage(text):
    text = text.strip()
    if len(text) < 4:
        return True
    low = text.lower()
    if low in ["page", "table of contents", "toc", "figure", "fig"]:
        return True
    if re.fullmatch(r"[\d\W]+", text):
        return True
    if re.fullmatch(r"([a-zA-Z]{1,3}\s?){1,3}", text) and len(text) < 10:
        return True
    return False

def extract_title(doc):
    largest = ""
    max_size = 0
    for p in doc[:min(3, len(doc))]:
        blocks = p.get_text("dict")["blocks"]
        for b in blocks:
            for l in b.get("lines", []):
                for s in l.get("spans", []):
                    t = clean_text(s.get("text", ""))
                    if is_garbage(t):
                        continue
                    if s["size"] > max_size and len(t) >= 6:
                        max_size = s["size"]
                        largest = t
    return largest if largest else "Untitled Document"

def extract_features(text, span):
    return [
        span["size"],
        int("Bold" in span["font"] or span["flags"] in [1, 2, 17]),
        len(text.split()),
        len(text)
    ]

def remove_duplicates_and_garbage(outline):
    seen_texts = set()
    cleaned = []
    for item in outline:
        text = item["text"].lower()
        if text in seen_texts:
            continue
        if is_garbage(item["text"]):
            continue
        seen_texts.add(text)
        cleaned.append(item)
    return cleaned

def process_pdf(path):
    doc = fitz.open(path)
    title = extract_title(doc)
    seen = set()
    outline = []
    for pno, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            for l in b.get("lines", []):
                line_text = " ".join([s.get("text", "").strip() for s in l.get("spans", [])])
                line_text = clean_text(line_text)
                if not line_text or is_garbage(line_text):
                    continue
                last_span = l.get("spans", [])[-1]
                features = extract_features(line_text, last_span)
                pred = model.predict(scaler.transform([features]))[0]
                label = label_map.get(pred, "None")
                key = (label, line_text.lower(), pno)
                if label == "None" or key in seen:
                    continue
                seen.add(key)
                outline.append({
                    "level": label,
                    "text": line_text,
                    "page": pno
                })
    return {"title": title, "outline": remove_duplicates_and_garbage(outline)}

def main():
    input_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
    for filename in pdf_files:
        pdf_path = os.path.join(input_dir, filename)
        result = process_pdf(pdf_path)
        with open(os.path.join(output_dir, filename.replace(".pdf", ".json")), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    print("✅ Processing complete.")

if __name__ == "__main__":
    main()
