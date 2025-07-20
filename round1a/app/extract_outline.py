# Updated extract_outline.py — Optimized + Clean Headings
import os, json, fitz, joblib, numpy as np, time

model = joblib.load("/app/model/heading_classifier.pkl")
scaler = joblib.load("/app/model/scaler.pkl")

label_map = {0: "None", 1: "H4", 2: "H3", 3: "H2", 4: "H1"}

def extract_title(doc):
    max_size, title = 0, ""
    for s in doc[0].get_text("dict")["blocks"]:
        for l in s.get("lines", []):
            for span in l.get("spans", []):
                t = span.get("text", "").strip()
                if len(t) > 5 and span["size"] > max_size:
                    max_size = span["size"]
                    title = t
    return title

def clean_text(t):
    return t.replace("©", "").replace("…", "").replace(".", "").strip()

def is_garbage(text):
    if len(text) < 6: return True
    if text.count(" ") <= 1 and sum(c.isalpha() for c in text)/len(text) < 0.4:
        return True
    return False

def extract_features(text, span):
    return [
        span["size"],
        int("Bold" in span["font"] or span["flags"] in [1, 2, 17]),
        len(text.split()),
        len(text)
    ]

def process_pdf(filepath):
    doc = fitz.open(filepath)
    title = extract_title(doc)
    outlines, seen = [], set()

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            for l in b.get("lines", []):
                line_text = " ".join([s["text"].strip() for s in l["spans"] if s.get("text")])
                if not line_text or is_garbage(line_text): continue
                last_span = l["spans"][-1]
                feats = extract_features(line_text, last_span)
                pred = model.predict(scaler.transform([feats]))[0]
                label = label_map.get(pred, "None")
                if label == "None": continue
                key = (label, line_text.lower(), page_num)
                if key in seen: continue
                seen.add(key)
                outlines.append({
                    "level": label,
                    "text": clean_text(line_text),
                    "page": page_num
                })

    return { "title": title, "outline": outlines }

def main():
    start = time.time()
    input_dir, output_dir = "/app/input", "/app/output"
    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            res = process_pdf(os.path.join(input_dir, filename))
            with open(os.path.join(output_dir, filename.replace(".pdf", ".json")), "w") as f:
                json.dump(res, f, indent=2)
    print(f"✅ Processed in {round(time.time() - start, 2)} sec")

if __name__ == "__main__":
    main()
