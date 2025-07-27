# Adobe India Hackathon 2025 - Round 1A Submission

## 🚀 Project Title: PDF Structure Extraction & Heading Classification

This project was developed as a submission for Round 1A of the Adobe India Hackathon 2025. It focuses on extracting structured outlines from unstructured PDF documents by detecting headings and their hierarchy levels.


---

## 📌 Problem Statement

Given a set of PDF documents, extract a hierarchical outline with headings classified into levels (e.g., H1, H2, H3, etc.). The challenge involved identifying noisy, broken, or irregularly styled headings and building a robust model to classify them correctly.

---

## 🛠️ Tech Stack

* Python
* PyMuPDF (fitz)
* Scikit-learn (joblib for model persistence)
* FastAPI (for API support - optional)
* Docker (for containerization)

---

## 📁 Directory Structure

```
round1a/
├── app/
│   ├── extract_outline.py          # Main script for PDF parsing & heading classification
│   ├── model/
│   │   ├── heading_classifier.pkl  # Trained classifier model
│   │   └── scaler.pkl              # Feature scaler used with the model
│   └── input/                      # Input PDFs
│   └── output/                     # Output JSONs
├── requirements.txt               # Python dependencies
├── Dockerfile                     # For running in Docker
```

---

## ⚙️ How to Run

### ▶️ Option 1: Run with Docker

**Build Docker image:**

```bash
docker build -t adobe-round1a .
```

**Run the container:**

```bash
docker run --rm -v $(pwd)/app/input:/app/input -v $(pwd)/app/output:/app/output adobe-round1a
```

### ▶️ Option 2: Run Locally

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Run the script:**

```bash
python app/extract_outline.py
```

Make sure `input/` contains the PDFs and `model/` has the `heading_classifier.pkl` and `scaler.pkl` files.

---

## ✅ Output Format

Each PDF generates a JSON file in the `output/` directory with the following structure:

```json
{
  "title": "Document Title",
  "outline": [
    { "level": "H1", "text": "Section Heading", "page": 1 },
    { "level": "H2", "text": "Subsection Heading", "page": 2 },
    ...
  ]
}
```




---

## 🏁 Final Notes

* Execution time optimized to ≤10s for multiple PDFs
* Headings cleaning, word merging, and de-duplication handled
* Docker ensures easy reproducibility across systems
