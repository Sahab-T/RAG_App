import os
import json
import pdfplumber

# === CONFIG ===
PDF_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/guide.pdf"))
OUTPUT_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/guide.jsonl"))

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def save_chunks_to_jsonl(chunks, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            json.dump({"text": chunk.strip().replace("\n", " ")}, f)
            f.write("\n")

def main():
    print(f"Extracting text from: {PDF_FILE}")
    text = extract_text_from_pdf(PDF_FILE)

    print("Chunking text...")
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

    print(f"Saving to JSONL: {OUTPUT_FILE}")
    save_chunks_to_jsonl(chunks, OUTPUT_FILE)

    print("Done.")

if __name__ == "__main__":
    main()
