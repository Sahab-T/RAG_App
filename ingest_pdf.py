import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle

CHUNK_SIZE = 500
OVERLAP = 100

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i+chunk_size])
        if chunk: chunks.append(chunk)
    return chunks

def extract_pdf_chunks(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        full_text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    return chunk_text(full_text)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def main():
    pdf_path = "procyon_guide.pdf"
    chunks = extract_pdf_chunks(pdf_path)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, show_progress_bar=True)
    index = build_faiss_index(np.array(embeddings).astype('float32'))
    # Save everything
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    faiss.write_index(index, "faiss.index")

if __name__ == "__main__":
    main()
