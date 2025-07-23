import os
import json
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# === CONFIG ===
JSONL_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/guide.jsonl"))
VECTOR_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../vector_store/faiss_index"))
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

os.makedirs(VECTOR_DIR, exist_ok=True)

def load_chunks(jsonl_file):
    chunks = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            chunks.append(data["text"])
    return chunks

def embed_chunks(chunks, model_name):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    return np.array(embeddings)

def save_faiss(embeddings, chunks, save_dir):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, os.path.join(save_dir, "index.faiss"))

    with open(os.path.join(save_dir, "chunks.txt"), "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.strip().replace("\n", " ") + "\n")

def main():
    print("Loading guide chunks...")
    chunks = load_chunks(JSONL_FILE)

    print("Embedding chunks...")
    embeddings = embed_chunks(chunks, EMBED_MODEL)

    print("Saving FAISS index and chunk text...")
    save_faiss(embeddings, chunks, VECTOR_DIR)

    print("Vector DB ready at:", VECTOR_DIR)

if __name__ == "__main__":
    main()
