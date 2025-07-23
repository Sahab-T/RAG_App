import os
import sys
import faiss
import numpy as np
from transformers import AutoTokenizer, TextIteratorStreamer
from optimum.intel.openvino import OVModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch
from threading import Thread

# ==== Config ====
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/llama-3.1-8b-int4"))
VECTOR_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../vector_store/faiss_index"))
K = 4  # top-K chunks to retrieve

def load_chunks():
    with open(os.path.join(VECTOR_DIR, "chunks.txt"), "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

def retrieve(query_emb, index, k):
    D, I = index.search(query_emb.astype("float32"), k)
    return I[0]

def main():
    print("RAG Q&A Chat — ask anything from the guide (type 'exit' to quit)\n")

    # Load FAISS index and chunks once
    print("Loading FAISS index and chunks...")
    index = faiss.read_index(os.path.join(VECTOR_DIR, "index.faiss"))
    chunks = load_chunks()

    # Load embedder and model once
    print("Loading embedding model and LLaMA 3.1 INT4...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = OVModelForCausalLM.from_pretrained(MODEL_DIR)

    while True:
        try:
            query = input("Query: ")
            if query.lower() in ["exit", "quit"]:
                print("Session ended.")
                break

            # Embed query
            print(" Embedding query...")
            query_emb = embedder.encode([query])
            top_ids = retrieve(query_emb, index, K)
            context = "\n\n".join([chunks[i] for i in top_ids])

            prompt = f"<|user|>\nAnswer the question using the context below.\n\nContext:\n{context}\n\nQuestion: {query}\n<|assistant|>\n"
            inputs = tokenizer(prompt, return_tensors="pt")
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)

            gen_kwargs = {
                **inputs,
                "streamer": streamer,
                "max_new_tokens": 300,
                "pad_token_id": tokenizer.eos_token_id,
                "do_sample": False
            }

            print("Generating answer...\n")
            thread = Thread(target=model.generate, kwargs=gen_kwargs)
            thread.start()

            for token in streamer:
                print(token, end="", flush=True)
            print("\n")

        except KeyboardInterrupt:
            print("\n Session ended by user.")
            break

if __name__ == "__main__":
    main()
