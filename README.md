# RAG CLI with LLaMA 3.1 8B INT4 + OpenVINO

This is a CLI application that answers questions from a guide PDF using Retrieval-Augmented Generation (RAG) with LLaMA 3.1 8B Instruct quantized to INT4 and accelerated with OpenVINO.

---

## Features

- Query any document using LLaMA 3.1 locally
- Fast inference with OpenVINO (INT4 quantized)
- FAISS-powered vector search
- SentenceTransformers for chunk embeddings
- Fully offline RAG pipeline


---

##  Installation

### 1. Clone Repo & Create Virtual Environment

```bash
git clone <your-repo-url>
cd <project-root>
python3.12 -m venv myenv
source myenv/bin/activate


---

## Project Structure
#require install library

pip install --upgrade pip
pip install -r requirements.txt

openvino==2024.1.0
faiss-cpu==1.7.4
sentence-transformers==2.2.2
pdfplumber==0.10.2
transformers==4.41.2
torch==2.3.0
numpy
tqdm==4.66.2
optimum[openvino]

#1 Download model 
chmod +x download_model.sh
./download_model.sh

#2 convert_to_int4_openvino.sh
./convert_to_int4_openvino.sh
chmod +x convert_to_int4_openvino.sh
#output
models/llama-3.1-8b-int4/

#rag_cli.py main file for run the project
python rag_cli.py
#use access_token for run thid mode
Access_Token

#3. Extract Chunks from PDF
python scripts/extract_pdf.py

# 4Embed Chunks into FAISS Index
python scripts/embed_chunks.py

# 5. Download and Convert Model to INT4
python scripts/convert_model.py

# python scripts/rag_cli.py
python scripts/rag_cli.py


# Structure

# project-root/
# ├── data/
# │ └── guide_pdf.pdf # Input PDF guide
# │
# ├── models/
# │ └── llama-3.1-8b-int4/ 
# │
# ├── vector_store/
# │ └── faiss_index/
# │ ├── index.faiss # FAISS index
# │ └── chunks.txt # Source text chunks
# │
# ├── scripts/
# │ ├── extract_pdf.py # Extracts chunks from PDF
# │ ├── embed_chunks.py # Embeds and stores FAISS index
# │ ├── convert_model.py # Converts LLaMA to OpenVINO INT4
# │ └── rag_cli.py # Main CLI Q&A app
# │
# ├── run_demo.sh # Optional automation script
# └── README.md # You're here!

