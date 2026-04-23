import os

UPLOAD_DIR = "./uploaded_docs"
CHROMA_DIR = "./chroma_db"

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY = "ollama"
MODEL_NAME = "speakleash/bielik-11b-v3.0-instruct:Q4_K_M"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

os.makedirs(UPLOAD_DIR, exist_ok=True)
