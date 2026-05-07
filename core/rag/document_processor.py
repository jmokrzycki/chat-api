import os
import hashlib
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from core.config import UPLOAD_DIR
from core.rag.vector_store import vector_store, delete_document_from_vector_store

def calculate_file_hash(file_path: str) -> str:
    """Oblicza hash SHA-256 zawartości pliku."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_cached_filenames() -> List[str]:
    """
    Zwraca listę plików w bazie wektorowej, których hash zgadza się
    z hashem pliku aktualnie znajdującego się na dysku (w Stage).
    """
    data = vector_store.get()
    metadatas = data.get("metadatas", [])

    db_files = {}
    for m in metadatas:
        if "filename" in m:
            fname = m["filename"]
            if fname not in db_files:
                db_files[fname] = m.get("file_hash", "")

    valid_cached_files = []

    for fname, stored_hash in db_files.items():
        file_path = os.path.join(UPLOAD_DIR, fname)
        if os.path.exists(file_path):
            current_hash = calculate_file_hash(file_path)
            if current_hash == stored_hash:
                valid_cached_files.append(fname)

    return valid_cached_files

def process_and_add_document(filename: str) -> None:
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise ValueError(f"Plik {filename} nie istnieje na serwerze.")

    current_hash = calculate_file_hash(file_path)

    existing_data = vector_store.get(where={"filename": filename})

    if existing_data and existing_data.get("metadatas"):
        stored_hash = existing_data["metadatas"][0].get("file_hash")

        if stored_hash == current_hash:
            print(f"--- CACHE HIT: Plik {filename} nie zmienił treści. Pomijam re-processing. ---")
            return
        else:
            print(f"--- CACHE INVALID: Wykryto zmianę w {filename}. Aktualizacja wektorów... ---")
            delete_document_from_vector_store(filename)

    if filename.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif filename.lower().endswith(".txt"):
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        raise ValueError("Nieobsługiwany format pliku.")

    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    for split in splits:
        split.metadata["filename"] = filename
        split.metadata["file_hash"] = current_hash

    vector_store.add_documents(splits)