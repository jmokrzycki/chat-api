import json
import os
import hashlib
from typing import AsyncGenerator, List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.config import (
    UPLOAD_DIR, CHROMA_DIR, OLLAMA_BASE_URL,
    OLLAMA_API_KEY, MODEL_NAME, EMBEDDING_MODEL
)
from core.settings import get_trained_files_list

llm = ChatOpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key=OLLAMA_API_KEY,
    model=MODEL_NAME,
    streaming=True
)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

vector_store = Chroma(
    collection_name="bielik_rag",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR
)

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

def clear_vector_store() -> None:
    data = vector_store.get()
    ids = data.get("ids", [])
    if ids:
        vector_store.delete(ids=ids)

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def get_history_text(chat_history: List[Dict[str, str]]) -> str:
    if not chat_history:
        return ""
    return "\n".join([f"{'UŻYTKOWNIK' if msg['sender'] == 'user' else 'ASYSTENT'}: {msg['text']}" for msg in chat_history])

async def rephrase_question(user_prompt: str, chat_history: List[Dict[str, str]], custom_rephrase: str) -> str:
    if not chat_history:
        return user_prompt

    history_text = get_history_text(chat_history)
    rephrase_str = custom_rephrase.strip() + "\n\nHISTORIA ROZMOWY:\n{chat_history}\n\nNAJNOWSZE PYTANIE:\n{question}\n\nSAMODZIELNE ZAPYTANIE DO BAZY DANYCH:"

    prompt = ChatPromptTemplate.from_template(rephrase_str)
    chain = prompt | llm | StrOutputParser()

    try:
        standalone_q = await chain.ainvoke({"chat_history": history_text, "question": user_prompt})
        return standalone_q.strip().strip('"').strip("'")
    except Exception:
        return user_prompt

async def generate_chat_response(user_prompt: str, custom_template: str, custom_rephrase: str, chat_history: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
    history_text = get_history_text(chat_history)
    search_query = await rephrase_question(user_prompt, chat_history, custom_rephrase)

    active_files = get_trained_files_list()

    if not active_files:
        retrieved_docs = []
    else:
        dynamic_retriever = vector_store.as_retriever(
            search_kwargs={"k": 3, "filter": {"filename": {"$in": active_files}}}
        )
        retrieved_docs = await dynamic_retriever.ainvoke(search_query)

    template_str = custom_template.strip()
    if chat_history: template_str += "\n\nHISTORIA ROZMOWY:\n{chat_history}"
    if retrieved_docs:
        context_text = format_docs(retrieved_docs)
        template_str += "\n\nKONTEKST Z DOKUMENTÓW:\n{context}"
    else:
        context_text = ""
    template_str += "\n\nUŻYTKOWNIK:\n{question}\n\nASYSTENT:"

    prompt = ChatPromptTemplate.from_template(template_str)
    chain = prompt | llm | StrOutputParser()

    try:
        async for chunk in chain.astream({"context": context_text, "question": user_prompt, "chat_history": history_text}):
            if chunk: yield json.dumps({"response": chunk}) + "\n"
    except Exception as e:
        yield json.dumps({"response": f"\n[Błąd: {str(e)}]"}) + "\n"

def delete_document_from_vector_store(filename: str) -> None:
    data = vector_store.get(where={"filename": filename})
    ids = data.get("ids", [])
    if ids:
        vector_store.delete(ids=ids)