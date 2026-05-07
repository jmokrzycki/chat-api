from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from core.config import CHROMA_DIR, EMBEDDING_MODEL

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

vector_store = Chroma(
    collection_name="bielik_rag",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR
)

def clear_vector_store() -> None:
    """Czyszczenie całej bazy wektorowej."""
    data = vector_store.get()
    ids = data.get("ids", [])
    if ids:
        vector_store.delete(ids=ids)

def delete_document_from_vector_store(filename: str) -> None:
    """Usuwanie konkretnego dokumentu z bazy wektorowej."""
    data = vector_store.get(where={"filename": filename})
    ids = data.get("ids", [])
    if ids:
        vector_store.delete(ids=ids)
