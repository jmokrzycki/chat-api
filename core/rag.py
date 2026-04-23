import json
import os
from typing import AsyncGenerator

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

from core.config import (
    UPLOAD_DIR, CHROMA_DIR, OLLAMA_BASE_URL,
    OLLAMA_API_KEY, MODEL_NAME, EMBEDDING_MODEL
)
from core.settings import get_saved_template

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

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

def process_and_add_document(filename: str) -> None:
    file_path = os.path.join(UPLOAD_DIR, filename)

    if filename.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif filename.lower().endswith(".txt"):
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        os.remove(file_path)
        raise ValueError("Nieobsługiwany format pliku. Użyj .pdf lub .txt.")

    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vector_store.add_documents(splits)

    os.remove(file_path)

def clear_vector_store() -> None:
    data = vector_store.get()
    ids = data.get("ids", [])
    if ids:
        vector_store.delete(ids=ids)

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

async def generate_chat_response(user_prompt: str, custom_template: str) -> AsyncGenerator[str, None]:
    default_template = get_saved_template()
    template_str = custom_template.strip() if custom_template.strip() else default_template

    if "{context}" not in template_str:
        template_str += "\n\nKONTEKST:\n{context}"
    if "{question}" not in template_str:
        template_str += "\n\nPYTANIE UŻYTKOWNIKA:\n{question}"

    prompt = ChatPromptTemplate.from_template(template_str)

    rag_chain = (
            RunnableParallel(
                context=retriever | RunnableLambda(format_docs),
                question=RunnablePassthrough()
            )
            | prompt
            | llm
            | StrOutputParser()
    )

    try:
        async for chunk in rag_chain.astream(user_prompt):
            if chunk:
                yield json.dumps({"response": chunk}) + "\n"
    except Exception as e:
        print(f"Błąd podczas generowania: {e}")
        yield json.dumps({"response": f"\n[Błąd przetwarzania: {str(e)}]"}) + "\n"
