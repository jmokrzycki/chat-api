import json
import os
from typing import AsyncGenerator
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

app = FastAPI()

app.add_middleware(
    CORSMiddleware, # type: ignore
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "./uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

llm = ChatOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    model="speakleash/bielik-11b-v3.0-instruct:Q4_K_M",
    streaming=True
)

embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

vector_store = Chroma(
    collection_name="bielik_rag",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        filename = file.filename or "unknown_file"
        file_path = os.path.join(UPLOAD_DIR, filename)

        with open(file_path, "wb") as buffer:
            while content := await file.read(1024 * 1024):
                buffer.write(content)

        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
        elif filename.lower().endswith(".txt"):
            loader = TextLoader(file_path, encoding='utf-8')
            docs = loader.load()
        else:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="Nieobsługiwany format pliku. Użyj .pdf lub .txt.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vector_store.add_documents(splits)
        os.remove(file_path)

        return {"message": f"Plik {filename} został pomyślnie przetworzony i dodany do bazy wiedzy."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd podczas przetwarzania pliku: {str(e)}")

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

@app.post("/api/chat")
async def ask_bielik(data: dict):
    user_prompt = str(data.get("prompt", "Brak pytania"))
    template = """Jesteś inteligentnym i pomocnym asystentem AI.
Został Ci dostarczony poniższy KONTEKST w postaci fragmentów dokumentów.
Odpowiedz na pytanie bazując na tym kontekście.
Jeśli nie potrafisz znaleźć odpowiedzi w kontekście, powiedz o tym, a następnie odpowiedz zgodnie z własną wiedzą.

KONTEKST:
{context}

PYTANIE UŻYTKOWNIKA:
{question}
"""
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
            RunnableParallel(
                context=retriever | RunnableLambda(format_docs),
                question=RunnablePassthrough()
            )
            | prompt
            | llm
            | StrOutputParser()
    )

    async def generate_response() -> AsyncGenerator[str, None]:
        try:
            async for chunk in rag_chain.astream(user_prompt):
                if chunk:
                    yield json.dumps({"response": chunk}) + "\n"
        except Exception as e:
            print(f"Błąd podczas generowania: {e}")
            yield json.dumps({"response": f"\n[Błąd przetwarzania: {str(e)}]"}) + "\n"

    return StreamingResponse(
        generate_response(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
