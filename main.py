import json
import os
import shutil
from typing import cast, List, Union
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

Message = Union[ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam]

UPLOAD_DIR = "./uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

vector_store = Chroma(
    collection_name="bielik_rag",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)  # type: ignore
        if file.filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
        elif file.filename.lower().endswith(".txt"):
            loader = TextLoader(file_path, encoding='utf-8')
            docs = loader.load()
        else:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="Nieobsługiwany format pliku. Użyj .pdf lub .txt.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vector_store.add_documents(splits)
        os.remove(file_path)

        return {"message": f"Plik {file.filename} został pomyślnie przetworzony i dodany do bazy wiedzy."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd podczas przetwarzania pliku: {str(e)}")


@app.post("/api/chat")
async def ask_bielik(data: dict):
    user_prompt = str(data.get("prompt", "Brak pytania"))

    context = ""
    try:
        results = vector_store.similarity_search(user_prompt, k=3)
        if results:
            context = "\n\n---\n\n".join([doc.page_content for doc in results])
    except Exception as e:
        print(f"Błąd przeszukiwania bazy wektorowej: {e}")

    messages: List[Message]

    if context.strip():
        system_message = (
            "Jesteś inteligentnym i pomocnym asystentem AI. "
            "Został Ci dostarczony poniższy KONTEKST w postaci fragmentów dokumentów wgranych przez użytkownika. "
            "Odpowiedz na pytanie bazując na tym kontekście. "
            "Jeśli nie potrafisz znaleźć odpowiedzi w kontekście, powiedz o tym, a następnie odpowiedz zgodnie z własną wiedzą.\n\n"
            f"KONTEKST:\n{context}\n"
        )

        messages = [
            ChatCompletionSystemMessageParam(role="system", content=system_message),
            ChatCompletionUserMessageParam(role="user", content=user_prompt)
        ]
    else:
        messages = [ChatCompletionUserMessageParam(role="user", content=user_prompt)]

    async def proxy_stream():
        try:
            response = await client.chat.completions.create(
                model="speakleash/bielik-11b-v3.0-instruct:Q4_K_M",
                messages=messages,
                stream=True
            )

            stream = cast(AsyncStream[ChatCompletionChunk], cast(object, response))

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    yield json.dumps({"response": content}) + "\n"

        except Exception as e:
            yield json.dumps({"response": f"\n[Błąd połączenia: {str(e)}]"}) + "\n"

    return StreamingResponse(
        proxy_stream(),
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
