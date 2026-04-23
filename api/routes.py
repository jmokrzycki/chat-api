import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from core.config import UPLOAD_DIR
from core.rag import process_and_add_document, clear_vector_store, generate_chat_response

router = APIRouter(prefix="/api")

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        filename = file.filename or "unknown_file"
        file_path = os.path.join(UPLOAD_DIR, filename)

        with open(file_path, "wb") as buffer:
            while content := await file.read(1024 * 1024):
                buffer.write(content)

        process_and_add_document(filename)

        return {"message": f"Plik {filename} został pomyślnie przetworzony."}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd przetwarzania: {str(e)}")


@router.post("/reset")
async def reset_rag():
    try:
        clear_vector_store()
        return {"message": "Baza wiedzy została pomyślnie wyczyszczona."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd podczas czyszczenia bazy: {str(e)}")


@router.post("/chat")
async def ask_bielik(data: dict):
    user_prompt = str(data.get("prompt", "Brak pytania"))
    custom_template = str(data.get("template", ""))

    return StreamingResponse(
        generate_chat_response(user_prompt, custom_template),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
