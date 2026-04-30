import os
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from core.config import UPLOAD_DIR
from core.rag import (
    process_and_add_document,
    clear_vector_store,
    delete_document_from_vector_store
)
from core.settings import (
    add_trained_files_to_list,
    get_trained_files_list,
    clear_trained_files_list,
    remove_trained_file_from_list
)

router = APIRouter(tags=["Documents"])

class TrainData(BaseModel):
    filenames: List[str]

@router.post("/files")
async def upload_file_only(file: UploadFile = File(...)):
    try:
        filename = file.filename or "unknown_file"
        file_path = os.path.join(UPLOAD_DIR, filename)

        with open(file_path, "wb") as buffer:
            while content := await file.read(1024 * 1024):
                buffer.write(content)

        return {"message": f"Plik {filename} został przesłany na Stage.", "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd przesyłania: {str(e)}")

@router.get("/files")
async def list_files():
    try:
        if not os.path.exists(UPLOAD_DIR):
            return {"files": []}
        files = [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))]
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd pobierania listy plików: {str(e)}")

@router.delete("/files/{filename}")
async def delete_file(filename: str):
    try:
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            return {"message": f"Plik {filename} usunięty."}
        raise HTTPException(status_code=404, detail="Plik nie istnieje.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd usuwania pliku: {str(e)}")

@router.post("/train")
async def train_rag(data: TrainData):
    try:
        processed = []
        for filename in data.filenames:
            process_and_add_document(filename)
            processed.append(filename)

        add_trained_files_to_list(processed)

        return {"message": f"Pomyślnie przetworzono pliki: {', '.join(processed)}"}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd przetwarzania: {str(e)}")

@router.get("/trained")
async def get_trained_files():
    return {"files": get_trained_files_list()}

@router.delete("/trained/{filename}")
async def delete_trained_file(filename: str):
    try:
        delete_document_from_vector_store(filename)
        remove_trained_file_from_list(filename)

        return {"message": f"Plik {filename} usunięty z bazy wiedzy."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd usuwania z RAG: {str(e)}")

@router.post("/reset")
async def reset_rag():
    try:
        clear_vector_store()
        clear_trained_files_list()
        return {"message": "Baza wiedzy została pomyślnie wyczyszczona."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd podczas czyszczenia bazy: {str(e)}")