from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.rag import (
    process_and_add_document,
    clear_vector_store,
    get_cached_filenames
)
from core.state import (
    add_trained_files_to_list,
    get_trained_files_list,
    clear_trained_files_list,
    remove_trained_file_from_list
)

router = APIRouter(tags=["RAG Knowledge"])

class TrainData(BaseModel):
    filenames: List[str]

@router.get("/cache")
async def get_cached_files():
    try:
        files = get_cached_filenames()
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd pobierania cache: {str(e)}")

@router.post("/train")
async def train_rag(data: TrainData):
    try:
        processed = []
        for filename in data.filenames:
            process_and_add_document(filename)
            processed.append(filename)

        add_trained_files_to_list(processed)

        return {"message": f"Pomyślnie wczytano pliki: {', '.join(processed)}"}
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
        remove_trained_file_from_list(filename)
        return {"message": f"Plik {filename} usunięty z bazy wiedzy (zapamiętany w cache)."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd usuwania z RAG: {str(e)}")

@router.post("/reset")
async def reset_rag():
    try:
        clear_vector_store()
        clear_trained_files_list()
        return {"message": "Baza wiedzy i cache zostały pomyślnie wyczyszczone."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd podczas czyszczenia bazy: {str(e)}")