import os
from fastapi import APIRouter, UploadFile, File, HTTPException

from core.config import UPLOAD_DIR
from core.rag import delete_document_from_vector_store
from core.state import get_trained_files_list, remove_trained_file_from_list

router = APIRouter(tags=["Files"])

@router.post("/files")
async def upload_file_only(file: UploadFile = File(...)):
    try:
        filename = file.filename or "unknown_file"
        file_path = os.path.join(UPLOAD_DIR, filename)

        with open(file_path, "wb") as buffer:
            while content := await file.read(1024 * 1024):
                buffer.write(content)

        # Jeśli wgrywamy plik o tej samej nazwie, wyrzucamy go z listy wytrenowanych
        if filename in get_trained_files_list():
            remove_trained_file_from_list(filename)

        return {"message": f"Plik {filename} został przesłany i czeka na Stage.", "filename": filename}
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
            delete_document_from_vector_store(filename)
            return {"message": f"Plik {filename} usunięty."}
        raise HTTPException(status_code=404, detail="Plik nie istnieje.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd usuwania pliku: {str(e)}")