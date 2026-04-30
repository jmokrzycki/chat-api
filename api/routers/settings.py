from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.settings import get_settings_data, save_settings_data

router = APIRouter(tags=["Settings"])

class SettingsData(BaseModel):
    template: str
    rephrase_template: str
    history_limit: int
    memory_enabled: bool

@router.get("/settings")
async def get_settings_route():
    return get_settings_data()

@router.post("/settings")
async def save_settings_route(data: SettingsData):
    try:
        save_settings_data(
            data.template,
            data.rephrase_template,
            data.history_limit,
            data.memory_enabled
        )
        return {"message": "Ustawienia zostały zapisane pomyślnie."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd zapisu: {str(e)}")