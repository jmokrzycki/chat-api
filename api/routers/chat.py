from typing import List, Dict
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.rag import generate_chat_response

router = APIRouter(tags=["Chat"])

class ChatRequest(BaseModel):
    prompt: str
    template: str
    rephrase_template: str
    history: List[Dict[str, str]]

@router.post("/chat")
async def ask_bielik(data: ChatRequest):
    return StreamingResponse(
        generate_chat_response(
            data.prompt,
            data.template,
            data.rephrase_template,
            data.history
        ),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )