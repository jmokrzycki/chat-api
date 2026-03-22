import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/chat")
async def ask_bielik(data: dict):
    user_prompt = data.get("prompt", "Brak pytania")

    async def proxy_stream():
        async with httpx.AsyncClient() as client:
            async with client.stream(
                    "POST",
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "speakleash/bielik-11b-v3.0-instruct:Q4_K_M",
                        "prompt": user_prompt,
                        "stream": True
                    },
                    timeout=None
            ) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk

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