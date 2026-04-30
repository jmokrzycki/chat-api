from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import chat, settings, documents

app = FastAPI(title="Bielik RAG API")

app.add_middleware(
    CORSMiddleware, # type: ignore
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(settings.router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)