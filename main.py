import json
from typing import cast
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionChunk

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

@app.post("/api/chat")
async def ask_bielik(data: dict):
    user_prompt = str(data.get("prompt", "Brak pytania"))

    messages =[ChatCompletionUserMessageParam(role="user", content=user_prompt)]

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