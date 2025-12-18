from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
import json
import asyncio
from contextlib import asynccontextmanager

from llm import StatefulLLM

app = FastAPI()

OLLAMA_URL = "http://localhost:11434/api/generate"

sessions = dict()

class ChatRequest(BaseModel):
    prompt: str
    session_id: str = "default_session"

async def stream_string(text: str, delay: float = 0.05):
    for char in text:
        yield char
        await asyncio.sleep(delay)

@app.post("/api/chat")
async def chat(req: ChatRequest):

    if req.session_id not in sessions:
        sessions[req.session_id] = StatefulLLM()

    llm = sessions[req.session_id]

    return StreamingResponse(
        #stream_string("hello how are you"),
        llm.stream_completion(req.prompt),
        media_type="text/plain",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):

    sessions["default_session"] = StatefulLLM(
        "granite4:1b"
    )

    yield

