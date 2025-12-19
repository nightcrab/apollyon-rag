from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import json
import asyncio
from contextlib import asynccontextmanager

from files import save_file
from stateful_llm import StatefulLLM

from config import MODEL

OLLAMA_URL = "http://localhost:11434/api/generate"

sessions = dict()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    prompt: str
    session_id: str = "default_session"

def get_session(session_id):
    
    if session_id not in sessions:
        sessions[session_id] = StatefulLLM()

    llm = sessions[session_id]

    return llm

@app.post("/api/chat")
async def chat(req: ChatRequest):

    llm = get_session(req.session_id)

    return StreamingResponse(
        #stream_string("hello how are you"),
        llm.stream_completion(req.prompt),
        media_type="text/plain",
    )

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Form("default_session")
):
    try:
        outcome, file_path = save_file(file)

        response = JSONResponse(
            status_code=200,
            content=outcome
        )
        
        llm = get_session(session_id)
        print("adding file to session...")
        llm.add_file(file_path)
            
        return response
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"message": f"Error uploading file: {str(e)}"}
        )
    
    finally:
        await file.close()

@asynccontextmanager
async def lifespan(app: FastAPI):

    sessions["default_session"] = StatefulLLM(
        MODEL
    )

    yield

