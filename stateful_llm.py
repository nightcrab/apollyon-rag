import requests
import subprocess
import json
import httpx
from typing import Optional
import os

import hdb
import rag
import llm

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "ministral-3:14b"

class StatefulLLM:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        max_history_size: int = 256000, # max size in bytes of the history - not tokens
        use_rag: bool = True
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.history = []
        self.history_size = 0
        self.max_history_size = max_history_size
        self.rag = rag.RAGInstance()

    def add_file(self, path):
        self.rag.add_file(path)

    def build_full_prompt(self) -> str:
        messages = []

        for role, content in self.history:
            messages.append(f"{role}: {content}")

        return "\n".join(messages)

    def clean_history(self):
        while self.history_size > self.max_history_size:
            self.history_size -= utf8len(self.history[0][1])
            self.history = self.history[1:]

    async def stream_completion(self, prompt: str):
        self.history.append(("user", prompt))

        if self.rag.contains_documents():
            retrievals = self.rag.retrieve(prompt, iterations=2)
            self.history.append(("context", retrievals))
            self.history_size += utf8len(retrievals)

        self.history_size += utf8len(prompt)

        full_prompt = self.build_full_prompt()

        print(full_prompt)
        
        assistant_reply = ""

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                    "stream": True,
                },
            ) as response:

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    data = json.loads(line)
                    if "response" in data:
                        token = data["response"]
                        assistant_reply += token
                        yield token

        self.history.append(("assistant", assistant_reply))
        self.history_size += utf8len(assistant_reply)

        self.clean_history()

        
def utf8len(s):
    return len(s.encode('utf-8'))
