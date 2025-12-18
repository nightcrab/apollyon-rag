import requests
import subprocess
import json
import httpx
from typing import Optional
import os

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "granite4:1b"

class StatelessLLM:
    """
    Stateless wrapper for language model using ollama.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 32,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        install_model(self.model)


    def answer(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
            "stream": False,
        }

        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        return data.get("response", "").strip()

    def stop(self):
        subprocess.run(["ollama", "stop", self.model])


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


    def build_full_prompt(self, user_prompt: str) -> str:
        messages = []

        for role, content in self.history:
            messages.append(f"{role}: {content}")

        messages.append(f"user: {user_prompt}")
        return "\n".join(messages)


    def update_history(self, prompt, reply):
        self.history.append(("user", prompt))
        self.history.append(("assistant", assistant_reply))

        self.history_size += utf8len(prompt)
        self.history_size += utf8len(assistant_reply)

        while self.history_size > self.max_history_size:
            self.history_size -= utf8len(self.history[0][1])
            self.history = self.history[1:]

    async def stream_completion(self, prompt: str):
        full_prompt = self.build_full_prompt(prompt)

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
                assistant_reply = ""

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    data = json.loads(line)
                    print(data)
                    if "response" in data:
                        token = data["response"]
                        assistant_reply += token
                        yield token

def utf8len(s):
    return len(s.encode('utf-8'))


def install_model(model: str) -> None:
    """
    Install Ollama model if it doesn't already exist.
    """
    try:
        # Check if model exists
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=0.5)
        resp.raise_for_status()
        models = {m["name"] for m in resp.json().get("models", [])}
        if model in models:
            return
    except Exception:
        raise RuntimeError(
            "Could not connect to Ollama. Is `ollama serve` running?"
        )

    print(f"Pulling model: {model}")
    subprocess.run(
        ["ollama", "pull", model],
        check=True,
    )

