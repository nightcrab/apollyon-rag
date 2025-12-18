import requests
import subprocess
import json
import httpx
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor
import os
import time

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "ministral-3:14b"

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


    def answer(
        self, 
        prompt: str,
        verbose: bool = False
    ) -> str:
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

        ret = data.get("response", "").strip()

        if verbose:
            print(ret)

        return ret

    def batch_answer(
        self,
        prompts: List[str],
        max_workers: int = 16,
        verbose: bool = False
    ) -> List[str]:
        """
        Generate answers for a batch of prompts in parallel.
        """
        
        start_total = time.time()

        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self.answer, prompts))
    
        #for prompt in prompts:
            #results.append(self.answer(prompt))

        end_total = time.time()

        print(f"Batch processed in {end_total-start_total}s")

        return results

    def stop(self):
        subprocess.run(["ollama", "stop", self.model])


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

