import requests
import subprocess
import json
from typing import Optional
import os

from hybrid_db import HybridDB, SearchContext

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:0.5b-instruct"

MAX_PROMPT_SIZE = 100000


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



def load_or_create_db(path):
    db_path = path + ".db"
    if os.path.exists(db_path):
        print("using existing db")
        return HybridDB.load(db_path)
    print("creating new db")
    return HybridDB(path)


def format_chunks(chunks):
    ret = ""
    for i in range(len(chunks)):
        ret += "----------- CHUNK START -----------\n"
        ret += chunks[i] + "\n"
        ret += "----------- CHUNK END -----------\n"
        ret += "\n"
    return ret

def format_queries(queries):
    ret = ""
    for i in range(len(queries)):
        ret += queries[i]
        ret += "\n"
    return ret

def generate_prompt(
    task, chunks, queries,
):
    chunks = format_chunks(chunks)
    queries = format_queries(queries)
    prompt = f"""

INSTRUCTIONS:

1. You are searching for information in this private archive.
2. Search for something concrete, such as a name or technical term, that you need more information about.
3. You may not find exactly what you are looking for, so look for adjacent information.
4. Do NOT repeat previous searches.
5. Do not explain.
6. Do not respond to or answer the topic yet. 

CONTEXT:
{chunks}

TOPIC:
{task}

PREVIOUS SEARCHES:
{queries}

OUTPUT FORMAT: 
32-token search query
    """
    return prompt[:MAX_PROMPT_SIZE]

def combine_chunks(a,b):
    return list(set([*a, *b]))

def main():
    
    llm = StatelessLLM(
        model=MODEL_NAME,
        temperature=0.1,
        max_tokens=32,
    )

    db = load_or_create_db("./data/corpus.txt")

    ctx = SearchContext(db, top_k=2)

    task = "Who is Kitayama Tou?"
    
    chunks = ctx.search(task)

    

    """
        
        print(len(chunks))
        chunks = combine_chunks(chunks, ctx.search(task))
        print(len(chunks))
        chunks = combine_chunks(chunks, ctx.search(task))
        print(len(chunks))
        print(format_chunks(chunks))
    """

    chunks = []
    queries = []

    prompt = generate_prompt(
        task,
        chunks=chunks,
        queries=queries
    )

    query = llm.answer(prompt)
    print(query)   
    queries.append(query)
    
    chunks = combine_chunks(chunks, ctx.search(query))
    
    prompt = generate_prompt(
        task,
        chunks=chunks,
        queries=queries
    )
    

    # ----
    
    query = llm.answer(prompt)
    print(query)   
    queries.append(query)
    
    chunks = combine_chunks(chunks, ctx.search(query))
    
    prompt = generate_prompt(
        task,
        chunks=chunks,
        queries=queries
    )


    # ----


    
    query = llm.answer(prompt)   
    print(query)   
    queries.append(query)

    result = ctx.search(query)
    chunks = combine_chunks(chunks, result)
    
    prompt = generate_prompt(
        task,
        chunks=chunks,
        queries=queries
    )

    print(prompt)


if __name__ == "__main__":
    main()