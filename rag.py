import requests
import subprocess
import json
from typing import Optional
import os
import random

from hdb import HybridDB, SearchContext
from llm import StatelessLLM

MODEL_NAME = "ministral-3:14b"
MODEL_NAME_FINAL = "ministral-3:14b"

MAX_PROMPT_SIZE = 100000

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
        ret += f'"{queries[i]}"\n'
    return ret

def format_titles(titles):
    ret = ""
    for i in range(len(titles)):
        ret += f'"{titles[i]}"\n'
    return ret


def generate_prompt(
    task, chunks, queries, titles
):
    chunks = format_chunks(chunks)
    queries = format_queries(queries)
    titles = format_titles(titles)
    prompt = f"""

INSTRUCTIONS:

1. You are searching for as much information as you can in a corpus in relation to a final topic.
2. Search for something concrete, such as a name or technical term, that you need more information about.
3. You may not find exactly what you are looking for, so look for adjacent information.
4. Avoid repeating previous searches; try to find new information.
5. Do not explain.
6. Do not respond to or answer the topic question.
7. Do not waste tokens. 
8. Do not respond with anything except for the next search query.


TOPIC:
{task}

Here is a table of contents. This contains a list of documents that you may find helpful. In order to read their contents, search for their titles.

TABLE OF CONTENTS:
{titles}

Here are your previous searches. You will be doing more searches after this, so take your time. 

PREVIOUS SEARCHES:
{queries}

Try to search for something new and different.

Here are the documents you have read so far.

SEARCH RESULTS:
{chunks}    

SEARCH FORMAT: Text query up to 32 tokens.

Make the next thing you write the search query. Do NOT write anything such as the following:
"I'm ready to help you [...]"
"Based on the provided [...]"
"Here is my search query [...]"
You will jam the system and ruin the search machine. Don't say anything!

Do not just search the same things you already searched and do not just search the topic directly!
Investigate topics mentioned in the documents you already have found.

YOUR INPUT:
    """
    return prompt[:MAX_PROMPT_SIZE]

def combine_chunks(a,b):
    return list(set([*a, *b]))

class RAGInstance:
    """
    Contains state and methods for a persistent RAG session.
    """
    def __init__(
        self, 
        path: str = "./data/corpus.txt", 
        verbose: bool = False,
        max_titles: int = 128
    ):

        self.llm = StatelessLLM(
            model=MODEL_NAME,
            temperature=0.1,
            max_tokens=64, # more generous than the 32 stated
        )

        self.db = load_or_create_db(path)
        self.max_titles = max_titles

        self.ctx = SearchContext(self.db, top_k=2)
        self.verbose = verbose

        self.chunks = []
        self.queries = []

    def retrieve(
        self,
        task: str,
        iterations: int = 2
    ) -> str:
        """
        Make retrievals using iterative lookup.
        """

        self.chunks.extend(self.ctx.search(task))
        self.queries.append(task)

        new_chunks = []

        for x in range(iterations):

            prompt = generate_prompt(
                task,
                chunks=self.chunks,
                queries=self.queries,
                titles=random.sample( # randomise each time
                    self.db.titles, 
                    min(len(self.db.titles), self.max_titles)
                ) 
            )

            query = self.llm.answer(prompt)

            if self.verbose:
                print(f'query: {query}')

            self.queries.append(query)

            retrievals = self.ctx.search(query)
            new_chunks.extend(retrievals)
            
            self.chunks = combine_chunks(self.chunks, retrievals)

        if self.verbose:
            print(f"{len(new_chunks)} chunks retrieved")

        return format_chunks(new_chunks)