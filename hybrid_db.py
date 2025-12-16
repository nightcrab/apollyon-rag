import numpy as np
from typing import List
import pathlib
import hashlib
import os
import joblib

def load_text(path: str) -> str:
    return pathlib.Path(path).read_text(encoding="utf-8")


def chunk_text(text: str, chunk_size: int = 150, overlap: int = 75) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


class HybridDB:
    """
    Keyword + Vector database created from local text file.
    """
    def __init__(self, path):
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import TfidfVectorizer

        text = load_text(path)
        chunks = chunk_text(text)

        self.checksum = hashlib.md5(text.encode('utf-8')).hexdigest()
        self.chunks = chunks

        # Keyword side
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
        )
        self.keyword_matrix = self.vectorizer.fit_transform(chunks)

        # Embedding side
        self.embedder_name = "all-MiniLM-L6-v2"
        self.embedder = SentenceTransformer(self.embedder_name)
        self.embeddings = self.embedder.encode(
            chunks,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        self.save(path + ".db")

    def search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.3,  # balance keyword vs semantic
    ):
        from sklearn.metrics.pairwise import cosine_similarity
        # Keyword score
        q_kw = self.vectorizer.transform([query])
        kw_scores = cosine_similarity(q_kw, self.keyword_matrix)[0]

        # Semantic score
        q_emb = self.embedder.encode(
            [query],
            normalize_embeddings=True,
        )[0]
        sem_scores = self.embeddings @ q_emb

        # Combine
        scores = alpha * kw_scores + (1 - alpha) * sem_scores
        idxs = scores.argsort()[::-1][:top_k]

        idxs = sorted(idxs)

        merged = []

        previous_idx = -2
        for idx in idxs:
            if idx == previous_idx+1:
                # remove overlap
                merged[-1] += " " + " ".join(self.chunks[idx].split(" ")[75:])
            else:
                merged.append(self.chunks[idx])
            previous_idx = idx

        return merged

    def save(self, path: str):
        joblib.dump(
            {
                "checksum": self.checksum,
                "chunks": self.chunks,
                "vectorizer": self.vectorizer,
                "keyword_matrix": self.keyword_matrix,
                "embeddings": self.embeddings,
                "embedder_name": self.embedder_name,
            },
            path,
        )

    @classmethod
    def load(cls, path: str):
        from sentence_transformers import SentenceTransformer
        data = joblib.load(path)

        obj = cls.__new__(cls)
        obj.checksum = data["checksum"]
        obj.chunks = data["chunks"]
        obj.vectorizer = data["vectorizer"]
        obj.keyword_matrix = data["keyword_matrix"]
        obj.embeddings = data["embeddings"]
        obj.embedder_name = data["embedder_name"]
        obj.embedder = SentenceTransformer(obj.embedder_name)

        return obj
