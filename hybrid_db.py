import numpy as np
from typing import List
import pathlib
import hashlib
import os
import joblib
import re

def load_text(path: str) -> str:
    return pathlib.Path(path).read_text(encoding="utf-8")


def chunk_text(
    text: str,
    chunk_size: int = 300,
    overlap: int = 0
) -> List[str]:
    # Basic sentence splitter (works well for most English text)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_len = len(sentence_words)

        # If adding this sentence would exceed chunk size, finalize chunk
        if current_len + sentence_len > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))

            # Handle overlap
            if overlap > 0:
                overlap_words = " ".join(current_chunk).split()[-overlap:]
                current_chunk = [" ".join(overlap_words)]
                current_len = len(overlap_words)
            else:
                current_chunk = []
                current_len = 0

        current_chunk.append(sentence)
        current_len += sentence_len

    # Add final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

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

    def _1d_filter(self, scores, side_weight=0.25, center_weight=0.5):
        context_scores = scores.copy()
        for i in range(len(scores)):
            acc = center_weight * scores[i]
            if i > 0:
                acc += side_weight * scores[i - 1]
            if i < len(scores) - 1:
                acc += side_weight * scores[i + 1]
            context_scores[i] = acc
        return context_scores

    def search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,  # alpha*keyword + (1-alpha)*semantic
        excluded_idxs: list = []
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

        raw_scores = alpha * kw_scores + (1 - alpha) * sem_scores

        scores = self._1d_filter(raw_scores)
        
        for idx in excluded_idxs:
            if 0 <= idx < len(scores):
                scores[idx] = -np.inf 

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

        return merged, idxs

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


class SearchContext:
    def __init__(self, database, top_k = 5, alpha_max = 0.9, alpha_min = 0.5):
        self.database = database
        self.history = []
        self.top_k = top_k
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min

    def search(self, query):
        # uniform spread over alphas
        alphas = [
            (self.alpha_min + (1 / self.top_k) * x * self.alpha_max) 
            for x in range(self.top_k)
        ]

        total_chunks = []

        for alpha in alphas: 
            chunks, idxs = self.database.search(query, top_k = 1, excluded_idxs=self.history)
            self.history = [*self.history, *idxs]
            total_chunks = [*total_chunks, *chunks]

        return total_chunks
