import numpy as np
from typing import List, Optional, Dict, Any
import pathlib
import hashlib
import joblib
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

from llm import StatelessLLM

class Chunker:
    """Chunker using RecursiveCharacterTextSplitter"""
    
    def __init__(
        self,
        chunk_size: int = 300,
        chunk_overlap: int = 150,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        model_path: str = "granite4:1b"
    ):
        if separators is None:
            separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]
            
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            keep_separator=keep_separator,
            length_function=lambda text: len(text.split())  # Count by words
        )

        self.model_path = model_path
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        documents = self.text_splitter.create_documents([text])
        return [doc.page_content for doc in documents]

    def create_titles(self, chunks):
        print("Creating chunk titles...")
        titles = []
        llm = StatelessLLM(
            self.model_path,
            max_tokens=32
        )
        for chunk in chunks[::2]:
            title = llm.answer(f"Provide a short title in less than 16 tokens describing the contents of this document chunk. Do not write anything else.\n{chunk}")
            titles.append(title)
            print(title)
        print("Done creating titles.")
        return titles
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        return self.text_splitter.split_documents(documents)


class HybridDB:
    """
    Keyword + Vector database created from local text file.
    """
    def __init__(
        self,
        path: str,
        chunk_size: int = 300,
        overlap: int = 100,
        embedder_name: str = "all-MiniLM-L6-v2"
    ):
        # Load text using LangChain
        loader = TextLoader(path, encoding="utf-8")
        documents = loader.load()
        
        # Get full text for checksum
        text = documents[0].page_content
        self.checksum = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        # Chunk document
        self.chunker = Chunker(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )
        chunked_docs = self.chunker.chunk_documents(documents)
        self.chunks = [doc.page_content for doc in chunked_docs]
        self.documents = chunked_docs  # Keep Document objects if needed
        self.titles = self.chunker.create_titles(self.chunks)

        # background corpus for tf-idf
        import nltk
        from nltk.corpus import brown
        from nltk.corpus import gutenberg

        nltk.download('brown')
        nltk.download('gutenberg')

        background_docs = [" ".join(brown.words(fileid)) for fileid in brown.fileids()][:100]
        background_docs += [gutenberg.raw(f) for f in gutenberg.fileids()][:100]
        
        # Keyword side
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1,3),
            sublinear_tf=True,
        )
        tfidf_matrix = self.vectorizer.fit_transform(self.chunks + background_docs)

        self.keyword_matrix = tfidf_matrix[:len(self.chunks)]
        
        # Embedding side
        self.embedder_name = embedder_name
        self.embedder = SentenceTransformer(self.embedder_name)
        self.embeddings = self.embedder.encode(
            self.chunks,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        
        self.save(path + ".db")
    
    def _1d_filter(self, scores, side_weight=0.3, center_weight=0.4):
        """Apply 1D context filter to scores"""
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
        k: int = 60, 
        excluded_idxs: list = None
    ):
        """Search using RRF hybrid approach"""
        if excluded_idxs is None:
            excluded_idxs = []
        
        # Get keyword scores and ranks
        q_kw = self.vectorizer.transform([query])
        kw_scores = cosine_similarity(q_kw, self.keyword_matrix)[0]
        kw_ranks = (-kw_scores).argsort().argsort() + 1  # Convert to ranks (1-based)
        
        # Get semantic scores and ranks
        q_emb = self.embedder.encode(
            [query],
            normalize_embeddings=True,
        )[0]
        sem_scores = self.embeddings @ q_emb
        sem_ranks = (-sem_scores).argsort().argsort() + 1  # Convert to ranks (1-based)
        
        # RRF
        rrf_scores = np.zeros_like(kw_scores)
        for i in range(len(kw_scores)):
            rrf_scores[i] = 1/(k + kw_ranks[i]) + 1/(k + sem_ranks[i])
        
        # Apply convolution filter
        scores = self._1d_filter(rrf_scores)
        
        # Exclude specified indices
        for idx in excluded_idxs:
            if 0 <= idx < len(scores):
                scores[idx] = -np.inf
        
        # Get top-k indices
        idxs = scores.argsort()[::-1][:top_k]
        
        # don't merge chunks
        return [self.chunks[x] for x in idxs], idxs

    def contextual_search(
        self,
        query: str,
        top_k_direct: int = 2,
        top_k: int = 5,
        similarity_threshold: float = 0.1,
        excluded_idxs: list = None
    ):
        """
        Find related chunks using TF-IDF keyword similarity
        """
        if excluded_idxs is None:
            excluded_idxs = []
        
        # STEP 1: Get direct matches
        direct_matches, direct_indices = self.search(
            query, 
            top_k=top_k_direct,
            excluded_idxs=excluded_idxs
        )
        
        # STEP 2: Get TF-IDF vectors for direct matches
        direct_tfidf = self.keyword_matrix[direct_indices]
        
        # Average TF-IDF vectors of direct matches to get combined keyword profile
        combined_vector = direct_tfidf.mean(axis=0)
        combined_vector = csr_matrix(combined_vector)
        
        # STEP 3: Find chunks similar to this keyword profile
        # Compute cosine similarity between combined vector and all chunks
        similarities = cosine_similarity(combined_vector, self.keyword_matrix)[0]
        
        # Apply context filter
        similarities = self._1d_filter(similarities)
        
        # STEP 4: Filter and rank
        # Remove already selected indices
        all_selected = set([*direct_indices, *excluded_idxs])
        
        for i in range(len(similarities)):
            if i in all_selected:
                similarities[i] = -1
            elif similarities[i] < similarity_threshold:
                similarities[i] = -1
        
        # Get top related indices
        top_related_count = max(0, top_k - len(direct_indices))
        related_indices = similarities.argsort()[::-1][:top_related_count]
        
        # Filter out invalid indices
        related_indices = [idx for idx in related_indices if similarities[idx] > similarity_threshold]
        
        # Combine results
        all_indices = list(direct_indices) + list(related_indices)
        
        # Merge consecutive chunks
        return self._merge_chunks(all_indices), all_indices

    def _merge_chunks(self, indices: List[int]) -> List[str]:
        """Merge consecutive chunks while removing overlap"""
        if not indices:
            return []

        merged = []
        previous_idx = -2
        current_chunk = ""

        for idx in sorted(indices):
            chunk = self.chunks[idx]
            
            if previous_idx == -2:
                # First chunk, just add it
                merged.append(chunk)
            elif idx == previous_idx + 1:
                prev_chunk = merged[-1]
                # Find overlap size by comparing suffix of previous chunk with prefix of current chunk
                overlap_size = 0
                max_overlap = min(len(prev_chunk), len(chunk))
                # simple heuristic: check longest suffix-prefix match
                for i in range(1, max_overlap + 1):
                    if prev_chunk[-i:] == chunk[:i]:
                        overlap_size = i
                # Merge without repeating overlap
                merged[-1] += " " + chunk[overlap_size:]
            else:
                # Non-consecutive chunk, just append
                merged.append(chunk)

            previous_idx = idx

        return merged


    def save(self, path: str):
        """Save database to disk"""
        joblib.dump(
            {
                "checksum": self.checksum,
                "chunks": self.chunks,
                "documents": self.documents,
                "titles": self.titles,
                "vectorizer": self.vectorizer,
                "keyword_matrix": self.keyword_matrix,
                "embeddings": self.embeddings,
                "embedder_name": self.embedder_name,
            },
            path,
        )
    
    @classmethod
    def load(cls, path: str):
        """Load database from disk"""
        data = joblib.load(path)
        
        obj = cls.__new__(cls)
        obj.checksum = data["checksum"]
        obj.chunks = data["chunks"]
        obj.documents = data.get("documents", [])
        obj.titles = data.get("titles", [])
        obj.vectorizer = data["vectorizer"]
        obj.keyword_matrix = data["keyword_matrix"]
        obj.embeddings = data["embeddings"]
        obj.embedder_name = data["embedder_name"]
        obj.embedder = SentenceTransformer(obj.embedder_name)
        obj.chunker = Chunker()  # Default chunker
        
        return obj


class SearchContext:
    """Context manager for search with history"""
    
    def __init__(
        self,
        database: HybridDB,
        top_k: int = 5,
    ):
        self.database = database
        self.history = []
        self.top_k = top_k
    
    def search(self, query: str) -> List[str]:
        """Search with memory state"""
        
        total_chunks = []

        chunks, idxs = self.database.contextual_search(
            query,
            top_k=self.top_k,
            excluded_idxs=self.history
        )

        self.history.extend(idxs)

        total_chunks.extend(chunks)
        
        return total_chunks
    

    def reset(self):
        """Reset search history"""
        self.history = []