from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
import time
import numpy as np

class Chunker:
    """Chunker using RecursiveCharacterTextSplitter"""
    
    def __init__(
        self,
        chunk_size: int = 300,
        chunk_overlap: int = 150,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        model_path: str = "granite3.3:2b"
    ):
        if separators is None:
            separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        
        # Lazy imports inside __init__
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
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
        from llm import StatelessLLM
        
        print("Creating chunk titles...")
        titles = []
        llm = StatelessLLM(
            self.model_path,
            max_tokens=32
        )
        prompts = []

        for chunk in chunks[::2]:
            prompt = f"Provide a short title in less than 16 tokens describing the contents of this document chunk. Do not write anything else.\n{chunk}"
            prompts.append(prompt)

        titles = llm.batch_answer(prompts)

        print("Done creating titles.")
        return titles
    
    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        """Split documents into chunks"""
        return self.text_splitter.split_documents(documents)


class HybridDB:
    """
    Keyword + Vector database created from local text file.
    """
    def __init__(
        self,
        path: str = None,
        chunk_size: int = 300,
        overlap: int = 100,
        embedder_name: str = "all-MiniLM-L6-v2",
    ):
        self.chunks = []
        self.documents = []
        self.titles = []
        self.embeddings = []
        self.checksums = []
        self.keyword_matrix = None  # Will be initialized on first add

        self.save_path = None
        
        # Chunker and embedder are initialized lazily when needed
        self.embedder_name = embedder_name
        self._embedder = None
        self._chunker = None
        self._vectorizer = None

        if path:
            self.save_path = path + ".db"
            self._process_document(path)

    def _tokenize(self, text: str):
        return re.findall(r"[A-Za-z0-9_+-]{3,}", text.lower())

    @property
    def chunker(self):
        if self._chunker is None:
            self._chunker = Chunker()
        return self._chunker

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedder_name)
        return self._embedder

    @property
    def vectorizer(self):
        if self._vectorizer is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words=None,        
                min_df=1,               
                max_df=1.0,             
                max_features=None,      
                ngram_range=(1, 2),
                sublinear_tf=True
            )
        return self._vectorizer

    def _load_background_corpus(self):
        import nltk
        from nltk.corpus import brown, gutenberg

        nltk.download('brown', quiet=True)
        nltk.download('gutenberg', quiet=True)
        background_docs = [" ".join(brown.words(fileid)) for fileid in brown.fileids()][:10]
        background_docs += [gutenberg.raw(f) for f in gutenberg.fileids()][:10]
        return background_docs

    def _process_document(self, path: str):
        import pathlib
        import hashlib
        from langchain_community.document_loaders import TextLoader
        from langchain_core.documents import Document
        from sklearn.metrics.pairwise import cosine_similarity
        from scipy.sparse import csr_matrix

        if not path or not isinstance(path, str):
            raise ValueError("Invalid path provided.")

        pathlib.Path(path).resolve()  # Validate path exists indirectly

        # Load text
        loader = TextLoader(path, encoding="utf-8")
        documents = loader.load()

        # Compute checksum
        text = documents[0].page_content
        checksum = hashlib.md5(text.encode('utf-8')).hexdigest()
        self.checksums.append(checksum)

        # Chunk document
        chunked_docs = self.chunker.chunk_documents(documents)
        new_chunks = [doc.page_content for doc in chunked_docs]

        # Append to existing
        self.chunks.extend(new_chunks)
        self.documents.extend(chunked_docs)
        new_titles = self.chunker.create_titles(new_chunks)
        self.titles.extend(new_titles)

        # Embeddings: Encode only new chunks and append
        new_embeddings = self.embedder.encode(
            new_chunks,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        if len(self.embeddings):
            self.embeddings = np.vstack((self.embeddings, new_embeddings))
        else:
            self.embeddings = new_embeddings

        # TF-IDF: fit on background + new chunks, or transform new only
        background_docs = self._load_background_corpus()
        all_for_tfidf = new_chunks + background_docs

        if self.keyword_matrix is None:
            tfidf_matrix = self.vectorizer.fit_transform(all_for_tfidf)
            self.keyword_matrix = tfidf_matrix[:len(new_chunks)]
        else:
            new_tfidf = self.vectorizer.transform(new_chunks)
            old_dense = self.keyword_matrix.toarray() if hasattr(self.keyword_matrix, "toarray") else self.keyword_matrix
            combined = np.vstack((old_dense, new_tfidf.toarray()))
            self.keyword_matrix = combined

        if not self.save_path:
            self.save_path = f"{path}.db"

        print("Saving DB")
        self.save(self.save_path)


    def add_file(self, path: str) -> List[str]:
        """Add a new document file and process it incrementally."""
        try:
            print(f"Processing document {path}")

            self._process_document(path)

            print(f"Successfully added {path}")
        except Exception as e:
            print(f"Error adding {path}: {e}")


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
        from sklearn.metrics.pairwise import cosine_similarity

        if excluded_idxs is None:
            excluded_idxs = []
        
        # Get keyword scores and ranks
        q_kw = self.vectorizer.transform([query])
        
        if q_kw.nnz == 0:
            print("⚠️ Query has no in-vocab keywords")

        kw_scores = cosine_similarity(q_kw, self.keyword_matrix)[0]
        kw_ranks = (-kw_scores).argsort().argsort() + 1
        
        # Get semantic scores and ranks
        q_emb = self.embedder.encode(
            [query],
            normalize_embeddings=True,
        )[0]
        sem_scores = self.embeddings @ q_emb
        sem_ranks = (-sem_scores).argsort().argsort() + 1
        
        # RRF
        rrf_scores = np.zeros_like(kw_scores)
        for i in range(len(kw_scores)):
            rrf_scores[i] = 1/(k + kw_ranks[i]) + 1/(k + sem_ranks[i])
        
        # Apply convolution filter
        scores = self._1d_filter(rrf_scores)
        
        # Exclude specified indices
        excluded_idxs = set(excluded_idxs)

        for idx in excluded_idxs:
            if 0 <= idx < len(scores):
                scores[idx] = -np.inf
        
        # Get top-k indices
        idxs = scores.argsort()[::-1][:top_k]

        idxs = list(set([idx for idx in idxs if idx not in excluded_idxs]))
        
        return [self.chunks[x] for x in idxs], idxs

    def contextual_search(
        self,
        query: str,
        top_k: int = 2,
        similarity_threshold: float = 0.1,
        excluded_idxs: list = None
    ):
        from sklearn.metrics.pairwise import cosine_similarity
        from scipy.sparse import csr_matrix

        if excluded_idxs is None:
            excluded_idxs = []
        
        # 2-step search using tf-idf
        
        top_k_direct = max(1, top_k // 2)
        print(top_k_direct)

        direct_matches, direct_indices = self.search(
            query, 
            top_k=top_k_direct,
            excluded_idxs=excluded_idxs
        )
        
        direct_tfidf = self.keyword_matrix[direct_indices]
        
        # TF-IDF vectors
        combined_vector = direct_tfidf.mean(axis=0)
        combined_vector = csr_matrix(combined_vector)
        feature_names = self.vectorizer.get_feature_names_out()

        similarities = cosine_similarity(combined_vector, self.keyword_matrix)[0]
        
        # smoothing filter
        similarities = self._1d_filter(similarities)
        
        excluded_idxs = set(list(direct_indices) + excluded_idxs)
        
        top_related_count = max(0, top_k - len(direct_indices))

        for idx in range(len(similarities)):
            if idx < similarity_threshold:
                similarities[idx] = -1
            if idx in excluded_idxs:
                similarities[idx] = -1

        related_indices = similarities.argsort()[::-1][:top_related_count]

        related_indices = [
            idx for idx in related_indices if 
            similarities[idx] > similarity_threshold and
            idx not in excluded_idxs
        ]
        
        all_indices = sorted(set(list(direct_indices) + list(related_indices)))
        
        return self._merge_chunks(all_indices), all_indices

    def _merge_chunks(self, indices: List[int]) -> List[str]:
        """Merge consecutive chunks while removing overlap"""
        if not indices:
            return []

        merged = []
        previous_idx = -2

        for idx in sorted(indices):
            chunk = self.chunks[idx]
            
            if previous_idx == -2:
                merged.append(chunk)
            elif idx == previous_idx + 1:
                prev_chunk = merged[-1]
                overlap_size = 0
                max_overlap = min(len(prev_chunk), len(chunk))
                for i in range(1, max_overlap + 1):
                    if prev_chunk[-i:] == chunk[:i]:
                        overlap_size = i
                        break  # Use longest match found
                merged[-1] += " " + chunk[overlap_size:]
            else:
                merged.append(chunk)

            previous_idx = idx

        return merged

    def save(self, path: str):
        """Save database to disk"""
        import joblib

        joblib.dump(
            {
                "checksums": self.checksums,  # Fixed: was self.checksum (singular)
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
        import joblib

        data = joblib.load(path)
        
        obj = cls.__new__(cls)
        obj.checksums = data.get("checksums", data.get("checksum", []))  # Backward compat
        obj.chunks = data["chunks"]
        obj.documents = data.get("documents", [])
        obj.titles = data.get("titles", [])
        obj._vectorizer = data["vectorizer"]
        obj.keyword_matrix = data["keyword_matrix"]
        obj.embeddings = data["embeddings"]
        obj.embedder_name = data["embedder_name"]
        obj._embedder = SentenceTransformer(obj.embedder_name)
        obj._chunker = Chunker()
        
        return obj


class SearchContext:
    """Context manager for search with history"""
    
    def __init__(
        self,
        database: HybridDB,
        top_k: int = 3,
    ):
        self.database = database
        self.history = []
        self.top_k = top_k
    
    def search(self, query: str) -> List[str]:
        """Search with memory state"""
        chunks, idxs = self.database.contextual_search(
            query,
            top_k=self.top_k,
            excluded_idxs=self.history
        )

        self.history.extend(idxs)
        print(self.history)
        return chunks
    
    def reset(self):
        """Reset search history"""
        self.history = []