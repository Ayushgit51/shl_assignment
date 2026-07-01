"""
retrieval/engine.py
--------------------
Hybrid search over the SHL catalog:
  1. BM25 keyword search   - exact name matching (Java 8, OPQ32r etc.)
  2. Chroma semantic search - intent matching (personality test for leadership)
  3. Score fusion + optional metadata filters

Chroma uses onnxruntime internally — no separate model download needed.
Works with Python 3.14+.
"""

import re
import math
import numpy as np
from typing import List, Dict, Optional
from catalog.loader import CATALOG

# ─────────────────────────────────────────────
# BM25 (pure Python — always works)
# ─────────────────────────────────────────────

class BM25:
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = [self._tokenize(doc) for doc in corpus]
        self.N = len(self.corpus)
        self.avgdl = sum(len(d) for d in self.corpus) / max(self.N, 1)
        self.df: Dict[str, int] = {}
        for doc in self.corpus:
            for word in set(doc):
                self.df[word] = self.df.get(word, 0) + 1
        self.idf: Dict[str, float] = {}
        for word, freq in self.df.items():
            self.idf[word] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def get_scores(self, query: str) -> np.ndarray:
        query_tokens = self._tokenize(query)
        scores = np.zeros(self.N)
        for i, doc in enumerate(self.corpus):
            dl = len(doc)
            tf_map: Dict[str, int] = {}
            for word in doc:
                tf_map[word] = tf_map.get(word, 0) + 1
            score = 0.0
            for token in query_tokens:
                if token not in self.idf:
                    continue
                tf = tf_map.get(token, 0)
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += self.idf[token] * num / den
            scores[i] = score
        return scores


# ─────────────────────────────────────────────
# Retrieval Engine (BM25 + Chroma hybrid)
# ─────────────────────────────────────────────

class RetrievalEngine:
    def __init__(self):
        self.catalog = CATALOG
        self.texts = [item["search_text"] for item in self.catalog]
        self.use_chroma = False

        print("[Retrieval] Building BM25 index...")
        self.bm25 = BM25(self.texts)

        # Try Chroma for semantic search
        try:
            import chromadb
            from chromadb.utils import embedding_functions

            print("[Retrieval] Building Chroma index...")
            self.chroma_client = chromadb.Client()

            # Use Chroma's built-in embedding (onnxruntime, no HuggingFace download)
            self.embed_fn = embedding_functions.DefaultEmbeddingFunction()

            # Create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name="shl_catalog",
                embedding_function=self.embed_fn,
                metadata={"hnsw:space": "cosine"}
            )

            # Add documents in batches
            ids = [item["entity_id"] for item in self.catalog]
            documents = self.texts
            metadatas = [
                {
                    "name": item["name"],
                    "link": item["link"],
                    "test_type": item["test_type"],
                    "job_levels": "|".join(item["job_levels"]),
                    "keys": "|".join(item["keys"]),
                    "duration": item["duration"] or "",
                }
                for item in self.catalog
            ]

            # Add in batches of 50
            batch_size = 50
            for i in range(0, len(ids), batch_size):
                self.collection.add(
                    ids=ids[i:i+batch_size],
                    documents=documents[i:i+batch_size],
                    metadatas=metadatas[i:i+batch_size],
                )

            self.use_chroma = True
            print("[Retrieval] Hybrid mode (BM25 + Chroma) ready.")

        except Exception as e:
            print(f"[Retrieval] Chroma unavailable ({type(e).__name__}: {e}). Using BM25-only mode.")
            self.use_chroma = False

        print(f"[Retrieval] Ready. {len(self.catalog)} items indexed.")

    def _chroma_scores(self, query: str, top_n: int) -> Dict[str, float]:
        """Returns entity_id -> similarity score from Chroma."""
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_n, len(self.catalog)),
        )
        scores = {}
        ids = results["ids"][0]
        distances = results["distances"][0]
        for eid, dist in zip(ids, distances):
            # Chroma cosine returns distance (0=identical, 2=opposite)
            # Convert to similarity: 1 - dist/2
            scores[eid] = 1.0 - dist / 2.0
        return scores

    def search(
        self,
        query: str,
        top_k: int = 10,
        job_level_filter: Optional[str] = None,
        test_type_filter: Optional[List[str]] = None,
        semantic_weight: float = 0.55,
        bm25_weight: float = 0.45,
    ) -> List[Dict]:
        """
        Hybrid search. Returns top_k items.
        Soft filters: applied post-scoring, fallback to unfiltered if <3 results.
        """
        # BM25 scores
        bm25_scores = self.bm25.get_scores(query)
        max_bm25 = bm25_scores.max() or 1.0
        bm25_norm = bm25_scores / max_bm25

        # Hybrid fusion
        if self.use_chroma:
            chroma_map = self._chroma_scores(query, top_n=len(self.catalog))
            sem_scores = np.array([
                chroma_map.get(item["entity_id"], 0.0)
                for item in self.catalog
            ])
            hybrid = semantic_weight * sem_scores + bm25_weight * bm25_norm
        else:
            hybrid = bm25_norm

        ranked_indices = np.argsort(hybrid)[::-1]

        # Apply filters
        def passes_filter(item: Dict) -> bool:
            if job_level_filter:
                levels = " ".join(item.get("job_levels", [])).lower()
                if job_level_filter.lower() not in levels:
                    return False
            if test_type_filter:
                item_codes = [c.strip() for c in item["test_type"].split(",")]
                if not any(c in item_codes for c in test_type_filter):
                    return False
            return True

        filtered = [
            self.catalog[i] for i in ranked_indices
            if passes_filter(self.catalog[i])
        ]

        # Fallback to unfiltered if too few
        if len(filtered) < 3:
            filtered = [self.catalog[i] for i in ranked_indices]

        return filtered[:top_k]


# Singleton
_engine: Optional[RetrievalEngine] = None

def get_engine() -> RetrievalEngine:
    global _engine
    if _engine is None:
        _engine = RetrievalEngine()
    return _engine
