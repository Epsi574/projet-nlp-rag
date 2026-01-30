# src/retriever.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
from src.embeddings import Embedder
from src.index import VectorIndex, IndexedChunk

@dataclass(frozen=True)
class Retrieved:
    chunk: IndexedChunk
    score: float

class Retriever:
    def __init__(self, index_dir: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", top_k: int = 5):
        self.index = VectorIndex.load(index_dir)
        self.embedder = Embedder(model_name=model_name, normalize=True)
        self.top_k = top_k

    def retrieve(self, question: str) -> List[Retrieved]:
        q_vec = self.embedder.encode([question], batch_size=1)[0]
        results = self.index.search(q_vec, top_k=self.top_k)
        return [Retrieved(chunk=ch, score=score) for ch, score in results]
