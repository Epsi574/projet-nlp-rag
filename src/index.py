# src/index.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict
import json
import numpy as np

@dataclass
class IndexedChunk:
    chunk_id: str
    doc_id: str
    source: str
    text: str

class VectorIndex:
    """
    FAISS index with:
    - index.faiss (vectors)
    - chunks_meta.jsonl (vector_id -> chunk metadata + text)
    """
    def __init__(self):
        self.index = None
        self.meta: List[IndexedChunk] = []

    @staticmethod
    def build(vectors: np.ndarray, chunks: List[Dict]) -> "VectorIndex":
        import faiss

        if vectors.dtype != np.float32:
            vectors = vectors.astype("float32")

        d = vectors.shape[1]
        idx = VectorIndex()
        idx.index = faiss.IndexFlatIP(d)  # cosine if vectors normalized
        idx.index.add(vectors)

        idx.meta = [
            IndexedChunk(
                chunk_id=c["chunk_id"],
                doc_id=c["doc_id"],
                source=c["source"],
                text=c["text"],
            )
            for c in chunks
        ]

        if len(idx.meta) != idx.index.ntotal:
            raise RuntimeError("Meta size and FAISS index size mismatch.")
        return idx

    def save(self, out_dir: str | Path) -> None:
        import faiss
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(out_dir / "index.faiss"))

        with (out_dir / "chunks_meta.jsonl").open("w", encoding="utf-8") as f:
            for ch in self.meta:
                f.write(json.dumps(ch.__dict__, ensure_ascii=False) + "\n")

    @staticmethod
    def load(out_dir: str | Path) -> "VectorIndex":
        import faiss
        out_dir = Path(out_dir)

        idx = VectorIndex()
        idx.index = faiss.read_index(str(out_dir / "index.faiss"))

        meta = []
        with (out_dir / "chunks_meta.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                meta.append(IndexedChunk(**json.loads(line)))
        idx.meta = meta

        if len(idx.meta) != idx.index.ntotal:
            raise RuntimeError("Meta size and FAISS index size mismatch.")
        return idx

    def search(self, query_vec: np.ndarray, top_k: int = 5):
        """
        query_vec: shape (d,) or (1,d), float32 normalized.
        returns: list of (IndexedChunk, score)
        """
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        scores, ids = self.index.search(query_vec.astype("float32"), top_k)
        results = []
        for i, score in zip(ids[0], scores[0]):
            if i == -1:
                continue
            results.append((self.meta[int(i)], float(score)))
        return results
