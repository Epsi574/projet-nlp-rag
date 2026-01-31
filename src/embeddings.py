# src/embeddings.py
from __future__ import annotations
from typing import List
import numpy as np

class Embedder:
    """
    Wrap sentence-transformers.
    - model_name aligns with your choix_parametres.md (all-MiniLM-L6-v2).
    - normalize=True to enable cosine similarity via inner product.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", normalize: bool = True):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        ).astype("float32")

        if self.normalize:
            # L2 normalize rows for cosine similarity with inner product
            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
            emb = emb / norms
        return emb
