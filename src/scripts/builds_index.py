# src/scripts/build_index.py
from __future__ import annotations
import argparse
import json
from pathlib import Path

from src.embeddings import Embedder
from src.index import VectorIndex

def load_chunks_jsonl(path: str | Path):
    chunks = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/artifacts/chunks.jsonl")
    ap.add_argument("--out_dir", default="data/artifacts/index")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    chunks = load_chunks_jsonl(args.chunks)
    texts = [c["text"] for c in chunks]

    emb = Embedder(model_name=args.model, normalize=True)
    vectors = emb.encode(texts, batch_size=args.batch_size)

    vindex = VectorIndex.build(vectors, chunks)
    vindex.save(args.out_dir)

    print(f"[OK] Built index with {len(chunks)} vectors in {args.out_dir}")

if __name__ == "__main__":
    main()
