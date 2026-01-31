from __future__ import annotations
import json
import re
import csv
from pathlib import Path
from dotenv import load_dotenv

import numpy as np
import hnswlib
import certifi
import os
import sys

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# Force httpx / transformers à utiliser le certificat CA de certifi pour venv
os.environ["SSL_CERT_FILE"] = certifi.where()

# Ajouter le répertoire parent au path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.llm_client import LLMClient
from src.questions_parser import parse_questions_from_text


# -----------------------
# Tokenisation BM25 (simple et robuste)
# -----------------------
_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+", re.UNICODE)

def tokenize(text: str) -> list[str]:
    # Lowercase + extraction de mots (gère accents)
    return _WORD_RE.findall(text.lower())


# -----------------------
# Fusion RRF (Reciprocal Rank Fusion)
# -----------------------
def rrf_fuse(
    dense_ranked_ids: list[int],
    bm25_ranked_ids: list[int],
    rrf_k: int = 60
) -> list[int]:
    """
    dense_ranked_ids : liste d'indices chunks triés (meilleur -> moins bon)
    bm25_ranked_ids  : idem pour BM25
    rrf_k            : constante RRF (souvent 60)

    Retourne une liste d'indices fusionnés triée par score RRF décroissant.
    """
    scores: dict[int, float] = {}

    # Rang commence à 1 en RRF
    for rank, idx in enumerate(dense_ranked_ids, start=1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank)

    for rank, idx in enumerate(bm25_ranked_ids, start=1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank)

    # Tri par score RRF décroissant
    return [idx for idx, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


# -----------------------
# Chargement des chunks
# -----------------------
chunks: list[dict] = []     # {"id":..., "text":..., "source":...}
texts: list[str] = []

with open("data/chunks.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        chunks.append({
            "id": item.get("chunk_id", "?"),
            "text": item["text"],
            "source": item.get("title", item.get("source", "Unknown"))
        })
        texts.append(item["text"])


# -----------------------
# Modèle embeddings + index dense (HNSW)
# -----------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

dim = embeddings.shape[1]
index = hnswlib.Index(space="cosine", dim=dim)
index.init_index(max_elements=len(texts), ef_construction=200, M=16)
index.add_items(embeddings, np.arange(len(texts)))
index.set_ef(50)


# -----------------------
# Index BM25 (lexical)
# -----------------------
tokenized_corpus = [tokenize(t) for t in texts]
bm25 = BM25Okapi(tokenized_corpus)


# -----------------------
# Recherche dense / BM25 / hybride
# -----------------------
def dense_search(query: str, k: int) -> list[int]:
    q_vec = model.encode([query], convert_to_numpy=True)
    q_vec = q_vec / np.linalg.norm(q_vec, axis=1, keepdims=True)

    labels, _distances = index.knn_query(q_vec, k=k)
    return [int(i) for i in labels[0]]


def bm25_search(query: str, k: int) -> list[int]:
    q_tok = tokenize(query)
    scores = bm25.get_scores(q_tok)  # numpy array (n_chunks,)

    # top-k indices triés par score décroissant
    top_idx = np.argsort(scores)[::-1][:k]
    return [int(i) for i in top_idx]


def hybrid_search(query: str, k_dense: int = 10, k_bm25: int = 10, k_final: int = 5, rrf_k: int = 60) -> list[dict]:
    dense_ids = dense_search(query, k=k_dense)
    bm25_ids = bm25_search(query, k=k_bm25)

    fused_ids = rrf_fuse(dense_ids, bm25_ids, rrf_k=rrf_k)[:k_final]

    results = []
    for i in fused_ids:
        results.append({
            "id": chunks[i]["id"],
            "text": chunks[i]["text"],
            "source": chunks[i]["source"]
        })
    return results


BASELINE_SYSTEM_PROMPT = """Tu es un assistant de questions-réponses.

Règles strictes :
1) Réponds UNIQUEMENT à partir du CONTEXTE fourni.
2) Si l'information n'est pas présente dans le contexte, réponds : "Je ne peux pas répondre avec certitude à partir des sources fournies."
3) Ne complète pas avec des connaissances externes.
4) Donne une réponse concise, puis liste les sources sous forme de puces.

Format de sortie :
Réponse : <ta réponse>

Sources :
- <doc_id ou titre> (chunk=<id>)
- ...
"""


def build_rag_prompt(question: str, res: list[dict]) -> str:
    context = ""
    for r in res:
        context += f"SOURCE: {r['source']} (chunk={r.get('id','?')})\nTEXT: {r['text']}\n{'-'*60}\n"
    return f"{BASELINE_SYSTEM_PROMPT}\nCONTEXTE :\n{context}\nQUESTION :\n{question}"


def main() -> None:
    load_dotenv()

    api_url = os.getenv("LLM_API_URL", "").strip()
    ollama_model = os.getenv("LLM_MODEL", "llama3").strip()
    questions_file = os.getenv("QUESTIONS_FILE", "QuestionsRéponses.txt").strip()
    out_csv = os.getenv("OUT_CSV", "RAG_outputs.csv").strip()

    # Paramètres retrieval (tu peux les exposer aussi dans .env si vous voulez)
    K_DENSE = int(os.getenv("K_DENSE", "10"))
    K_BM25 = int(os.getenv("K_BM25", "10"))
    K_FINAL = int(os.getenv("K_FINAL", "5"))
    RRF_K = int(os.getenv("RRF_K", "60"))

    client = LLMClient(api_url, ollama_model)

    qpath = Path(questions_file)
    if not qpath.exists():
        raise FileNotFoundError(
            f"Fichier de questions introuvable: {questions_file}\n"
            f"Astuce: copie QuestionsRéponses.txt à la racine du repo, ou modifie QUESTIONS_FILE dans .env"
        )

    raw = qpath.read_text(encoding="utf-8", errors="ignore")
    questions = parse_questions_from_text(raw)
    if not questions:
        raise ValueError("Aucune question détectée. Vérifie le format du fichier QUESTIONS_FILE.")

    print(f"OK: {len(questions)} questions chargées depuis {questions_file}")
    print(f"Hybrid retrieval: K_DENSE={K_DENSE}, K_BM25={K_BM25}, K_FINAL={K_FINAL}, RRF_K={RRF_K}")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "section", "question", "answer"])
        writer.writeheader()

        for i, q in enumerate(questions, start=1):
            retrieved = hybrid_search(
                q["question"],
                k_dense=K_DENSE,
                k_bm25=K_BM25,
                k_final=K_FINAL,
                rrf_k=RRF_K
            )
            prompt = build_rag_prompt(q["question"], retrieved)
            answer = client.generate(prompt)

            writer.writerow({
                "id": q["id"],
                "section": q["section"],
                "question": q["question"],
                "answer": answer
            })

            print("\n" + "=" * 80)
            print(f"[{i}/{len(questions)}] {q['id']} ({q['section']})")
            print("Q:", q["question"])
            print("Retrieved sources:")
            for r in retrieved:
                print(f" - {r['source']} (chunk={r['id']})")
            print("A:", answer)

    print("\nTerminé.")
    print(f"Résultats écrits dans: {out_csv}")


if __name__ == "__main__":
    main()
