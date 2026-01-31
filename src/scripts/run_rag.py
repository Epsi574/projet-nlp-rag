from __future__ import annotations
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import hnswlib
import certifi
import csv
from pathlib import Path
from dotenv import load_dotenv

import sys
import os
import certifi

# Force httpx / transformers à utiliser le certificat CA de certifi pour venv
os.environ["SSL_CERT_FILE"] = certifi.where()
# Ajouter le répertoire parent au path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


from src.llm_client import LLMClient
from src.questions_parser import parse_questions_from_text

os.environ["SSL_CERT_FILE"] = certifi.where()

# ----------- Model -----------  
model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------- Data loading -----------  
chunks = []   # liste de dicts: {"text": ..., "source": ...}
texts = []    # فقط les textes pour embeddings

with open('data/chunks.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        chunks.append({
            "text": item["text"],
            "source": item["title"]
        })
        texts.append(item["text"])

# ----------- Indexation embeddings -----------  
embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

dim = embeddings.shape[1]
index = hnswlib.Index(space='cosine', dim=dim)
index.init_index(max_elements=len(texts), ef_construction=200, M=16)
index.add_items(embeddings, np.arange(len(texts)))
index.set_ef(50)

# ----------- RAG search -----------  
def search(query, model, index, chunks, k=5):
    query_vec = model.encode([query], convert_to_numpy=True)
    query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
    labels, distances = index.knn_query(query_vec, k=k)

    results = []
    for i in labels[0]:
        results.append({
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
    """
    Construit le prompt RAG strict avec contexte et sources.
    
    Args:
        question: la question posée par l'utilisateur
        res: liste de dictionnaires {"source": ..., "text": ..., "id": ...} 
             représentant les chunks récupérés

    Returns:
        prompt complet prêt à envoyer au LLM
    """
    # Construction du contexte avec tous les chunks
    context = ""
    for r in res:
        context += f"SOURCE: {r['source']} (chunk={r.get('id','?')})\nTEXT: {r['text']}\n{'-'*60}\n"

    prompt = f"{BASELINE_SYSTEM_PROMPT}\nCONTEXTE :\n{context}\nQUESTION :\n{question}"
    return prompt



def main() -> None:
    load_dotenv()

    api_url = os.getenv("LLM_API_URL", "").strip()
    print("LLM_API_URL =", api_url)
    ollama_model = os.getenv("LLM_MODEL", "llama3").strip()
    questions_file = os.getenv("QUESTIONS_FILE", "QuestionsRéponses.txt").strip()
    out_csv = os.getenv("OUT_CSV", "RAG_outputs.csv").strip()

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

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "section", "question", "answer"])
        writer.writeheader()

        for i, q in enumerate(questions, start=1):
            prompt = build_rag_prompt(q["question"], search(q["question"], model, index, chunks, k=3))
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
            print("A:", answer)

    print("\nTerminé.")
    print(f"Résultats écrits dans: {out_csv}")


if __name__ == "__main__":
    main()