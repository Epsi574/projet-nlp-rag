from __future__ import annotations
import csv
import os
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


BASELINE_SYSTEM_PROMPT = """Tu es un assistant. Réponds de façon concise et factuelle.
Si tu n'es pas sûr, dis-le explicitement.
"""


def build_baseline_prompt(question: str) -> str:
    # Baseline = pas de contexte externe (tronc commun)
    return f"{BASELINE_SYSTEM_PROMPT}\nQuestion: {question}\nRéponse:"


def main() -> None:
    load_dotenv()

    api_url = os.getenv("LLM_API_URL", "").strip()
    print("LLM_API_URL =", api_url)
    model = os.getenv("LLM_MODEL", "llama3").strip()
    questions_file = os.getenv("QUESTIONS_FILE", "QuestionsRéponses.txt").strip()
    out_csv = os.getenv("OUT_CSV", "baseline_outputs.csv").strip()

    client = LLMClient(api_url, model)

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
            prompt = build_baseline_prompt(q["question"])
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
