from __future__ import annotations
import re
from typing import List, Dict


_Q_RE = re.compile(r"^\s*(\d+)\s+(.*\S)\s*$")


def parse_questions_from_text(raw: str) -> List[Dict]:
    """
    Parse le fichier de Niamouille (questions/réponses) en extrayant uniquement les questions.

    Règle :
    - Une question est une ligne qui commence par un numéro + un espace
      (ex: "1 Quand débute l’entre-deux-guerres ?")
    - Les lignes de réponse (souvent indentées / sans numéro) sont ignorées.
    - Les sections "Niveau facile", "Moyen", "Difficile" etc. sont ignorées.
    """
    questions: List[Dict] = []
    section = "unknown"

    for line in raw.splitlines():
        s = line.strip()

        if not s:
            continue

        # Détection simple des sections
        low = s.lower()
        if "niveau facile" in low:
            section = "easy"
            continue
        if low == "moyen":
            section = "medium"
            continue
        if low == "difficile":
            section = "hard"
            continue

        m = _Q_RE.match(line)
        if m:
            qnum = m.group(1)
            qtxt = m.group(2).strip()
            # heuristique : garder uniquement les lignes qui finissent par ? ou semblent être une question
            if "?" in qtxt:
                questions.append({"id": f"{section}-{qnum}", "question": qtxt, "section": section})

    return questions
