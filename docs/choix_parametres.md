# Choix des paramètres (Tronc commun RAG)

## Chunking
- Taille cible : **350 mots** par chunk  
  Justification : compromis entre contexte suffisant et bruit limité.
- Overlap : **80 mots**  
  Justification : éviter la perte d'information aux frontières (noms propres/dates).

## Retrieval
- top-k : **5**  
  Justification : assez de rappel sans saturer le prompt.

## Embeddings
- Modèle : **sentence-transformers/all-MiniLM-L6-v2**  
  Justification : rapide, robuste, bon baseline dense retrieval.

## Index
- Base vectorielle : **FAISS (IndexFlatIP)** avec vecteurs **normalisés** (cosine ≈ inner product).

## Génération (API LLM)
- Baseline : question → LLM
- RAG : question + top-k chunks → LLM
- Règle : si info absente du contexte, le modèle doit l’indiquer explicitement.
