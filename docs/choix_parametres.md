# Choix des paramètres (Tronc commun RAG)

## Chunking
- Taille cible : **2000 caractères** (~300-400 mots) par chunk  
  Justification : compromis entre contexte suffisant et bruit limité, avec découpe intelligente sur paragraphes/phrases.
- Overlap : **300 caractères** (~50-60 mots)  
  Justification : éviter la perte d'information aux frontières (noms propres/dates), assurer continuité contextuelle.
- Stratégie : découpe prioritaire sur paragraphes (\\n\\n), puis fin de phrase (. ! ?), puis saut de ligne simple.

## Retrieval
- top-k : **5**  
  Justification : assez de rappel sans saturer le prompt.

## Embeddings
- Modèle : **sentence-transformers/all-MiniLM-L6-v2**  
  Justification : rapide, robuste, bon baseline dense retrieval.

## Index
- Base vectorielle : **FAISS (IndexFlatIP)** avec vecteurs **normalisés** (cosine ≈ inner product).

## Génération (API LLM) sdvjbg6
- Baseline : question → LLM
- RAG : question + top-k chunks → LLM
- Règle : si info absente du contexte, le modèle doit l’indiquer explicitement.
