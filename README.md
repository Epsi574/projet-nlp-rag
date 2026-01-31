# Projet RAG (Retrieval-Augmented Generation)

Ce projet met en œuvre une pipeline complète de question-réponse augmentée par la recherche (RAG) sur un corpus historique, constitué de documents et sources liés à l'entre-deux-guerres (1919-1939), incluant traités, témoignages, textes officiels, extraits d'encyclopédies et documents d'archives.

Il permet de comparer les réponses d'un LLM seul (baseline) et d'un LLM assisté par retrieval (RAG), avec affichage des sources utilisées.

## Prérequis

- Python 3.10+ recommandé
- pip (gestionnaire de paquets Python)
- [Ollama](https://ollama.com/) ou un serveur LLM compatible API (ou accès à une API cloud)
- Jupyter Notebook pour la démo interactive


**Installer les dépendances**

```bash
pip install -r requirements.txt
```

**Configurer les variables d'environnement**

Créer un fichier `.env` à la racine du projet et renseigner :

```
LLM_API_URL=http://localhost:11434/api/generate  # ou URL de votre serveur LLM
LLM_MODEL=llama3  # ou le modèle souhaité
QUESTIONS_FILE=QuestionsRéponses.txt
OUT_CSV=RAG_outputs.csv
```

## Structure du projet

- `src/` : code source principal
  - `data_processor.py` : pipeline de préparation des données (scan, clean, chunk, embeddings)
  - `llm_client.py` : client pour interroger le LLM (Ollama, proxy, etc.)
  - `retriever.py`, `embeddings.py`, ... : modules utilitaires
  - `scripts/` : scripts CLI
    - `build_data_pipeline.py` : lance toute la préparation des données
    - `run_baseline.py` : exécute le LLM seul sur les questions
    - `run_rag.py` : exécute le pipeline RAG complet
- `data/` : données
  - `raw/` : textes bruts
  - `clean/` : textes nettoyés
  - `chunks.jsonl` : chunks générés
- `QuestionsRéponses.txt` : questions de test
- `demo_evaluation_rag.ipynb` : notebook de démonstration et d'évaluation

## Pipeline complète

1. **Préparation des données**

Lancer la pipeline complète (scan, clean, chunk, embeddings) :

```bash
python -m src.scripts.build_data_pipeline
```

2. **Exécution baseline (LLM seul)**

```bash
python -m src.scripts.run_baseline
```

3. **Exécution RAG (LLM + retrieval)**

```bash
python -m src.scripts.run_rag
```

Les résultats sont enregistrés dans des fichiers CSV (voir OUT_CSV dans .env).

4. **Démo interactive**

Lancer le notebook Jupyter pour tester, comparer et poser vos propres questions :

```bash
jupyter notebook demo_evaluation_rag.ipynb
```

## Conseils d'utilisation

- Placez vos documents texte dans `data/raw/` (format .txt, UTF-8 recommandé).
- Modifiez ou ajoutez des questions dans `QuestionsRéponses.txt` (format Q/A).
- Adaptez les paramètres dans `.env` selon votre environnement et vos besoins.
- Pour utiliser un LLM distant, renseignez l'URL et la clé API si nécessaire.

## Dépendances principales

- sentence-transformers
- hnswlib
- numpy, pandas
- python-dotenv
- requests
- tqdm
- jupyter, ipywidgets

## Auteurs / Encadrement

- Projet réalisé par Ella GADELLE, David MITSAKIS, Pierre PILLOT, Romain TANCREZ
- Encadrant : Ismail BOUAJAJA
