# Sujet 01 : RAG Avancé (Retrieval-Augmented Generation)

## Rappel du sujet

Le **RAG (Retrieval-Augmented Generation)** est une technique qui combine la recherche d'information avec la génération de texte par un modèle de langage. Au lieu de se fier uniquement à la mémoire interne du modèle, le système recherche d'abord des documents pertinents dans une base de connaissances, puis utilise ces informations pour générer une réponse plus précise et mieux fondée.

**Cas d'usage typiques** :
- Systèmes de questions-réponses sur documentation technique
- Assistants conversationnels avec sources
- Synthèse de recherche documentaire
- Support client automatisé

## Tronc commun (obligatoire)

Pour valider ce sujet, tous les groupes travaillant sur ce sujet doivent réaliser **au minimum** les étapes suivantes :

1. **Préparation des données**
   - Choisir un corpus de documents (minimum 20-50 documents)
   - Nettoyer et prétraiter les textes
   - Découper les documents en chunks (fragments) de taille appropriée

2. **Indexation par embeddings**
   - Encoder les chunks avec un modèle d'embeddings (sentence-transformers ou équivalent)
   - Stocker les vecteurs dans une base vectorielle simple (FAISS, Chroma, ou équivalent)

3. **Pipeline de recherche + génération**
   - Implémenter une fonction de recherche qui :
     - Prend une question en entrée
     - Encode la question en vecteur
     - Recherche les top-k chunks les plus similaires
   - Implémenter une fonction de génération qui :
     - Construit un prompt avec la question et les chunks récupérés
     - Appelle une API de modèle de langage
     - Retourne la réponse générée

4. **Baseline simple**
   - Implémenter une version sans RAG (question directement au modèle)
   - Comparer avec la version RAG

5. **Démonstration**
   - Préparer 5-10 questions de test
   - Montrer les réponses avec et sans RAG
   - Afficher les sources utilisées

## Extensions possibles (optionnelles)

### Extensions de niveau 1 (accessibles)

1. **Recherche hybride (BM25 + dense)**
   - Combiner recherche lexicale (BM25) et recherche vectorielle
   - Fusionner les scores pour améliorer la précision

2. **Métadonnées et filtres**
   - Ajouter des métadonnées aux documents (date, catégorie, auteur)
   - Permettre le filtrage lors de la recherche

### Extensions de niveau 2 (intermédiaires)

4. **Reranking**
   - Ajouter une étape de réordonnancement des résultats
   - Utiliser un modèle cross-encoder pour affiner le classement

5. **Multi-query retrieval**
   - Générer plusieurs reformulations de la question
   - Combiner les résultats de plusieurs recherches

6. **Chunking stratégique**
   - Tester différentes stratégies de découpage (taille fixe, par paragraphe, sémantique)
   - Comparer l'impact sur la qualité

### Extensions de niveau 3 (avancées)

7. **Parent-child retrieval**
   - Indexer de petits chunks mais récupérer des contextes plus larges
   - Améliorer la cohérence tout en gardant la précision

8. **RAG itératif / agent**
   - Permettre au système de faire plusieurs tours de recherche
   - Raffiner la réponse progressivement

9. **Évaluation systématique**
   - Créer un jeu de test avec questions et réponses de référence
   - Calculer des métriques automatiques (BLEU, ROUGE, etc.)
   - Analyser les cas d'échec

## Pipeline NLP typique pour ce sujet

### Étape 1 : Choix de la tâche et du dataset

**Tronc commun** :
- Choisir une tâche de Q&A ou de recherche d'information
- Sélectionner un corpus de documents accessibles et cohérents

**Extensions** :
- Tâches multi-domaines avec filtrage par catégorie
- Corpus multilingue
- Documents structurés (PDF, HTML)

**Suggestions de datasets** :
- Documentation technique (Python docs, frameworks)
- Articles Wikipedia sur un domaine spécifique
- FAQ d'entreprise ou de produit
- Corpus académique simplifié
- Transcriptions de vidéos ou podcasts

### Étape 2 : Prétraitement

**Tronc commun** :
- Nettoyage basique (suppression HTML, normalisation)
- Découpage en chunks de taille fixe (ex: 200-500 mots)

**Extensions** :
- Découpage sémantique (respectant les phrases/paragraphes)
- Extraction de métadonnées
- Gestion de documents structurés

### Étape 3 : Baseline simple

**Tronc commun** :
- Envoyer la question directement au modèle de langage
- Comparer avec la version RAG

### Étape 4 : Intégration de la technique RAG

**Tronc commun** :
- Embeddings + recherche vectorielle + génération

**Extensions** :
- Recherche hybride
- Reranking
- Multi-query

### Étape 5 : Évaluation

**Tronc commun** :
- Évaluation qualitative sur 5-10 exemples
- Comparaison baseline vs RAG
- Identification de 2-3 cas d'échec

**Extensions** :
- Métriques automatiques (Precision@k, Recall@k)
- Évaluation humaine structurée
- Analyse fine des erreurs (retrieval vs generation)

### Étape 6 : Démo et présentation

**Tronc commun** :
- Interface simple (console ou notebook)
- 3-5 questions de démonstration

**Extensions** :
- Visualisation des sources et scores
- Comparaison de plusieurs configurations

## Critères d'évaluation spécifiques à ce sujet

En plus des critères généraux (voir PEDAGOGIE_GLOBALE.md), ce sujet sera évalué sur :

1. **Compréhension du RAG** :
   - Différence entre retrieval et generation
   - Avantages et limites du RAG
   - Trade-offs (précision vs rappel, vitesse vs qualité)

2. **Qualité de la pipeline** :
   - Cohérence du chunking
   - Pertinence des chunks récupérés
   - Qualité de la génération finale

3. **Comparaison baseline vs RAG** :
   - Analyse claire de l'apport du RAG
   - Exemples concrets de différences
   - Cas où le RAG n'apporte rien ou dégrade

4. **Extensions** :
   - Choix d'extensions pertinentes
   - Implémentation correcte
   - Analyse de l'impact de chaque extension

## Conseils pratiques

### Démarrage rapide

1. Commencez avec un **petit corpus** (20-50 documents courts)
2. Utilisez **sentence-transformers** (ex: `all-MiniLM-L6-v2`) pour les embeddings
3. Utilisez **FAISS** pour la recherche vectorielle (simple et efficace)
4. Testez d'abord avec **quelques questions** avant de généraliser

### Pièges à éviter

- **Chunks trop grands** : Le modèle reçoit trop de bruit
- **Chunks trop petits** : Perte de contexte, réponses incomplètes
- **Top-k trop élevé** : Contexte trop long, latence accrue
- **Top-k trop faible** : Information manquante
- **Pas de baseline** : Impossible de prouver l'utilité du RAG

### Optimisations recommandées

1. **Chunking** : Commencez avec 300-500 tokens par chunk
2. **Top-k** : Testez avec k=3 à 5
3. **Modèle d'embeddings** : `all-MiniLM-L6-v2` (rapide) ou `all-mpnet-base-v2` (meilleur)
4. **Modèle de génération** : Un modèle chat compact suffit pour débuter

### Ressources utiles

- **Bibliothèques Python** :
  - `sentence-transformers` : embeddings de texte
  - `faiss-cpu` : recherche vectorielle rapide
  - `rank-bm25` : recherche lexicale (pour hybrid search)
  - `langchain` ou `llama-index` : frameworks RAG complets (optionnel)

- **Modèles d'embeddings** :
  - Sentence-transformers sur HuggingFace
  - OpenAI embeddings (via API, payant)
  - Modèles multilingues si besoin

- **Tutoriels** :
  - Documentation sentence-transformers
  - Tutoriels FAISS
  - Exemples LangChain RAG
