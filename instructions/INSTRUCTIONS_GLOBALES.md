# Instructions Globales — Projets NLP (Séances 5, 6 et 7)

## Contexte du projet

- **Durée** : Séances 5 et 6
- **Présentation** : Séance 7 (30 minutes par groupe)
- **Format** : Travail en groupes (10 groupes de 3-4 personnes)
- **Modalité** : Chaque groupe choisit **un sujet** parmi les sujets proposés

## Objectifs pédagogiques généraux

- **Mettre en place une pipeline NLP complète** :
  - Prétraitement des données
  - Cœur de la méthode (technique avancée)
  - Évaluation des résultats
  - Démonstration fonctionnelle

- **Explorer une technique NLP avancée** :
  - RAG (Retrieval-Augmented Generation)
  - Systèmes multi-agents
  - Multimodalité (texte + images)
  - Prompt engineering avancé
  - Utilisation d'outils externes (tool-calling / MCP)

- **Développer l'autonomie et la recherche** :
  - Chercher et comparer de la documentation
  - Justifier les choix techniques
  - Adapter les solutions au contexte

- **Renforcer l'analyse critique** :
  - Identifier les limites et biais
  - Proposer des pistes d'amélioration
  - Comparer différentes approches

## Organisation

### Travail en groupes

Chaque groupe :
- Choisit **un sujet** parmi les sujets proposés (ou une variante dérivée)
- Peut adapter le dataset selon la tâche choisie
- Peut modifier la tâche NLP tout en restant dans l'esprit du sujet
- Est libre de choisir ses bibliothèques et outils (sous réserve de cohérence)

### Séances 5 et 6 — Réalisation

1. **Choix du sujet et du dataset**
2. **Conception de la pipeline**
3. **Mise en place du socle "tronc commun"**
4. **Intégration d'extensions / variantes**
5. **Expérimentation et évaluation**
6. **Préparation de la démo et de la présentation**

### Séance 7 — Présentation

- **Format** : 30 minutes par groupe
- **Contenu** : Présentation orale + démonstration en direct

## Attendus de fond (contenu)

### Pipeline NLP complète

Chaque groupe doit développer une pipeline comportant :

1. **Prétraitement / chargement des données**
   - Nettoyage, formatage
   - Chargement dans une structure adaptée

2. **Baseline simple**
   - Solution de référence sans la technique avancée
   - Permet la comparaison

3. **Implémentation de la technique avancée**
   - Cœur du sujet choisi
   - Application concrète sur le dataset

4. **Mécanisme d'évaluation**
   - Métriques adaptées à la tâche
   - Exemples illustratifs
   - Comparaison baseline vs technique avancée

### Compréhension de la technique avancée

Le groupe doit démontrer une compréhension solide :
- **Définition claire** de la technique
- **Exemples d'applications** concrètes
- **Outils existants** et leur utilisation
- **Limites et risques** associés

### Justification des choix

Capacité à expliquer et argumenter :
- Choix du dataset
- Choix de la tâche NLP
- Choix des outils et modèles
- Choix des métriques d'évaluation

## Attendus de forme

### Dépôt de code

Un dépôt Git propre (ou dossier structuré) contenant :
- **Code lisible** et commenté
- **README clair** expliquant :
  - Le sujet et la tâche
  - Comment installer les dépendances
  - Comment exécuter le code
  - Comment reproduire les résultats
- **Notebook ou application** pour la démo

### Présentation orale (30 minutes)

Structure recommandée :
1. **Contexte et problématique**
   - Présentation du sujet et de la tâche
   - Dataset choisi et justification

2. **Technique avancée étudiée**
   - Principes de la technique
   - Outils et bibliothèques utilisés
   - Cas d'usage

3. **Pipeline mise en œuvre**
   - Architecture générale
   - Étapes de traitement
   - Choix techniques

4. **Démonstration**
   - Exécution en direct
   - Exemples concrets
   - Résultats obtenus

5. **Résultats, limites et perspectives**
   - Évaluation quantitative et qualitative
   - Analyse critique
   - Améliorations possibles

## Grille d'évaluation

L'évaluation porte sur 5 critères, chacun noté sur 4 points.

### 1. Compréhension de la technique avancée (sur 4)

### 2. Qualité de la pipeline NLP (sur 4)

### 3. Expérimentations et évaluation (sur 4)

### 4. Démonstration et présentation orale (sur 4)

### 5. Adaptation, créativité et esprit critique (sur 4)

## Niveaux globaux (indicatifs)

### Niveau "Tronc commun" (validation minimale) — 10-12/20

- Pipeline complète mais simple
- Au moins une évaluation avec métriques basiques
- Compréhension correcte de la technique
- Présentation structurée
- Démo fonctionnelle

### Niveau "Bon" — 13-15/20

- Tronc commun **+** au moins une extension significative
- Évaluation commentée et analysée
- Présentation claire et bien organisée
- Bon esprit critique
- Choix techniques justifiés

### Niveau "Excellent" — 16-20/20

- Plusieurs extensions ou comparaison de variantes
- Évaluation approfondie avec analyses fines
- Forte capacité d'analyse critique
- Présentation remarquable
- Originalité et créativité dans l'approche
- Code de haute qualité

## Conseils pratiques

### Pour réussir le tronc commun

1. Commencez par une **tâche simple** et un **petit dataset**
2. Implémentez d'abord une **baseline fonctionnelle**
3. Ajoutez la technique avancée **progressivement**
4. Testez **régulièrement** votre code
5. Documentez **au fur et à mesure**
6. Préparez la démo **suffisamment à l'avance**

### Pour viser l'excellence

1. Comparez **plusieurs variantes** de la technique
2. Proposez des **métriques adaptées** et originales
3. Analysez en profondeur les **cas d'échec**
4. Réfléchissez aux **biais potentiels**
5. Proposez des **pistes d'amélioration concrètes**
6. Soignez la **qualité du code** et la **documentation**

### Pièges à éviter

- Choisir un dataset **trop complexe** ou **trop volumineux**
- Vouloir implémenter **trop d'extensions** sans finir le tronc commun
- Négliger la **baseline** (essentielle pour la comparaison)
- Ne pas tester la **démo** avant la présentation
- Sous-estimer le temps nécessaire pour la **préparation de la présentation**
- Oublier de **documenter** les choix techniques

## Ressources et support

- **Documentation des outils** : Consultez les documentations officielles des bibliothèques utilisées
- **Exemples en ligne** : Recherchez des tutoriels et exemples (avec esprit critique)
- **Questions** : N'hésitez pas à poser des questions pendant les séances

Bon courage et bon travail !
