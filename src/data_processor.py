# src/data_processor.py
"""
Module pour le traitement des données du pipeline RAG.
Inclut les fonctions de nettoyage, chunking, et génération de métadonnées.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Any
import re
import json
import hashlib
import unicodedata
import pandas as pd
import numpy as np


# ============================================================================
# SCHEMAS ET CONSTANTES
# ============================================================================

DOC_SCHEMA = [
    "doc_id", "title", "date", "author",
    "source", "url", "local_path", "language"
]

CHUNK_SCHEMA = [
    "chunk_id", "doc_id", "text",
    "title", "date", "author", "source", "url",
    "start_char", "end_char"
]


# ============================================================================
# UTILITAIRES
# ============================================================================

def safe_filename(filename: str) -> str:
    """
    Normalise un nom de fichier pour éviter les problèmes de caractères spéciaux.
    Convertit les caractères accentués/spéciaux en ASCII safe.
    
    Args:
        filename: Nom du fichier à normaliser
        
    Returns:
        Nom de fichier normalisé
    """
    # Décompose les caractères Unicode (é -> e + ´)
    nfd = unicodedata.normalize('NFD', filename)
    # Garde uniquement les caractères ASCII
    ascii_str = nfd.encode('ascii', 'ignore').decode('ascii')
    
    # Remplace les caractères problématiques
    replacements = {
        ' ': '_',
        '−': '-',
        '–': '-',
        '—': '-',
        ''': "'",
        ''': "'",
        '"': '"',
        '"': '"',
        '(': '_',
        ')': '_',
        ',': '_',
    }
    
    for old, new in replacements.items():
        ascii_str = ascii_str.replace(old, new)
    
    # Supprime les caractères multiples
    ascii_str = re.sub(r'_+', '_', ascii_str)
    ascii_str = re.sub(r'-+', '-', ascii_str)
    
    return ascii_str.strip('_-')


def file_sha1(path: Path) -> str:
    """
    Calcule le hash SHA1 d'un fichier.
    
    Args:
        path: Chemin du fichier
        
    Returns:
        Hash SHA1 en hexadécimal
    """
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text(path: Path) -> str:
    """
    Lit un fichier texte avec gestion robuste de l'encodage.
    
    Args:
        path: Chemin du fichier
        
    Returns:
        Contenu du fichier
    """
    return path.read_text(encoding="utf-8", errors="ignore")


# ============================================================================
# EXTRACTION DE MÉTADONNÉES
# ============================================================================

def extract_auto_metadata(text: str, filename: str) -> Dict[str, str]:
    """
    Extrait automatiquement la source et la langue d'un document.
    
    Args:
        text: Contenu du document
        filename: Nom du fichier (sans extension)
        
    Returns:
        Dictionnaire avec 'source' et 'language'
    """
    header = text[:1000]
    
    # Déterminer la source
    source = "Archive historique"
    if "wikisource" in header.lower():
        source = "Wikisource"
    elif "gallica" in header.lower():
        source = "Gallica BnF"
    elif "journal officiel" in header.lower():
        source = "Journal Officiel"
    elif "convention" in filename.lower() or "déclaration" in filename.lower():
        source = "Document officiel"
    
    # Déterminer la langue
    language = "fr"
    english_words = re.findall(r'\b(the|and|of|in|to|for|with|that|this|from)\b', header[:500], re.IGNORECASE)
    french_words = re.findall(r'\b(le|la|de|et|les|des|un|une|dans|pour|par|sur)\b', header[:500], re.IGNORECASE)
    
    if len(english_words) > len(french_words) * 2:
        language = "en"
    
    return {
        "source": source,
        "language": language
    }


# ============================================================================
# NETTOYAGE DE TEXTE
# ============================================================================

def clean_text(text: str) -> str:
    """
    Nettoie et normalise un texte.
    - Supprime les balises HTML/meta
    - Coupe le contenu après les sections éditoriales
    - Normalise les espaces et sauts de ligne
    - Retire les césures de fin de ligne
    
    Args:
        text: Texte brut à nettoyer
        
    Returns:
        Texte nettoyé
    """
    # Supprimer les lignes HTML/meta
    lines = text.split('\n')
    filtered_lines = []
    for line in lines:
        stripped = line.strip()
        # Ignorer les balises HTML/meta
        if stripped.startswith('<meta') or stripped.startswith('<link') or stripped.startswith('<!DOCTYPE'):
            continue
        filtered_lines.append(line)
    
    text = '\n'.join(filtered_lines)
    
    # Couper tout après "À propos de cette édition électronique" et variantes
    cutoff_phrases = [
        "À propos de cette édition électronique",
        "à propos de cette édition électronique",
        "À PROPOS DE CETTE ÉDITION",
        "Voir aussi",
        "Notes et références",
        "Source",
        "Cette édition électronique",
    ]
    
    for phrase in cutoff_phrases:
        idx = text.find(phrase)
        if idx != -1:
            text = text[:idx]
            break  # Couper à la première occurrence trouvée
    
    # Normalisation basique
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)           # espaces multiples
    text = re.sub(r"\n{3,}", "\n\n", text)        # trop de sauts de ligne
    text = text.strip()

    # Retirer césures de fin de ligne "exem-\nple" -> "exemple"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    return text


# ============================================================================
# CHUNKING
# ============================================================================

def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 300) -> List[Tuple[int, int, str]]:
    """
    Découpe un texte en chunks avec chevauchement.
    
    Stratégie de découpe (par ordre de priorité):
    1. Par paragraphe (double saut de ligne)
    2. Par fin de phrase (. ! ?)
    3. Par saut de ligne simple
    4. Découpe brute si nécessaire
    
    Args:
        text: Texte à découper
        chunk_size: Taille cible d'un chunk en caractères (~2000 chars = 300-500 tokens)
        overlap: Chevauchement entre chunks en caractères (~300 chars = 50-75 tokens)
        
    Returns:
        Liste de tuples (start_pos, end_pos, chunk_text)
    """
    chunks = []
    n = len(text)
    start = 0
    
    while start < n:
        end = min(n, start + chunk_size)
        
        # Stratégie de découpe (par ordre de priorité)
        window = text[start:end]
        
        # Priorité max: paragraphe (double saut de ligne)
        para_cut = window.rfind("\n\n")
        if para_cut > int(0.5 * len(window)):
            end = start + para_cut + 2  # Garder les \n\n
        
        # Sinon: fin de phrase (. ! ?)
        elif any(marker in window for marker in [". ", "! ", "? "]):
            sentence_cuts = [
                window.rfind(". "),
                window.rfind("! "),
                window.rfind("? ")
            ]
            best_cut = max(sentence_cuts)
            if best_cut > int(0.4 * len(window)):
                end = start + best_cut + 2  # Inclure ponctuation + espace
        
        # Fallback: simple saut de ligne
        elif "\n" in window:
            line_cut = window.rfind("\n")
            if line_cut > int(0.3 * len(window)):
                end = start + line_cut + 1
        
        # Extraire le chunk
        chunk_raw = text[start:end]
        chunk = chunk_raw.strip()
        
        if chunk:
            # Trouver où commence le texte non-vide
            start_offset = len(chunk_raw) - len(chunk_raw.lstrip())
            end_offset = len(chunk_raw) - len(chunk_raw.rstrip())
            
            actual_start = start + start_offset
            actual_end = end - end_offset
            
            chunks.append((actual_start, actual_end, chunk))
        
        # Avancer avec overlap (sauf si on est à la fin)
        if end >= n:
            break
        
        # S'assurer qu'on avance d'au moins chunk_size - overlap pour éviter les duplicatas
        next_start = end - overlap
        if next_start <= start:
            next_start = start + max(1, chunk_size - overlap)
        start = next_start
    
    return chunks


# ============================================================================
# PIPELINE PRINCIPALE
# ============================================================================

class DataProcessor:
    """
    Classe principale pour le traitement des données RAG.
    """
    
    def __init__(self, data_dir: Path):
        """
        Initialise le processeur de données.
        
        Args:
            data_dir: Répertoire racine des données
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.clean_dir = self.data_dir / "clean"
        
        # Créer les répertoires si nécessaire
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.clean_dir.mkdir(parents=True, exist_ok=True)
        
        # Fichiers de sortie
        self.docs_csv = self.data_dir / "docs.csv"
        self.chunks_jsonl = self.data_dir / "chunks.jsonl"
        self.embeddings_npy = self.data_dir / "embeddings.npy"
        self.chunk_ids_npy = self.data_dir / "chunk_ids.npy"
    
    def scan_documents(self, verbose: bool = True) -> pd.DataFrame:
        """
        Scanne le répertoire raw/ et crée le catalogue de documents.
        
        Args:
            verbose: Afficher les informations de progression
            
        Returns:
            DataFrame avec les métadonnées des documents
        """
        docs = []
        doc_counter = 1
        
        for txt_file in sorted(self.raw_dir.glob("*.txt")):
            # Extraire le titre du nom de fichier (remplacer _ par espace)
            title = txt_file.stem.replace("_", " ")
            
            # Générer un doc_id unique
            doc_id = f"doc{doc_counter:03d}"
            
            # Lire le contenu pour extraire source et langue
            try:
                text_content = read_text(txt_file)
                metadata = extract_auto_metadata(text_content, txt_file.stem)
            except Exception as e:
                if verbose:
                    print(f"Erreur lecture {txt_file.name}: {e}")
                metadata = {"source": "Archive historique", "language": "fr"}
            
            # Créer l'entrée du document
            doc_entry = {
                "doc_id": doc_id,
                "title": title,
                "date": "",  # À remplir manuellement si nécessaire
                "author": "",  # À remplir manuellement si nécessaire
                "source": metadata["source"],
                "url": "",  # À remplir manuellement si nécessaire
                "local_path": str(txt_file.as_posix()),
                "language": metadata["language"],
            }
            
            docs.append(doc_entry)
            doc_counter += 1
            
            if verbose:
                print(f"{doc_id}: {title} | Source: {metadata['source']} | Langue: {metadata['language']}")
        
        # Créer le DataFrame et sauvegarder
        df_docs = pd.DataFrame(docs, columns=DOC_SCHEMA)
        df_docs.to_csv(self.docs_csv, index=False, encoding='utf-8')
        
        if verbose:
            print(f"\nTotal: {len(docs)} documents")
            print(f"Sauvegardé: {self.docs_csv}")
        
        return df_docs
    
    def clean_documents(self, verbose: bool = True) -> int:
        """
        Nettoie tous les documents du catalogue.
        
        Args:
            verbose: Afficher les informations de progression
            
        Returns:
            Nombre de documents nettoyés
        """
        df_docs = pd.read_csv(self.docs_csv)
        cleaned_count = 0
        
        for _, row in df_docs.iterrows():
            raw_path = Path(row["local_path"])
            if not raw_path.exists():
                if verbose:
                    print(f"MISSING: {raw_path}")
                continue
            
            txt = read_text(raw_path)
            txt_clean = clean_text(txt)
            
            # Normaliser le nom de fichier pour éviter les problèmes
            safe_name = safe_filename(raw_path.stem) + ".txt"
            clean_path = self.clean_dir / safe_name
            clean_path.write_text(txt_clean, encoding="utf-8")
            
            if verbose:
                print(f"{raw_path.name} -> {safe_name}")
            
            cleaned_count += 1
        
        if verbose:
            print(f"\nCleaned files in: {self.clean_dir}")
        
        return cleaned_count
    
    def generate_chunks(self, chunk_size: int = 2000, overlap: int = 200, verbose: bool = True) -> pd.DataFrame:
        """
        Génère les chunks à partir des documents nettoyés.
        
        Args:
            chunk_size: Taille cible des chunks en caractères
            overlap: Chevauchement entre chunks en caractères
            verbose: Afficher les informations de progression
            
        Returns:
            DataFrame avec tous les chunks
        """
        df_docs = pd.read_csv(self.docs_csv)
        
        with self.chunks_jsonl.open("w", encoding="utf-8") as out:
            total_chunks = 0
            skipped = 0
            
            for _, row in df_docs.iterrows():
                raw_path = Path(row["local_path"])
                
                # Utiliser le nom normalisé pour chercher le fichier clean
                safe_name = safe_filename(raw_path.stem) + ".txt"
                clean_path = self.clean_dir / safe_name
                
                if not clean_path.exists():
                    if verbose:
                        print(f"SKIP (no clean): {clean_path}")
                    skipped += 1
                    continue
                
                text = read_text(clean_path)
                spans = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
                
                for i, (s, e, chunk) in enumerate(spans):
                    chunk_id = f"{row['doc_id']}_{i:04d}"
                    rec = {
                        "chunk_id": chunk_id,
                        "doc_id": row["doc_id"],
                        "text": chunk,
                        "title": row.get("title", ""),
                        "date": row.get("date", ""),
                        "author": row.get("author", ""),
                        "source": row.get("source", ""),
                        "url": row.get("url", ""),
                        "start_char": int(s),
                        "end_char": int(e),
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total_chunks += 1
        
        if verbose:
            print(f"\nTotal chunks: {total_chunks}")
            if skipped > 0:
                print(f"Skipped: {skipped} documents")
            print(f"Wrote: {self.chunks_jsonl}")
        
        # Charger et retourner le DataFrame
        chunks = []
        with self.chunks_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
        
        return pd.DataFrame(chunks)
    
    def generate_embeddings(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                          batch_size: int = 64, verbose: bool = True) -> Tuple[np.ndarray, List[str]]:
        """
        Génère les embeddings pour tous les chunks.
        
        Args:
            model_name: Nom du modèle d'embeddings
            batch_size: Taille des batches pour l'encodage
            verbose: Afficher les informations de progression
            
        Returns:
            Tuple (embeddings, chunk_ids)
        """
        from .embeddings import Embedder
        
        # Charger les chunks
        chunks_data = []
        with self.chunks_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                chunks_data.append(json.loads(line))
        
        # Extraire les textes et IDs
        texts = [chunk["text"] for chunk in chunks_data]
        chunk_ids = [chunk["chunk_id"] for chunk in chunks_data]
        
        if verbose:
            print(f"Génération des embeddings pour {len(texts)} chunks...")
        
        # Initialiser l'embedder
        embedder = Embedder(model_name=model_name, normalize=True)
        
        # Générer les embeddings
        embeddings = embedder.encode(texts, batch_size=batch_size)
        
        # Sauvegarder
        np.save(self.embeddings_npy, embeddings)
        np.save(self.chunk_ids_npy, np.array(chunk_ids))
        
        if verbose:
            print(f"Embeddings sauvegardés : {self.embeddings_npy}")
            print(f"IDs sauvegardés : {self.chunk_ids_npy}")
            print(f"  Shape: {embeddings.shape}, dtype: {embeddings.dtype}")
        
        return embeddings, chunk_ids
    
    def run_full_pipeline(self, chunk_size: int = 2000, overlap: int = 200,
                         model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                         verbose: bool = True) -> Dict[str, Any]:
        """
        Exécute la pipeline complète de traitement des données.
        
        Args:
            chunk_size: Taille cible des chunks
            overlap: Chevauchement entre chunks
            model_name: Modèle d'embeddings à utiliser
            verbose: Afficher les informations de progression
            
        Returns:
            Dictionnaire avec les statistiques de la pipeline
        """
        if verbose:
            print("=" * 70)
            print("PIPELINE DE TRAITEMENT DES DONNÉES RAG")
            print("=" * 70)
        
        # Étape 1: Scanner les documents
        if verbose:
            print("\n[1/4] Scan des documents...")
        df_docs = self.scan_documents(verbose=verbose)
        
        # Étape 2: Nettoyer les documents
        if verbose:
            print("\n[2/4] Nettoyage des documents...")
        cleaned_count = self.clean_documents(verbose=verbose)
        
        # Étape 3: Générer les chunks
        if verbose:
            print(f"\n[3/4] Génération des chunks (size={chunk_size}, overlap={overlap})...")
        df_chunks = self.generate_chunks(chunk_size=chunk_size, overlap=overlap, verbose=verbose)
        
        # Étape 4: Générer les embeddings
        if verbose:
            print(f"\n[4/4] Génération des embeddings (model={model_name})...")
        embeddings, chunk_ids = self.generate_embeddings(model_name=model_name, verbose=verbose)
        
        if verbose:
            print("\n" + "=" * 70)
            print("PIPELINE COMPLÉTÉE AVEC SUCCÈS")
            print("=" * 70)
        
        return {
            "num_documents": len(df_docs),
            "num_cleaned": cleaned_count,
            "num_chunks": len(df_chunks),
            "embedding_dim": embeddings.shape[1],
            "files": {
                "docs": str(self.docs_csv),
                "chunks": str(self.chunks_jsonl),
                "embeddings": str(self.embeddings_npy),
                "chunk_ids": str(self.chunk_ids_npy),
            }
        }
