#!/usr/bin/env python3
"""
Script pour exécuter la pipeline complète de traitement des données RAG.

Usage:
    python -m src.scripts.build_data_pipeline [--data-dir DATA_DIR] [--chunk-size SIZE] [--overlap OVERLAP]
    
Exemple:
    python -m src.scripts.build_data_pipeline --chunk-size 2000 --overlap 200
"""
import argparse
from pathlib import Path
import sys
import os
import certifi

# Force httpx / transformers à utiliser le certificat CA de certifi pour venv
os.environ["SSL_CERT_FILE"] = certifi.where()
# Ajouter le répertoire parent au path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data_processor import DataProcessor


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de traitement des données RAG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data",
        help="Répertoire racine des données"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Taille cible des chunks en caractères"
    )
    
    parser.add_argument(
        "--overlap",
        type=int,
        default=200,
        help="Chevauchement entre chunks en caractères"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Modèle d'embeddings à utiliser"
    )
    
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Ne pas générer les embeddings (utile pour tester le chunking)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Mode silencieux (pas d'affichage détaillé)"
    )
    
    args = parser.parse_args()
    
    # Initialiser le processeur
    processor = DataProcessor(args.data_dir)
    
    # Exécuter la pipeline
    if args.no_embeddings:
        # Pipeline partielle sans embeddings
        print("=" * 70)
        print("PIPELINE DE TRAITEMENT DES DONNÉES (sans embeddings)")
        print("=" * 70)
        
        print("\n[1/3] Scan des documents...")
        df_docs = processor.scan_documents(verbose=not args.quiet)
        
        print("\n[2/3] Nettoyage des documents...")
        cleaned_count = processor.clean_documents(verbose=not args.quiet)
        
        print(f"\n[3/3] Génération des chunks (size={args.chunk_size}, overlap={args.overlap})...")
        df_chunks = processor.generate_chunks(
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            verbose=not args.quiet
        )
        
        print("\n" + "=" * 70)
        print("PIPELINE COMPLÉTÉE (sans embeddings)")
        print("=" * 70)
        print(f"Documents: {len(df_docs)}")
        print(f"Nettoyés: {cleaned_count}")
        print(f"Chunks: {len(df_chunks)}")
        
    else:
        # Pipeline complète
        stats = processor.run_full_pipeline(
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            model_name=args.model,
            verbose=not args.quiet
        )
        
        if not args.quiet:
            print(f"\nDocuments: {stats['num_documents']}")
            print(f"Nettoyés: {stats['num_cleaned']}")
            print(f"Chunks: {stats['num_chunks']}")
            print(f"Dimension embeddings: {stats['embedding_dim']}")
            print(f"\nFichiers générés:")
            for key, path in stats['files'].items():
                print(f"  - {key}: {path}")


if __name__ == "__main__":
    main()
