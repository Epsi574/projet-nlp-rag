# src/scripts/build_index.py
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from src.embeddings import Embedder
from src.index import VectorIndex

def load_chunks_jsonl(path: str | Path):
    chunks = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def main():
    ap = argparse.ArgumentParser(
        description="Construction de l'index FAISS à partir des embeddings pré-calculés",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Répertoire contenant chunks.jsonl et embeddings.npy"
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/index"),
        help="Répertoire de sortie pour l'index FAISS"
    )
    ap.add_argument(
        "--regenerate",
        action="store_true",
        help="Régénérer les embeddings au lieu d'utiliser ceux pré-calculés"
    )
    ap.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Modèle d'embeddings (uniquement si --regenerate)"
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Taille des batches (uniquement si --regenerate)"
    )
    args = ap.parse_args()

    # Chemins des fichiers
    chunks_file = args.data_dir / "chunks.jsonl"
    embeddings_file = args.data_dir / "embeddings.npy"
    chunk_ids_file = args.data_dir / "chunk_ids.npy"

    # Vérifications
    if not chunks_file.exists():
        raise FileNotFoundError(
            f"Fichier chunks.jsonl introuvable: {chunks_file}\n"
            f"Exécutez d'abord: python -m src.scripts.build_data_pipeline"
        )

    print(f"Chargement des chunks depuis {chunks_file}...")
    chunks = load_chunks_jsonl(chunks_file)
    print(f"   {len(chunks)} chunks chargés")

    # Charger ou générer les embeddings
    if args.regenerate:
        print(f"\nRégénération des embeddings avec {args.model}...")        
        texts = [c["text"] for c in chunks]
        emb = Embedder(model_name=args.model, normalize=True)
        vectors = emb.encode(texts, batch_size=args.batch_size)
        
        # Sauvegarder les nouveaux embeddings
        np.save(embeddings_file, vectors)
        chunk_ids = [c["chunk_id"] for c in chunks]
        np.save(chunk_ids_file, np.array(chunk_ids))
        print(f"   Embeddings sauvegardés dans {embeddings_file}")
        
    else:
        if not embeddings_file.exists():
            raise FileNotFoundError(
                f"Fichier embeddings.npy introuvable: {embeddings_file}\n"
                f"Options:\n"
                f"  1. Exécutez: python -m src.scripts.build_data_pipeline\n"
                f"  2. Utilisez --regenerate pour les générer maintenant"
            )
        
        print(f"\nChargement des embeddings pré-calculés depuis {embeddings_file}...")
        vectors = np.load(embeddings_file)
        print(f"   {vectors.shape[0]} vecteurs de dimension {vectors.shape[1]} chargés")
        
        # Vérification de cohérence
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Incohérence: {len(chunks)} chunks mais {len(vectors)} embeddings.\n"
                f"Utilisez --regenerate pour recalculer les embeddings."
            )

    # Construire l'index FAISS
    print(f"\nConstruction de l'index FAISS...")
    vindex = VectorIndex.build(vectors, chunks)
    
    # Sauvegarder l'index
    print(f"Sauvegarde de l'index dans {args.out_dir}...")
    vindex.save(args.out_dir)

    print(f"\nIndex construit avec succès!")
    print(f"   - {len(chunks)} vecteurs indexés")
    print(f"   - Dimension: {vectors.shape[1]}")
    print(f"   - Emplacement: {args.out_dir}")
    print(f"\nFichiers créés:")
    print(f"   - {args.out_dir / 'index.faiss'}")
    print(f"   - {args.out_dir / 'chunks_meta.jsonl'}")

if __name__ == "__main__":
    main()
