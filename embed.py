"""Build local MiniLM embeddings and a FAISS inner-product index."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parent
DB_PATH = ROOT / "catalog.db"
INDEX_PATH = ROOT / "products.faiss"
IDS_PATH = ROOT / "products.ids.npy"
MODEL_NAME = "all-MiniLM-L6-v2"


def build_embeddings() -> None:
    """Embed every product and rebuild both SQLite and FAISS artifacts."""
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "Phase 2 dependencies are missing. Run: pip install -r requirements.txt"
        ) from exc

    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT id, title, description FROM products ORDER BY id"
        ).fetchall()

    if not rows:
        raise RuntimeError("catalog.db has no products; run python data_gen.py first")

    ids = np.asarray([row[0] for row in rows], dtype=np.int64)
    texts = [f"{row[1]}. {row[2]}" for row in rows]
    model = SentenceTransformer(MODEL_NAME)
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")
    # Explicit normalization keeps the cosine/IP contract true across model versions.
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(INDEX_PATH))
    np.save(IDS_PATH, ids)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DROP TABLE IF EXISTS product_embeddings")
        conn.execute(
            "CREATE TABLE product_embeddings (product_id INTEGER PRIMARY KEY, embedding BLOB NOT NULL)"
        )
        conn.executemany(
            "INSERT INTO product_embeddings (product_id, embedding) VALUES (?, ?)",
            [(int(product_id), vector.tobytes()) for product_id, vector in zip(ids, vectors)],
        )

    print(f"Built {INDEX_PATH.name} with {len(ids)} normalized product embeddings.")


if __name__ == "__main__":
    build_embeddings()
