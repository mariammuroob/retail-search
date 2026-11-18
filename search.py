"""Keyword, semantic, and ranks-only hybrid retrieval for the prototype."""

from __future__ import annotations

import json
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).parent
DB_PATH = ROOT / "catalog.db"
INDEX_PATH = ROOT / "products.faiss"
IDS_PATH = ROOT / "products.ids.npy"
MODEL_NAME = "all-MiniLM-L6-v2"
PRICE_PATTERN = re.compile(
    r"(?:under|below|less\s+than)\s+\$?\s*(\d+(?:\.\d+)?)|\$\s*(\d+(?:\.\d+)?)\s+or\s+less",
    re.IGNORECASE,
)

_MODEL: Any = None
_INDEX: Any = None
_INDEX_IDS: np.ndarray | None = None


def parse_query(text: str) -> tuple[str, float | None]:
    """Extract a simple price ceiling before sending the rest to FTS5."""
    match = PRICE_PATTERN.search(text)
    if not match:
        return text.strip(), None
    price = float(next(value for value in match.groups() if value is not None))
    clean_query = re.sub(r"\s+", " ", (text[: match.start()] + text[match.end() :]).strip())
    return clean_query, price


def _fts_query(query: str) -> str:
    """Quote plain tokens and OR them so natural-language queries get partial matches."""
    tokens = re.findall(r"[\w]+", query, flags=re.UNICODE)
    return " OR ".join(f'"{token.replace(chr(34), "")}"' for token in tokens)


def keyword_search(conn: sqlite3.Connection, query: str, limit: int = 50) -> list[tuple[int, int]]:
    """Return (product_id, 1-based BM25 rank), best match first."""
    safe_query = _fts_query(query)
    if not safe_query or limit <= 0:
        return []
    rows = conn.execute(
        """
        SELECT rowid
        FROM product_fts
        WHERE product_fts MATCH ?
        ORDER BY bm25(product_fts)
        LIMIT ?
        """,
        (safe_query, limit),
    ).fetchall()
    return [(int(row[0]), rank) for rank, row in enumerate(rows, start=1)]


def _load_semantic_resources() -> tuple[Any, Any, np.ndarray]:
    """Load the model and index once; Streamlit can also wrap this in cache_resource."""
    global _MODEL, _INDEX, _INDEX_IDS
    if _MODEL is None or _INDEX is None or _INDEX_IDS is None:
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Semantic search needs sentence-transformers and faiss-cpu; run pip install -r requirements.txt"
            ) from exc
        if not INDEX_PATH.exists() or not IDS_PATH.exists():
            raise RuntimeError("Semantic index is missing; run python embed.py first")
        _MODEL = SentenceTransformer(MODEL_NAME)
        _INDEX = faiss.read_index(str(INDEX_PATH))
        _INDEX_IDS = np.load(IDS_PATH)
    return _MODEL, _INDEX, _INDEX_IDS


def _semantic_search_with_scores(query: str, limit: int = 50) -> list[tuple[int, int, float]]:
    if not query.strip() or limit <= 0:
        return []
    model, index, ids = _load_semantic_resources()
    # Embeddings compare meaning, so a warm fleece throw can match "cosy bedroom"
    # even when those exact query words never appear in its title or description.
    vector = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, positions = index.search(vector, min(limit, index.ntotal))
    return [
        (int(ids[position]), rank, float(score))
        for rank, (position, score) in enumerate(zip(positions[0], scores[0]), start=1)
        if position >= 0
    ]


def semantic_search(query: str, limit: int = 50) -> list[tuple[int, int]]:
    """Return (product_id, 1-based cosine-similarity rank)."""
    return [(product_id, rank) for product_id, rank, _score in _semantic_search_with_scores(query, limit)]


def rrf(
    kw_ranked: list[tuple[int, int]],
    sem_ranked: list[tuple[int, int]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Fuse ranked lists without weights because ranks already normalize score scales."""
    scores: dict[int, float] = {}
    for ranked in (kw_ranked, sem_ranked):
        for product_id, rank in ranked:
            scores[product_id] = scores.get(product_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def apply_filters(
    conn: sqlite3.Connection,
    product_ids: list[int] | list[tuple[int, int]] | list[tuple[int, float]],
    max_price: float | None = None,
    category: str | None = None,
    in_stock_only: bool = False,
) -> list[dict[str, Any]]:
    """Fetch products after retrieval, retaining the incoming ranking order."""
    ids = [item[0] if isinstance(item, tuple) else item for item in product_ids]
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    conditions = [f"p.id IN ({placeholders})"]
    params: list[Any] = list(ids)
    if max_price is not None:
        conditions.append("p.price <= ?")
        params.append(max_price)
    if category is not None:
        conditions.append("p.category = ?")
        params.append(category)
    if in_stock_only:
        conditions.append("i.stock_quantity > 0")
    rows = conn.execute(
        f"""
        SELECT p.id, p.title, p.description, p.brand, p.category, p.price, p.rating,
               i.stock_quantity
        FROM products AS p
        JOIN inventory AS i ON i.product_id = p.id
        WHERE {' AND '.join(conditions)}
        """,
        params,
    ).fetchall()
    keys = ("id", "title", "description", "brand", "category", "price", "rating", "stock_quantity")
    by_id = {int(row[0]): dict(zip(keys, row)) for row in rows}
    return [by_id[product_id] for product_id in ids if product_id in by_id]


def _why_keyword(product: dict[str, Any], query: str) -> str:
    text = f"{product['title']} {product['description']}".lower()
    words = [word for word in re.findall(r"[\w]+", query.lower()) if word in text]
    return "keyword words: " + (", ".join(words) if words else "BM25 text match")


def _suggestion(category: str | None, max_price: float | None) -> str:
    if max_price is not None:
        return f"Try relaxing the price to ${max_price * 1.25:.2f} or more."
    if category:
        return "Try dropping the category filter."
    return "Try broader wording or turn off the in-stock-only filter."


def _result_dict(product: dict[str, Any], rank: int, why: str) -> dict[str, Any]:
    return {
        "id": product["id"],
        "title": product["title"],
        "description": product["description"],
        "brand": product["brand"],
        "category": product["category"],
        "price": product["price"],
        "rating": product["rating"],
        "stock": product["stock_quantity"],
        "rank": rank,
        "why_matched": why,
    }


def search_all(
    conn: sqlite3.Connection,
    raw_query: str,
    category: str | None = None,
    in_stock_only: bool = True,
    max_price_override: float | None = None,
    top_n: int = 5,
) -> dict[str, Any]:
    """Run keyword, semantic, and RRF retrieval using the same post-retrieval filters."""
    started = time.perf_counter()
    clean_query, parsed_price = parse_query(raw_query)
    max_price = max_price_override if max_price_override is not None else parsed_price

    keyword_started = time.perf_counter()
    kw_ranked = keyword_search(conn, clean_query, limit=50)
    keyword_ms = (time.perf_counter() - keyword_started) * 1000

    semantic_started = time.perf_counter()
    semantic_scored = _semantic_search_with_scores(clean_query, limit=50)
    sem_ranked = [(product_id, rank) for product_id, rank, _score in semantic_scored]
    semantic_ms = (time.perf_counter() - semantic_started) * 1000
    hybrid_started = time.perf_counter()
    hybrid_ranked = rrf(kw_ranked, sem_ranked)
    hybrid_ms = (time.perf_counter() - hybrid_started) * 1000

    kw_products = apply_filters(conn, kw_ranked, max_price, category, in_stock_only)
    sem_products = apply_filters(conn, sem_ranked, max_price, category, in_stock_only)
    hybrid_products = apply_filters(conn, hybrid_ranked, max_price, category, in_stock_only)
    kw_ranks = dict(kw_ranked)
    sem_ranks = dict(sem_ranked)
    sem_scores = {product_id: score for product_id, _rank, score in semantic_scored}
    hybrid_scores = dict(hybrid_ranked)

    keyword_results = [_result_dict(product, rank, _why_keyword(product, clean_query)) for rank, product in enumerate(kw_products[:top_n], 1)]
    semantic_results = [
        _result_dict(product, rank, f"semantic cosine similarity: {sem_scores.get(product['id'], 0.0):.4f}")
        for rank, product in enumerate(sem_products[:top_n], 1)
    ]
    hybrid_results = [
        _result_dict(
            product,
            rank,
            f"hybrid kw_rank={kw_ranks.get(product['id'], '-')}, sem_rank={sem_ranks.get(product['id'], '-')}, rrf={hybrid_scores.get(product['id'], 0.0):.5f}",
        )
        for rank, product in enumerate(hybrid_products[:top_n], 1)
    ]
    results = {"keyword": keyword_results, "semantic": semantic_results, "hybrid": hybrid_results}
    fallback_used = any(not values for values in results.values())
    suggestion = _suggestion(category, max_price) if fallback_used else None
    latency_ms = {
        "keyword": round(keyword_ms, 3),
        "semantic": round(semantic_ms, 3),
        "hybrid": round(hybrid_ms, 3),
        "total": round((time.perf_counter() - started) * 1000, 3),
    }
    parsed = {"clean_query": clean_query, "max_price": max_price}
    log_filters = {"category": category, "in_stock_only": in_stock_only, "max_price": max_price}
    top5_ids = {method: [item["id"] for item in values] for method, values in results.items()}
    conn.execute(
        "INSERT INTO search_log (query, filters, top5_ids, latency_ms, fallback_used, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (raw_query, json.dumps(log_filters), json.dumps(top5_ids), latency_ms["total"], int(fallback_used), datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    return {**results, "parsed": parsed, "latency_ms": latency_ms, "fallback_used": fallback_used, "suggestion": suggestion}


if __name__ == "__main__":
    with sqlite3.connect(DB_PATH) as connection:
        try:
            output = search_all(connection, "something cosy for my bedroom under $30")
        except RuntimeError as exc:
            raise SystemExit(f"{exc}\nRun `python embed.py` after installing requirements.txt.") from exc
        print(f"Query: {output['parsed']['clean_query']!r} | max_price={output['parsed']['max_price']}")
        for method in ("keyword", "semantic", "hybrid"):
            print(f"\n{method.upper()}")
            for item in output[method]:
                print(f"{item['rank']}. {item['title']} - ${item['price']:.2f} - {item['why_matched']}")
