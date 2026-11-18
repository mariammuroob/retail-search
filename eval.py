"""Reproducible precision, hit-rate, no-results, and fail-soft evaluation."""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from pathlib import Path
from statistics import mean
from typing import Any

import search


ROOT = Path(__file__).parent
DB_PATH = ROOT / "catalog.db"
RESULTS_PATH = ROOT / "eval_results.csv"
STOPWORDS = {"a", "an", "and", "for", "from", "in", "my", "of", "on", "the", "to", "with"}
METHODS = ("keyword", "keyword_stemming", "semantic", "hybrid")
REPORT_METHODS = ("keyword", "semantic", "hybrid")
CATALOG_CATEGORIES = {
    "home": ("Home Decor", "Bedding"),
    "kitchen": ("Kitchen",),
    "toys": ("Toys",),
    "apparel": ("Apparel",),
    "beauty": ("Beauty",),
    "electronics": ("Electronics",),
}


def load_queries() -> list[dict[str, Any]]:
    with (ROOT / "test_queries.json").open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or len(data) != 30:
        raise ValueError("test_queries.json must contain 30 entries")
    return data


def category_values(entry: dict[str, Any], home_fix: bool) -> tuple[str, ...]:
    category = str(entry.get("category", "")).lower()
    if category == "home" and not home_fix:
        return ("Home Decor",)
    return CATALOG_CATEGORIES.get(category, (entry.get("category", ""),))


def porter_query(query: str) -> str:
    tokens = [token for token in re.findall(r"[\w]+", query, re.UNICODE) if token.lower() not in STOPWORDS]
    return " OR ".join(f'"{token.replace(chr(34), "")}"' for token in tokens)


def build_porter_fts(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS eval_product_fts_porter")
    conn.execute(
        """CREATE VIRTUAL TABLE eval_product_fts_porter USING fts5(
            title, description, content='products', content_rowid='id', tokenize='porter'
        )"""
    )
    conn.execute("INSERT INTO eval_product_fts_porter(eval_product_fts_porter) VALUES ('rebuild')")


def keyword_stemming_search(conn: sqlite3.Connection, query: str, limit: int = 50) -> list[tuple[int, int]]:
    safe_query = porter_query(query)
    if not safe_query or limit <= 0:
        return []
    rows = conn.execute(
        """
        SELECT rowid FROM eval_product_fts_porter
        WHERE eval_product_fts_porter MATCH ?
        ORDER BY bm25(eval_product_fts_porter)
        LIMIT ?
        """,
        (safe_query, limit),
    ).fetchall()
    return [(int(row[0]), rank) for rank, row in enumerate(rows, 1)]


def apply_eval_filters(
    conn: sqlite3.Connection,
    ranked: list[tuple[int, int]] | list[tuple[int, float]],
    categories: tuple[str, ...],
    max_price: float | None,
) -> list[dict[str, Any]]:
    ids = [item[0] for item in ranked]
    if not ids:
        return []
    id_placeholders = ",".join("?" for _ in ids)
    category_placeholders = ",".join("?" for _ in categories)
    conditions = [f"p.id IN ({id_placeholders})", f"p.category IN ({category_placeholders})", "i.stock_quantity > 0"]
    params: list[Any] = ids + list(categories)
    if max_price is not None:
        conditions.append("p.price <= ?")
        params.append(max_price)
    rows = conn.execute(
        f"""
        SELECT p.id, p.title, p.description, p.brand, p.category, p.price, p.rating, i.stock_quantity
        FROM products p JOIN inventory i ON i.product_id = p.id
        WHERE {' AND '.join(conditions)}
        """,
        params,
    ).fetchall()
    keys = ("id", "title", "description", "brand", "category", "price", "rating", "stock_quantity")
    by_id = {int(row[0]): dict(zip(keys, row)) for row in rows}
    return [by_id[product_id] for product_id in ids if product_id in by_id]


def retrieve(conn: sqlite3.Connection, entry: dict[str, Any], home_fix: bool, porter_ready: bool) -> dict[str, list[dict[str, Any]]]:
    clean_query, max_price = search.parse_query(entry["query"])
    categories = category_values(entry, home_fix)
    keyword = search.keyword_search(conn, clean_query, 50)
    semantic_scored = search._semantic_search_with_scores(clean_query, 50)
    semantic = [(product_id, rank) for product_id, rank, _score in semantic_scored]
    hybrid = search.rrf(keyword, semantic)
    results = {
        "keyword": apply_eval_filters(conn, keyword, categories, max_price),
        "semantic": apply_eval_filters(conn, semantic, categories, max_price),
        "hybrid": apply_eval_filters(conn, hybrid, categories, max_price),
    }
    if porter_ready:
        stemmed = keyword_stemming_search(conn, clean_query, 50)
        results["keyword_stemming"] = apply_eval_filters(conn, stemmed, categories, max_price)
    return results


def scored(entry: dict[str, Any]) -> bool:
    number = int(str(entry["query_id"]).lstrip("q"))
    return 1 <= number <= 26 and bool(entry.get("relevant_product_ids"))


def precision_at_5(items: list[dict[str, Any]], relevant: set[int]) -> float:
    return sum(item["id"] in relevant for item in items[:5]) / 5


def hit_at_5(items: list[dict[str, Any]], relevant: set[int]) -> int:
    return int(any(item["id"] in relevant for item in items[:5]))


def evaluate_set(queries: list[dict[str, Any]], conn: sqlite3.Connection, home_fix: bool, porter_ready: bool) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]], dict[str, tuple[int, int]]]:
    active_methods = METHODS if porter_ready else REPORT_METHODS
    rows: list[dict[str, Any]] = []
    precision_values = {method: [] for method in active_methods}
    hit_values = {method: [] for method in active_methods}
    no_results_all = {method: 0 for method in active_methods}
    no_results_existing = {method: 0 for method in active_methods}
    existing_categories = {row[0] for row in conn.execute("SELECT DISTINCT category FROM products").fetchall()}
    for entry in queries:
        number = int(str(entry["query_id"]).lstrip("q"))
        categories = category_values(entry, home_fix)
        category_exists = all(category in existing_categories for category in categories)
        results = retrieve(conn, entry, home_fix, porter_ready)
        is_scored = scored(entry)
        relevant = set(entry.get("relevant_product_ids", []))
        fail_soft = number in (28, 29) and all(not results[method] for method in REPORT_METHODS)
        row: dict[str, Any] = {
            "query_id": entry["query_id"], "query": entry["query"], "scored": is_scored,
            "category": " + ".join(categories), "category_exists": category_exists,
            "fail_soft_check": fail_soft if number in (28, 29) else "",
        }
        for method in active_methods:
            items = results[method]
            no_results_all[method] += not items
            no_results_existing[method] += not items if category_exists else 0
            row[f"{method}_count"] = len(items)
            row[f"{method}_precision_at_5"] = round(precision_at_5(items, relevant), 4) if is_scored else ""
            row[f"{method}_hit_at_5"] = hit_at_5(items, relevant) if is_scored else ""
            if is_scored:
                precision_values[method].append(precision_at_5(items, relevant))
                hit_values[method].append(hit_at_5(items, relevant))
        rows.append(row)
    metrics = {
        method: {"precision_at_5": mean(precision_values[method]), "hit_at_5": mean(hit_values[method])}
        for method in active_methods
    }
    no_results = {method: (no_results_all[method], no_results_existing[method]) for method in active_methods}
    return rows, metrics, no_results


def print_metrics(label: str, metrics: dict[str, dict[str, float]], no_results: dict[str, tuple[int, int]], n: int, existing_n: int, methods: tuple[str, ...]) -> None:
    print(f"{label} (n={n}; category-existing n={existing_n})")
    for method in methods:
        all_count, existing_count = no_results[method]
        print(f"  {method}: precision@5={metrics[method]['precision_at_5']:.4f}, hit@5={metrics[method]['hit_at_5']:.4f}, no-results all={all_count}/30 ({all_count / 30:.4f}), category-existing={existing_count}/{existing_n} ({existing_count / existing_n:.4f})")


def main() -> None:
    queries = load_queries()
    n = sum(scored(entry) for entry in queries)
    excluded = [entry["query_id"] for entry in queries if 1 <= int(str(entry["query_id"]).lstrip("q")) <= 26 and not entry.get("relevant_product_ids")]
    with sqlite3.connect(DB_PATH) as conn:
        catalogue_categories = {row[0] for row in conn.execute("SELECT DISTINCT category FROM products").fetchall()}
        existing_n = sum(all(category in catalogue_categories for category in category_values(entry, True)) for entry in queries)
        build_porter_fts(conn)
        try:
            before_rows, before_metrics, before_no_results = evaluate_set(queries, conn, False, False)
            after_rows, after_metrics, after_no_results = evaluate_set(queries, conn, True, True)
            fail_details = {}
            for entry in queries:
                if entry["query_id"] in ("q28", "q29"):
                    fail_details[entry["query_id"]] = retrieve(conn, entry, True, True)
        finally:
            conn.execute("DROP TABLE IF EXISTS eval_product_fts_porter")
            conn.commit()
    print(f"Excluded from precision/hit metrics: {', '.join(excluded)}")
    print_metrics("BEFORE home-category fix (home = Home Decor)", before_metrics, before_no_results, n, existing_n, REPORT_METHODS)
    print_metrics("AFTER home-category fix (home = Home Decor + Bedding)", after_metrics, after_no_results, n, existing_n, METHODS)
    print("Fail-soft checks (strict: all keyword/semantic/hybrid lists empty):")
    for query_id in ("q28", "q29"):
        row = next(item for item in after_rows if item["query_id"] == query_id)
        print(f"  {query_id}: {'PASS' if row['fail_soft_check'] else 'FAIL'}")
    print("q28/q29 returned products after fix:")
    for query_id, results in fail_details.items():
        for method in METHODS:
            products = [f"{item['id']} {item['title']} ${item['price']:.2f}" for item in results[method][:5]]
            print(f"  {query_id} {method}: {products}")
    print("AFTER per-query precision@5 / hit@5:")
    for row in after_rows:
        if row["scored"]:
            print(f"  {row['query_id']}: kw={row['keyword_precision_at_5']}/{row['keyword_hit_at_5']}, kw+stem={row['keyword_stemming_precision_at_5']}/{row['keyword_stemming_hit_at_5']}, sem={row['semantic_precision_at_5']}/{row['semantic_hit_at_5']}, hybrid={row['hybrid_precision_at_5']}/{row['hybrid_hit_at_5']}")
    fieldnames = list(after_rows[0].keys())
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(after_rows)
    print(f"Saved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
