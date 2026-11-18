"""Reproducible precision and fail-soft evaluation for the 30-query set."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from statistics import mean
from typing import Any

import search


ROOT = Path(__file__).parent
DB_PATH = ROOT / "catalog.db"
RESULTS_PATH = ROOT / "eval_results.csv"
CATALOG_CATEGORIES = {
    "home": "Home Decor",
    "bedding": "Bedding",
    "kitchen": "Kitchen",
    "toys": "Toys",
    "apparel": "Apparel",
    "beauty": "Beauty",
    "electronics": "Electronics",
}


def find_test_file() -> Path:
    option = ROOT / "test_queries.json"
    if not option.exists():
        raise FileNotFoundError("./test_queries.json was not found; it was not recreated.")
    return option.resolve()


def load_queries(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or len(data) != 30:
        raise ValueError("test_queries.json must contain the existing 30 entries")
    return data


def precision_at_5(items: list[dict[str, Any]], relevant: set[int]) -> float:
    return sum(item["id"] in relevant for item in items[:5]) / 5


def evaluate(queries: list[dict[str, Any]], conn: sqlite3.Connection) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, int]]:
    methods = ("keyword", "semantic", "hybrid")
    rows: list[dict[str, Any]] = []
    precision_values = {method: [] for method in methods}
    no_results = {method: 0 for method in methods}
    for entry in queries:
        query_number = int(str(entry["query_id"]).lstrip("q"))
        category = CATALOG_CATEGORIES.get(str(entry.get("category", "")).lower(), entry.get("category"))
        output = search.search_all(conn, entry["query"], category=category, in_stock_only=True, top_n=5)
        scored = 1 <= query_number <= 26
        relevant = set(entry.get("relevant_product_ids", []))
        if scored and not relevant:
            raise RuntimeError(f"{entry['query_id']} has no hand-confirmed relevant_product_ids; run python label_helper.py --interactive first")
        fail_soft = query_number in (28, 29) and all(not output[method] for method in methods) and bool(output.get("suggestion"))
        row: dict[str, Any] = {
            "query_id": entry["query_id"],
            "query": entry["query"],
            "kind": entry.get("kind", ""),
            "category": category or "",
            "scored": scored,
            "fail_soft_check": fail_soft if query_number in (28, 29) else "",
            "suggestion": output.get("suggestion") or "",
        }
        for method in methods:
            items = output[method]
            no_results[method] += not items
            row[f"{method}_count"] = len(items)
            row[f"{method}_precision_at_5"] = round(precision_at_5(items, relevant), 4) if scored else ""
            row[f"{method}_latency_ms"] = output["latency_ms"].get(method, "")
            if scored:
                precision_values[method].append(precision_at_5(items, relevant))
        rows.append(row)
    means = {method: mean(values) for method, values in precision_values.items()}
    return rows, means, no_results


def main() -> None:
    test_path = find_test_file()
    queries = load_queries(test_path)
    with sqlite3.connect(DB_PATH) as conn:
        rows, means, no_results = evaluate(queries, conn)
    fieldnames = list(rows[0].keys())
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Evaluated {test_path}")
    print("Mean precision@5 (n=26):")
    for method in ("keyword", "semantic", "hybrid"):
        print(f"  {method}: {means[method]:.4f}")
    print("No-results rate (n=30):")
    for method in ("keyword", "semantic", "hybrid"):
        print(f"  {method}: {no_results[method] / 30:.4f} ({no_results[method]}/30)")
    print("Fail-soft checks:")
    for row in rows:
        if row["query_id"] in ("q28", "q29"):
            print(f"  {row['query_id']}: {'PASS' if row['fail_soft_check'] else 'FAIL'}")
    delta = (means["hybrid"] - means["keyword"]) * 100
    keyword_rate = no_results["keyword"] / 30 * 100
    hybrid_rate = no_results["hybrid"] / 30 * 100
    print(f"On my 30-query labelled set, hybrid beat keyword-only by {delta:.2f} points on precision@5 and cut no-results from {keyword_rate:.2f}% to {hybrid_rate:.2f}%.")
    print(f"Saved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
