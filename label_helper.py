"""Propose and manually record relevance labels for the existing query file.

The helper never decides the final labels. It only prints candidates; a person
confirms 2-3 product ids per query in interactive mode.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any


ROOT = Path(__file__).parent
DB_PATH = ROOT / "catalog.db"
DEFAULT_CANDIDATE_OUTPUT = ROOT / "label_candidates.json"
CATALOG_CATEGORIES = {
    "home": "Home Decor",
    "bedding": "Bedding",
    "kitchen": "Kitchen",
    "toys": "Toys",
    "apparel": "Apparel",
    "beauty": "Beauty",
    "electronics": "Electronics",
}
STOPWORDS = {"a", "an", "and", "for", "from", "in", "my", "of", "on", "the", "to", "with"}


def find_test_file() -> Path:
    """Use the project-root copy by default; --test-file remains an override."""
    option = ROOT / "test_queries.json"
    if not option.exists():
        raise FileNotFoundError("./test_queries.json was not found; it was not recreated.")
    return option.resolve()


def load_queries(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or len(data) != 30:
        raise ValueError("test_queries.json must be the existing 30-entry JSON array")
    return data


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def propose_candidates(conn: sqlite3.Connection, entry: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
    """Rank likely candidates using the file's required_tags/category rule."""
    category = CATALOG_CATEGORIES.get(str(entry.get("category", "")).lower())
    required_tags = entry.get("required_tags") or []
    query_tokens = _tokens(str(entry.get("query", "")))
    rows = conn.execute(
        "SELECT id, title, description, price, category FROM products ORDER BY id"
    ).fetchall()
    candidates: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        product_text = f"{row[1]} {row[2]}".lower()
        tag_hits = sum(1 for tag in required_tags if _tokens(str(tag)) & _tokens(product_text))
        query_hits = len(query_tokens & _tokens(product_text))
        category_hit = bool(category and row[4] == category)
        # Category and tag overlap are only proposal signals; a person confirms relevance.
        score = (6 if category_hit else 0) + tag_hits * 4 + query_hits
        if score:
            candidates.append(
                (
                    score,
                    {
                        "id": int(row[0]),
                        "title": row[1],
                        "description": row[2],
                        "price": row[3],
                        "category": row[4],
                        "proposal_score": score,
                    },
                )
            )
    candidates.sort(key=lambda item: (-item[0], item[1]["id"]))
    tag_candidates = [item for _score, item in candidates]
    semantic: list[dict[str, Any]] = []
    try:
        import search

        semantic_rows = search._semantic_search_with_scores(str(entry.get("query", "")), limit=20)
        semantic_ids = [product_id for product_id, _rank, _score in semantic_rows]
        placeholders = ",".join("?" for _ in semantic_ids)
        if semantic_ids:
            rows_by_id = {
                int(row[0]): row
                for row in conn.execute(
                    f"SELECT id, title, description, price, category FROM products WHERE id IN ({placeholders})",
                    semantic_ids,
                ).fetchall()
            }
            query_words = (_tokens(str(entry.get("query", ""))) - STOPWORDS) | {_token for tag in required_tags for _token in _tokens(str(tag))}
            semantic_matches = []
            for product_id in semantic_ids:
                row = rows_by_id.get(product_id)
                if not row:
                    continue
                if query_words & _tokens(f"{row[1]} {row[2]}"):
                    continue
                semantic_matches.append((0 if category and row[4] == category else 1, {"id": int(row[0]), "title": row[1], "description": row[2], "price": row[3], "category": row[4]}))
            semantic = [product for _category_distance, product in sorted(semantic_matches, key=lambda item: item[0])]
        combined = tag_candidates + semantic
    except Exception:
        combined = tag_candidates
    # Reserve up to three slots for semantic-only suggestions so lexical matches
    # do not hide the cases a human reviewer most needs to inspect.
    combined = tag_candidates[: max(0, limit - 3)] + semantic[:3] + tag_candidates[max(0, limit - 3) :]
    unique: list[dict[str, Any]] = []
    seen: set[int] = set()
    for product in combined:
        product = {key: product[key] for key in ("id", "title", "description", "price", "category")}
        if product["id"] not in seen:
            unique.append(product)
            seen.add(product["id"])
    return unique[:limit]


def build_candidate_file(conn: sqlite3.Connection, queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "query_id": entry["query_id"],
            "query": entry["query"],
            "category": entry.get("category"),
            "required_tags": entry.get("required_tags", []),
            "rule_source": "relevant_if" if "relevant_if" in entry else "required_tags+category",
            "candidates": propose_candidates(conn, entry),
        }
        for entry in queries if int(str(entry["query_id"]).lstrip("q")) not in (28, 29)
    ]


def _save_json(path: Path, queries: list[dict[str, Any]]) -> None:
    """Replace JSON atomically so an interrupted prompt cannot leave half a file."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(queries, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def save_one_label(path: Path, queries: list[dict[str, Any]], query_number: int, ids: list[int], conn: sqlite3.Connection) -> None:
    """Persist one answer immediately in both JSON and the convenience DB table."""
    entry = next(item for item in queries if int(str(item["query_id"]).lstrip("q")) == query_number)
    entry["relevant_product_ids"] = ids
    _save_json(path, queries)
    conn.execute("DELETE FROM relevance_judgements WHERE query_id = ?", (query_number,))
    conn.executemany(
        "INSERT INTO relevance_judgements (query_id, product_id, relevant) VALUES (?, ?, ?)",
        [(query_number, product_id, 1) for product_id in ids],
    )
    conn.commit()


def parse_label_input(raw: str, allowed_ids: set[int]) -> tuple[bool, list[int], str]:
    """Validate none or 1-3 unique candidate IDs without ever calling int unsafely."""
    value = raw.strip().lower()
    if value == "none":
        return True, [], ""
    parts = [part.strip() for part in raw.split(",")]
    if not parts or any(not part.isdigit() for part in parts) or not 1 <= len(parts) <= 3:
        return False, [], "Invalid, try again"
    ids = [int(part) for part in parts]
    invalid = sorted(set(ids) - allowed_ids)
    if invalid:
        return False, [], f"Warning: ids not in the shown candidates: {invalid}\nInvalid, try again"
    if len(set(ids)) != len(ids):
        return False, [], "Invalid, try again"
    return True, ids, ""


def ask_for_label(prompt: str, allowed_ids: set[int]) -> list[int]:
    while True:
        raw = input(prompt).strip()
        valid, ids, message = parse_label_input(raw, allowed_ids)
        if valid:
            return ids
        print(message)


def main() -> None:
    parser = argparse.ArgumentParser(description="Propose candidates and optionally record hand-checked labels.")
    parser.add_argument("--interactive", action="store_true", help="Prompt for confirmed ids and save after each query.")
    parser.add_argument("--test-file", type=Path, default=None, help="Override the existing test_queries.json path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_CANDIDATE_OUTPUT, help="Candidate proposal JSON output.")
    args = parser.parse_args()
    test_path = (args.test_file or find_test_file()).resolve()
    queries = load_queries(test_path)
    with sqlite3.connect(DB_PATH) as conn:
        proposals = build_candidate_file(conn, queries)
        args.output.write_text(json.dumps(proposals, indent=2) + "\n", encoding="utf-8")
        print(f"Candidate proposals written to {args.output.resolve()}")
        if args.interactive:
            print("Type 1-3 comma-separated candidate ids, 'none' for an empty label set, or Ctrl+C to stop.")
            by_query = {item["query_id"]: item for item in proposals}
            for entry in queries:
                query_number = int(str(entry["query_id"]).lstrip("q"))
                if query_number in (28, 29):
                    continue
                proposal = by_query[entry["query_id"]]
                if "relevant_product_ids" in entry:
                    while True:
                        decision = input(f"\n{entry['query_id']} already has labels {entry['relevant_product_ids']}. Keep, redo, or skip? [k/r/s]: ").strip().lower()
                        if decision in {"k", "keep", "s", "skip"}:
                            break
                        if decision in {"r", "redo"}:
                            break
                        print("Invalid, try again")
                    if decision in {"k", "keep", "s", "skip"}:
                        continue
                print(f"\n{entry['query_id']}: {entry['query']} | category={entry.get('category')} | tags={entry.get('required_tags', [])}")
                if query_number in (27, 30):
                    print("Loose/off-catalogue query: type 'none' to leave an empty label set.")
                for product in proposal["candidates"]:
                    print(f"  {product['id']}: {product['title']} | {product['description']} | ${product['price']:.2f} | {product['category']}")
                allowed = {product["id"] for product in proposal["candidates"]}
                ids = ask_for_label("Confirmed ids: ", allowed)
                save_one_label(test_path, queries, query_number, ids, conn)
                print(f"Saved {entry['query_id']} immediately to {test_path}.")


if __name__ == "__main__":
    main()
