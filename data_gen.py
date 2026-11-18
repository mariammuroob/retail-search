"""Generate a small deterministic retail catalogue and SQLite search database."""

from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path


DB_PATH = Path(__file__).with_name("catalog.db")
TEST_QUERIES_PATH = Path(__file__).with_name("test_queries.json")
SEED = 42

CATEGORIES = [
    "Home Decor",
    "Bedding",
    "Kitchen",
    "Toys",
    "Apparel",
    "Beauty",
    "Electronics",
]

BRANDS = ["Northstar", "BrightNest", "Willow & Co", "Merryday", "Cedar Lane"]

CATALOGUE = {
    "Home Decor": ["candle", "vase", "wall art", "throw pillow", "table lamp"],
    "Bedding": ["blanket", "sheet set", "duvet cover", "pillow", "mattress pad"],
    "Kitchen": ["mug", "mixing bowl", "cutting board", "water bottle", "tea towel"],
    "Toys": ["plush bear", "building set", "puzzle", "play kitchen", "toy truck"],
    "Apparel": ["hoodie", "socks", "t-shirt", "cardigan", "sneakers"],
    "Beauty": ["face cream", "shampoo", "lip balm", "bath salts", "hair brush"],
    "Electronics": ["desk speaker", "phone stand", "headphones", "reading light", "power bank"],
}

ADJECTIVES = [
    "cosy",
    "sturdy",
    "cute",
    "durable",
    "soft",
    "sleek",
    "lightweight",
    "cheerful",
    "handy",
    "modern",
]

DESCRIPTION_TEMPLATES = [
    "A {adj} {item} made for everyday use. Its simple design adds a practical touch to your home.",
    "This {adj} {item} is easy to use and built for busy days. A thoughtful pick for gifting or keeping.",
    "Bring home a {adj} {item} with a clean finish and comfortable feel. It is a reliable everyday essential.",
]


def sync_relevance_judgements(conn: sqlite3.Connection, path: Path | None = None) -> None:
    """Rebuild labels from JSON so the JSON remains the only label source of truth."""
    path = path or TEST_QUERIES_PATH
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        queries = json.load(handle)
    conn.execute("DELETE FROM relevance_judgements")
    labels = []
    for entry in queries:
        query_id = int(str(entry["query_id"]).lstrip("q"))
        labels.extend((query_id, int(product_id), 1) for product_id in entry.get("relevant_product_ids", []))
    conn.executemany(
        "INSERT INTO relevance_judgements (query_id, product_id, relevant) VALUES (?, ?, ?)",
        labels,
    )


def make_products() -> list[tuple]:
    """Return 400 products; the fixed seed keeps demos and evaluations repeatable."""
    rng = random.Random(SEED)
    products: list[tuple] = []

    # These rows make later semantic-search demos useful without changing keyword rules.
    curated = [
        ("Cosy Knit Blanket", "A cosy knitted layer with a soft feel for quiet evenings.", "Bedding", 22.99, 4.8),
        ("Plush Fleece Throw", "A warm layer for chilly evenings, made from fluffy fleece.", "Bedding", 24.99, 4.7),
        ("Cloud Nap Cover", "A gentle cover for restful naps with a brushed, pillowy texture.", "Bedding", 28.50, 4.6),
        ("Soft Lounge Cardigan", "A relaxed, light layer for quiet mornings and cool rooms.", "Apparel", 29.00, 4.4),
    ]
    for index, (title, description, category, price, rating) in enumerate(curated, start=1):
        products.append((index, title, description, BRANDS[index % len(BRANDS)], category, price, rating))

    for product_id in range(len(products) + 1, 401):
        category = rng.choice(CATEGORIES)
        item = rng.choice(CATALOGUE[category])
        adj = rng.choice(ADJECTIVES)
        title = f"{adj.title()} {item.title()}"
        description = rng.choice(DESCRIPTION_TEMPLATES).format(adj=adj, item=item)
        products.append(
            (
                product_id,
                title,
                description,
                rng.choice(BRANDS),
                category,
                round(rng.uniform(3, 120), 2),
                round(rng.uniform(2.5, 5.0), 1),
            )
        )
    return products


def build_database(path: Path = DB_PATH) -> None:
    """Drop and recreate every table so running this script is safe and repeatable."""
    products = make_products()
    rng = random.Random(SEED + 1)

    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(
            """
            DROP TABLE IF EXISTS product_fts;
            DROP TABLE IF EXISTS product_embeddings;
            DROP TABLE IF EXISTS search_log;
            DROP TABLE IF EXISTS relevance_judgements;
            DROP TABLE IF EXISTS inventory;
            DROP TABLE IF EXISTS products;

            CREATE TABLE products (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                brand TEXT NOT NULL,
                category TEXT NOT NULL,
                price REAL NOT NULL,
                rating REAL NOT NULL
            );
            CREATE TABLE inventory (
                product_id INTEGER PRIMARY KEY,
                stock_quantity INTEGER NOT NULL,
                FOREIGN KEY(product_id) REFERENCES products(id)
            );
            CREATE TABLE relevance_judgements (
                query_id INTEGER NOT NULL,
                product_id INTEGER NOT NULL,
                relevant INTEGER NOT NULL,
                FOREIGN KEY(product_id) REFERENCES products(id)
            );
            CREATE TABLE search_log (
                id INTEGER PRIMARY KEY,
                query TEXT NOT NULL,
                filters TEXT NOT NULL,
                top5_ids TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                fallback_used INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.executemany(
            "INSERT INTO products (id, title, description, brand, category, price, rating) VALUES (?, ?, ?, ?, ?, ?, ?)",
            products,
        )
        curated_count = 4
        inventory = [
            (product[0], rng.randint(1, 80) if product[0] <= curated_count else (0 if rng.random() < 0.12 else rng.randint(1, 80)))
            for product in products
        ]
        conn.executemany("INSERT INTO inventory (product_id, stock_quantity) VALUES (?, ?)", inventory)
        # Content-linked FTS keeps the searchable copy aligned with products.
        conn.execute(
            """CREATE VIRTUAL TABLE product_fts USING fts5(
                title, description, content='products', content_rowid='id'
            )"""
        )
        conn.execute("INSERT INTO product_fts(product_fts) VALUES ('rebuild')")
        sync_relevance_judgements(conn)
        conn.commit()
    conn.close()

    print(f"Built {path} with {len(products)} products and FTS5 keyword search.")


if __name__ == "__main__":
    build_database()
