import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import data_gen
from search import keyword_search, parse_query, rrf


class SearchTests(unittest.TestCase):
    def test_empty_query(self):
        with sqlite3.connect(data_gen.DB_PATH) as conn:
            self.assertEqual(keyword_search(conn, ""), [])

    def test_injection_style_string(self):
        with sqlite3.connect(data_gen.DB_PATH) as conn:
            self.assertLessEqual(len(keyword_search(conn, '"* - () AND OR NOT')), 50)

    def test_no_match_query(self):
        with sqlite3.connect(data_gen.DB_PATH) as conn:
            self.assertEqual(keyword_search(conn, "zzzz-no-product"), [])

    def test_price_parsing(self):
        self.assertEqual(parse_query("blanket under $30"), ("blanket", 30.0))

    def test_rrf_ordering(self):
        self.assertEqual([item[0] for item in rrf([(1, 1), (2, 3)], [(2, 1), (3, 2)])], [2, 1, 3])

    def test_labels_survive_rebuild(self):
        with tempfile.TemporaryDirectory() as folder:
            query_path = Path(folder) / "test_queries.json"
            query_path.write_text(json.dumps([{"query_id": "q01", "relevant_product_ids": [1, 2]}]), encoding="utf-8")
            old_path = data_gen.TEST_QUERIES_PATH
            data_gen.TEST_QUERIES_PATH = query_path
            try:
                db_path = Path(folder) / "catalog.db"
                data_gen.build_database(db_path)
                conn = sqlite3.connect(db_path)
                try:
                    self.assertEqual(conn.execute("SELECT product_id FROM relevance_judgements ORDER BY product_id").fetchall(), [(1,), (2,)])
                finally:
                    conn.close()
            finally:
                data_gen.TEST_QUERIES_PATH = old_path


if __name__ == "__main__":
    unittest.main()
