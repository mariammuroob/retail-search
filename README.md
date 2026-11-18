# Retail Search Intelligence

## Research question

For a small synthetic retail catalogue, does combining lexical retrieval with semantic retrieval improve top-five relevance and reduce no-results compared with keyword search alone?

Synthetic catalogue - not Target data.

## Architecture

Indexing happens offline:

1. `data_gen.py` creates deterministic products, SQLite tables, and the FTS5 index.
2. `embed.py` embeds `title. description` with `all-MiniLM-L6-v2`, normalizes vectors, and writes a FAISS `IndexFlatIP` plus ID map.
3. Relevance labels are read from the root `test_queries.json` and synchronized into SQLite after rebuilds.

Searching happens at runtime:

1. `search.py` retrieves up to 50 keyword and semantic candidates.
2. SQL filters are applied after retrieval.
3. RRF combines ranks using `1 / (60 + keyword_rank) + 1 / (60 + semantic_rank)`.
4. `app.py` renders keyword, semantic, and hybrid results side by side.

RRF is a standard method for combining rankings.

## Stack

Python, Streamlit, SQLite FTS5, `sentence-transformers` with `all-MiniLM-L6-v2`, NumPy, and FAISS. There is no FastAPI, custom frontend, LangChain, scraping, per-user pricing, or extra service.

## Run locally

```powershell
python data_gen.py
python embed.py
streamlit run app.py
```

If `catalog.db`, the FAISS index, or the ID map is missing, the Streamlit app auto-builds them on first run by calling the existing indexing functions. This lazy auto-build is the simpler option.

## Label and evaluate

The existing 30-query file is copied into the project root and is the single source of truth:

```powershell
python label_helper.py
python label_helper.py --interactive
python eval.py
```

The evaluator scores queries 1-26, skips precision scoring for 27 and 30, and runs fail-soft checks for 28 and 29. Results are written to `eval_results.csv`.

## Screenshot

`[Screenshot placeholder: add a screenshot of the Streamlit comparison lab here.]`

## Target reference

This prototype is inspired by the Google Cloud blog, 3 Sep 2025, describing Target's publicly reported hybrid-search rebuild. Target's reported results are Target's own, not mine.

## Not claimed

- AlloyDB or ScaNN parity
- Production scale or production readiness
- Target's numbers as my results
- Real Target data
- Personalised pricing
- Latency beyond my local runs
- A replica of Target's system

## Results

Placeholder: fill after labelling and evaluation.

`n=26, labels hand-judged by me`

| Method | Mean precision@5 | No-results rate |
|---|---:|---:|
| Keyword (BM25) | TBD | TBD |
| Semantic | TBD | TBD |
| Hybrid (RRF) | TBD | TBD |

## Streamlit Community Cloud

- Main file: `app.py`
- Add `GROQ_API_KEY` under App settings → Secrets only if optional AI explanations are wanted.
- Phone test checklist: search field is usable; example chips wrap; sidebar filters are reachable; all three columns stack or remain readable; cards show price, rating, and stock; empty states show a suggestion; no API key still leaves local results working.
