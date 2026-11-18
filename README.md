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

`n=18, labels hand-judged by me`. Empty-label queries were excluded from precision@5 and hit@5. q27/q30 were not scored; q28/q29 were used only for the fail-soft check.

No-results rates are shown over all 30 queries and over the 23 queries whose category exists in the catalogue.

| Run | Method | Precision@5 | Hit@5 | No-results / 30 | No-results / category-existing 23 |
|---|---|---:|---:|---:|---:|
| Before home fix | Keyword (BM25) | 0.0556 | 0.2778 | 19/30 (0.6333) | 12/23 (0.5217) |
| Before home fix | Keyword + stemming | 0.0667 | 0.3333 | 16/30 (0.5333) | 9/23 (0.3913) |
| Before home fix | Semantic | 0.1556 | 0.7222 | 7/30 (0.2333) | 0/23 (0.0000) |
| Before home fix | Hybrid (RRF) | 0.1667 | 0.7222 | 7/30 (0.2333) | 0/23 (0.0000) |
| After home fix | Keyword (BM25) | 0.0778 | 0.3333 | 19/30 (0.6333) | 12/23 (0.5217) |
| After home fix | Keyword + stemming | 0.0778 | 0.3889 | 15/30 (0.5000) | 8/23 (0.3478) |
| After home fix | Semantic | 0.1556 | 0.7222 | 7/30 (0.2333) | 0/23 (0.0000) |
| After home fix | Hybrid (RRF) | 0.1889 | 0.7778 | 7/30 (0.2333) | 0/23 (0.0000) |

On 18 scoreable queries, semantic and hybrid retrieved a relevant product in the top 5 for 13-14 queries versus 6-7 for keyword-only (with or without stemming). Hybrid and semantic are within one query of each other, which is not a meaningful difference at this sample size.

Target's reported numbers are Target's own and not mine.

### Limitations

- n=18; 8 of the 26 candidate queries have no relevant product in the synthetic catalogue (no grocery, sports or organiser items) and are excluded.
- Labels were reviewed and confirmed by me; some were suggested by an AI assistant.
- The home-category mapping bug was found after seeing the first results; both runs are reported.
- q28/q29 fail-soft check is a FAIL: semantic search always returns nearest neighbours and there is no similarity threshold. With real query logs I would add one.
- q11's label (a Home Decor vase) sits outside the Kitchen filter and cannot be retrieved.
- The candidate proposal tool filters by category and ignores price limits.
- Many catalogue descriptions are near-identical template text; the catalogue is synthetic.

## Streamlit Community Cloud

- Main file: `app.py`
- Add `GROQ_API_KEY` under App settings → Secrets only if optional AI explanations are wanted.
- Phone test checklist: search field is usable; example chips wrap; sidebar filters are reachable; all three columns stack or remain readable; cards show price, rating, and stock; empty states show a suggestion; no API key still leaves local results working.
