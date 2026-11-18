"""Streamlit UI for the Retail Search Intelligence prototype."""

from __future__ import annotations

import html
import json
import os
import sqlite3
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

import search


ROOT = Path(__file__).parent
EXAMPLES = [
    "something cosy for my bedroom under $30",
    "sturdy toy for a toddler",
    "cute desk decor",
    "gift for a 7 year old",
]
CATEGORIES = ["All", "Home Decor", "Bedding", "Kitchen", "Toys", "Apparel", "Beauty", "Electronics"]


st.set_page_config(page_title="Retail Search Intelligence", page_icon="🛍️", layout="wide")


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        :root { --ink:#1F2328; --muted:#6B7280; --red:#CC0000; --paper:#F7F7F8; --line:#E7E8EB; }
        .stApp { background:var(--paper); color:#1F2328; font-family:'Inter',sans-serif; }
        #MainMenu, footer, header, div[data-testid="stToolbar"] { visibility:hidden; height:0; }
        .block-container { max-width:1440px; padding:2.5rem 4rem 4rem; }
        .hero { text-align:center; padding:1.1rem 0 1.55rem; }
        .eyebrow { color:var(--red); font-size:.72rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; margin-bottom:.55rem; }
        .hero h1 { color:var(--ink); font-size:clamp(2rem,4vw,3.45rem); letter-spacing:-.055em; line-height:1.05; margin:0; }
        .hero p { color:var(--muted); font-size:1.05rem; margin:.75rem 0 1.6rem; }
        .search-row { background:#fff; border:1px solid var(--line); border-radius:16px; box-shadow:0 12px 35px rgba(31,35,40,.08); padding:.45rem; }
        .search-row div[data-baseweb="input"] { border:0; background:#fff; }
        .search-row input { background:#fff !important; color:#1F2328 !important; font-size:1rem; padding:.8rem .9rem; }
        .search-row .stButton button { min-height:3rem; }
        .stButton button { background:#fff; color:#1F2328; border:1px solid #E2E4E8; border-radius:9px; font-weight:700; transition:all .15s ease; }
        .stButton button:hover { background:#fff; border-color:var(--red); color:var(--red); }
        .search-row .stButton button, button[kind="primary"] { background:var(--red); border-color:var(--red); color:#fff; }
        .search-row .stButton button:hover, button[kind="primary"]:hover { background:#a90000; border-color:#a90000; color:#fff; }
        .notice { background:#FFF4F4; border:1px solid #F2CACA; border-radius:10px; color:#7B1E1E; font-size:.78rem; padding:.7rem 1rem; text-align:center; margin:0 auto 1.25rem; }
        .chip-row { display:flex; flex-wrap:wrap; justify-content:center; gap:.45rem; margin:0 0 2.2rem; }
        .chip-row .stButton button, .stButton button[kind="secondary"] { background:#fff; border:1px solid #E2E4E8; color:#1F2328; font-size:.76rem; font-weight:600; padding:.35rem .7rem; }
        .chip-row .stButton button:hover, .stButton button[kind="secondary"]:hover { background:#fff; border-color:var(--red); color:var(--red); }
        [data-testid="stSidebar"] { background:#fff; border-right:1px solid var(--line); }
        [data-testid="stSidebar"] .block-container { padding:2rem 1.35rem; }
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] [data-testid="stWidgetLabel"], [data-testid="stSidebar"] [data-testid="stWidgetLabel"] * { color:#1F2328 !important; }
        [data-testid="stSidebar"] [data-baseweb="select"] *, [data-testid="stSidebar"] [data-baseweb="slider"] *, [data-testid="stSidebar"] input { color:#1F2328 !important; }
        .sidebar-title { font-size:1.05rem; font-weight:800; margin-bottom:.2rem; }
        .sidebar-copy { color:var(--muted); font-size:.8rem; margin-bottom:1.4rem; }
        .price-pill, .category-chip, .stock-chip, .meaning-tag { display:inline-block; border-radius:999px; font-size:.68rem; font-weight:700; padding:.25rem .52rem; }
        .price-pill { background:#FFF0F0; color:var(--red); margin:.35rem 0 1rem; }
        .lab-header { display:flex; align-items:end; justify-content:space-between; margin:0 0 1rem; }
        .lab-header h2 { font-size:1.45rem; letter-spacing:-.035em; margin:0; }
        .lab-header p { color:var(--muted); font-size:.78rem; margin:0; }
        .method-head { border-top:4px solid var(--red); padding:.85rem .05rem .95rem; margin-bottom:.15rem; }
        .method-head.semantic { border-color:#5B63D3; }
        .method-head.hybrid { border-color:#E09B27; }
        .method-head h3 { font-size:1rem; margin:0 0 .25rem; }
        .method-head p { color:var(--muted); font-size:.72rem; margin:0; }
        .result-card { background:#fff; border:1px solid var(--line); border-radius:14px; box-shadow:0 5px 15px rgba(31,35,40,.055); padding:1rem; margin:.7rem 0; min-height:145px; }
        .result-card.found-meaning { border-color:#E3B84D; box-shadow:0 5px 17px rgba(224,155,39,.13); }
        .result-top { display:flex; align-items:center; gap:.45rem; margin-bottom:.65rem; }
        .rank { align-items:center; background:#F0F1F3; border-radius:7px; color:#4B5563; display:inline-flex; font-size:.7rem; font-weight:800; height:24px; justify-content:center; width:24px; }
        .result-card h4 { font-size:.92rem; line-height:1.2; margin:0 0 .45rem; }
        .category-chip { background:#F1F2F4; color:#59616C; margin-right:.3rem; }
        .meaning-tag { background:#FFF5D9; color:#8B5D00; }
        .price { color:var(--ink); font-size:1.05rem; font-weight:800; margin:.55rem 0 .35rem; }
        .rating { color:#D88A00; font-size:.75rem; }
        .stock-chip { margin-left:.35rem; }
        .stock-chip.in { background:#EAF7EF; color:#20733D; }
        .stock-chip.out { background:#F0F1F3; color:#6B7280; }
        .why { color:#7A8088; font-size:.68rem; line-height:1.35; margin-top:.7rem; }
        .empty-card { background:#fff; border:1px dashed #D2D5DA; border-radius:14px; color:#5B6270; font-size:.8rem; line-height:1.5; padding:1.2rem; margin:1.5rem 0; position:relative; z-index:1; }
        .empty-card strong { color:var(--ink); display:block; font-size:.9rem; margin-bottom:.2rem; }
        .empty-card code { color:#5B6270; }
        div[data-testid="stSpinner"], .stSpinner { background:transparent !important; color:#1F2328 !important; }
        div[data-testid="stSpinner"] *, .stSpinner * { color:#1F2328 !important; }
        .metric-strip { background:#fff; border:1px solid var(--line); border-radius:12px; display:flex; justify-content:space-around; margin-top:1.2rem; padding:.8rem; }
        .metric { color:var(--muted); font-size:.7rem; text-align:center; }
        .metric strong { color:var(--ink); display:block; font-size:.95rem; margin-bottom:.15rem; }
        .ai-note { color:#7A8088; font-size:.7rem; margin:.6rem 0 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_resources() -> tuple[sqlite3.Connection, Any, Any, np.ndarray]:
    """Cache the long-lived DB connection, embedding model, and FAISS index."""
    import faiss
    from sentence_transformers import SentenceTransformer
    if not search.DB_PATH.exists():
        import data_gen

        data_gen.build_database()
    embedding_table = sqlite3.connect(search.DB_PATH)
    try:
        has_embeddings = embedding_table.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", ("product_embeddings",)
        ).fetchone() is not None
    finally:
        embedding_table.close()
    if not search.INDEX_PATH.exists() or not search.IDS_PATH.exists() or not has_embeddings:
        import embed

        embed.build_embeddings()

    connection = sqlite3.connect(search.DB_PATH, check_same_thread=False)
    model = SentenceTransformer(search.MODEL_NAME)
    index = faiss.read_index(str(search.INDEX_PATH))
    ids = np.load(search.IDS_PATH)
    return connection, model, index, ids


def result_card(item: dict[str, Any], found_by_meaning: bool = False, explanation: str | None = None) -> str:
    title = html.escape(str(item["title"]))
    category = html.escape(str(item["category"]))
    why = html.escape(explanation or str(item["why_matched"]))
    stock = int(item["stock"])
    stock_text = f"In stock · {stock}" if stock else "Out of stock"
    stock_class = "in" if stock else "out"
    meaning = '<span class="meaning-tag">found by meaning</span>' if found_by_meaning else ""
    return f"""
    <div class="result-card {'found-meaning' if found_by_meaning else ''}">
      <div class="result-top"><span class="rank">{item['rank']}</span>{meaning}</div>
      <h4>{title}</h4>
      <span class="category-chip">{category}</span>
      <div class="price">${float(item['price']):.2f}</div>
      <span class="rating">★ {float(item['rating']):.1f}</span>
      <span class="stock-chip {stock_class}">{stock_text}</span>
      <div class="why">{why}</div>
    </div>
    """


def get_groq_key() -> str | None:
    try:
        return st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    except Exception:
        return os.getenv("GROQ_API_KEY")


def groq_explanations(items: list[dict[str, Any]], api_key: str) -> dict[int, str]:
    """Ask Groq once for grounded one-line explanations; failures are intentionally silent."""
    payload_items = [
        {key: item[key] for key in ("id", "title", "description", "price", "category")}
        for item in items
    ]
    body = {
        "model": "llama-3.1-8b-instant",
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "Return only JSON: an object whose string keys are product ids and values are one short factual explanation. Use only the supplied product fields. Never add facts."},
            {"role": "user", "content": json.dumps({"products": payload_items})},
        ],
    }
    request = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        content = json.loads(response.read().decode("utf-8"))["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    return {int(key): str(value) for key, value in parsed.items() if str(key).isdigit() and isinstance(value, str)}


def render_empty(suggestion: str | None) -> None:
    st.markdown(
        f'<div class="empty-card"><strong>No matches.</strong>Try: {html.escape(suggestion or "broader wording")}</div>',
        unsafe_allow_html=True,
    )


def choose_example(example: str) -> None:
    """Update the search widget before the next rerun renders it."""
    st.session_state.query_input = example
    st.session_state.run_search = True


def main() -> None:
    inject_css()
    if "query_input" not in st.session_state:
        st.session_state.query_input = ""
    if "run_search" not in st.session_state:
        st.session_state.run_search = False
    if "ai_cache" not in st.session_state:
        st.session_state.ai_cache = {}

    st.markdown('<div class="hero"><div class="eyebrow">Search intelligence lab</div><h1>Retail Search Intelligence</h1><p>Keyword vs semantic vs hybrid search, side by side.</p></div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="search-row">', unsafe_allow_html=True)
        search_col, button_col = st.columns([6, 1])
        with search_col:
            st.text_input("Search catalogue", key="query_input", placeholder="Try: something cosy for my bedroom under $30", label_visibility="collapsed")
        with button_col:
            if st.button("Search", type="primary", use_container_width=True):
                st.session_state.run_search = True
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="notice">Synthetic catalogue - not Target data. A small prototype inspired by Target\'s publicly described hybrid search rebuild.</div>', unsafe_allow_html=True)

    chip_cols = st.columns(len(EXAMPLES))
    for col, example in zip(chip_cols, EXAMPLES):
        with col:
            st.button(example, key=f"example_{example}", use_container_width=True, on_click=choose_example, args=(example,))

    with st.sidebar:
        st.markdown('<div class="sidebar-title">Refine results</div><div class="sidebar-copy">Filters apply after each method retrieves its top 50 candidates.</div>', unsafe_allow_html=True)
        max_price_slider = st.slider("Max price", min_value=3, max_value=120, value=120, step=1, format="$%d")
        category = st.selectbox("Category", CATEGORIES)
        in_stock_only = st.toggle("In stock only", value=True)
        ai_enabled = st.toggle("AI explanations (Groq)", value=False)
        clean_query, parsed_price = search.parse_query(st.session_state.query_input)
        if parsed_price is not None:
            st.markdown(f'<span class="price-pill">Detected: under ${parsed_price:g}</span>', unsafe_allow_html=True)

    if not st.session_state.run_search or not st.session_state.query_input.strip():
        st.markdown('<div class="empty-card"><strong>Start with a product idea.</strong>Search above or choose an example query to compare retrieval methods.</div>', unsafe_allow_html=True)
        return

    try:
        with st.spinner("Loading local search resources..."):
            connection, model, index, ids = load_resources()
            search._MODEL, search._INDEX, search._INDEX_IDS = model, index, ids
            output = search.search_all(
                connection,
                st.session_state.query_input,
                category=None if category == "All" else category,
                in_stock_only=in_stock_only,
                max_price_override=None if parsed_price is not None else float(max_price_slider),
                top_n=5,
            )
    except Exception:
        st.markdown('<div class="empty-card"><strong>Search is warming up.</strong>Run <code>python embed.py</code> once to build the local index, then try again.</div>', unsafe_allow_html=True)
        return

    explanations: dict[int, str] = {}
    hybrid = output["hybrid"]
    if ai_enabled and hybrid and get_groq_key():
        cache_key = tuple(item["id"] for item in hybrid)
        if cache_key not in st.session_state.ai_cache:
            try:
                st.session_state.ai_cache[cache_key] = groq_explanations(hybrid, get_groq_key() or "")
            except Exception:
                st.session_state.ai_cache[cache_key] = {}
        explanations = st.session_state.ai_cache[cache_key]

    st.markdown('<div class="lab-header"><div><h2>Comparison lab</h2><p>Same query, same filters, three retrieval strategies.</p></div></div>', unsafe_allow_html=True)
    methods = [("keyword", "Keyword (BM25)", "Exact terms and lexical relevance.", ""), ("semantic", "Semantic (embeddings)", "Meaning and contextual similarity.", "semantic"), ("hybrid", "Hybrid (RRF)", "Ranks combined with no score weights.", "hybrid")]
    keyword_ids = {item["id"] for item in output["keyword"]}
    columns = st.columns(3)
    for column, (method, title, explainer, css_class) in zip(columns, methods):
        with column:
            st.markdown(f'<div class="method-head {css_class}"><h3>{title}</h3><p>{explainer}</p></div>', unsafe_allow_html=True)
            items = output[method]
            if not items:
                render_empty(output.get("suggestion"))
            for item in items:
                explanation = explanations.get(item["id"]) if method == "hybrid" else None
                st.markdown(result_card(item, method == "hybrid" and item["id"] not in keyword_ids, explanation), unsafe_allow_html=True)

    latency = output["latency_ms"]
    st.markdown(
        f'<div class="metric-strip"><div class="metric"><strong>{latency["keyword"]:.1f} ms</strong>Keyword latency · {len(output["keyword"])} results</div><div class="metric"><strong>{latency["semantic"]:.1f} ms</strong>Semantic latency · {len(output["semantic"])} results</div><div class="metric"><strong>{latency["hybrid"]:.1f} ms</strong>Hybrid latency · {len(output["hybrid"])} results</div></div>',
        unsafe_allow_html=True,
    )
    if ai_enabled and not get_groq_key():
        st.markdown('<div class="ai-note">AI explanations unavailable; showing local grounded match reasons.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
