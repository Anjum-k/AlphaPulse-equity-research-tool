import os
import time
# pyrefly: ignore[missing-import]
import streamlit as st
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import (
    AVAILABLE_LLMS,
    AVAILABLE_EMBEDDINGS,
    VECTOR_STORES,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
)
from core.scraper import NewsArticleScraper
from core.vector_store import VectorStoreManager
from core.rag_engine import EquityRAGEngine
from core.utils import parse_urls_from_text, extract_domain

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AlphaPulse | Equity Research Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Modern Financial Terminal Aesthetics
st.markdown("""
<style>
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #1A2234;
        border: 1px solid #2B3A55;
        border-radius: 8px;
        padding: 12px 18px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    div[data-testid="stMetricValue"] {
        color: #00D084;
        font-weight: 700;
    }
    /* Buttons */
    .stButton>button {
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 208, 132, 0.3);
    }
    /* Source Card */
    .source-card {
        background: #141B2D;
        border: 1px solid #28354E;
        border-left: 4px solid #00D084;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    .source-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #64B5F6;
        text-decoration: none;
    }
    .source-title:hover {
        text-decoration: underline;
    }
    .source-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        background: #1E293B;
        color: #94A3B8;
        margin-right: 8px;
    }
    .badge-score {
        background: rgba(0, 208, 132, 0.15);
        color: #00D084;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "indexed_articles" not in st.session_state:
    st.session_state.indexed_articles = []
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# ---------------------------------------------------------
# Sidebar: Setup & Ingestion
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 📈 AlphaPulse Terminal")
    st.caption("AI-Powered News RAG Engine for Equity Research")
    st.divider()

    # 1. API Configuration
    st.markdown("### 🔑 API Configuration")
    env_key = os.getenv("OPENAI_API_KEY", "")
    api_key_input = st.text_input(
        "OpenAI API Key",
        value=env_key,
        type="password",
        placeholder="sk-proj-...",
        help="Enter your OpenAI API key or set it in .env file.",
    )
    api_key = api_key_input.strip() if api_key_input else None

    if api_key:
        st.success("API Key Active", icon="✅")
    else:
        st.warning("Please provide an OpenAI API key to process articles and ask questions.", icon="⚠️")

    st.divider()

    # 2. Engine & Vector Store Selection
    st.markdown("### ⚙️ Engine Settings")
    col_db, col_llm = st.columns(2)
    with col_db:
        selected_db = st.selectbox(
            "Vector Database",
            options=VECTOR_STORES,
            index=0,
            help="FAISS for high-performance in-memory indexing, ChromaDB for persistent collections.",
        )
    with col_llm:
        selected_llm = st.selectbox(
            "LLM Model",
            options=AVAILABLE_LLMS,
            index=0,
        )

    selected_embedding = st.selectbox(
        "Embedding Model",
        options=AVAILABLE_EMBEDDINGS,
        index=0,
    )

    with st.expander("🛠️ Advanced RAG Hyperparameters"):
        chunk_size = st.slider("Chunk Size", 300, 2000, DEFAULT_CHUNK_SIZE, step=100)
        chunk_overlap = st.slider("Chunk Overlap", 50, 500, DEFAULT_CHUNK_OVERLAP, step=50)
        top_k = st.slider("Top Sources to Retrieve (k)", 2, 8, DEFAULT_TOP_K)

    st.divider()

    # 3. Input Modes
    st.markdown("### 📰 Ingestion & Sources")
    ingest_tab1, ingest_tab2 = st.tabs(["🌐 Direct URLs", "📁 Batch / File Upload"])

    urls_to_process: List[str] = []

    with ingest_tab1:
        st.caption("Enter 1-3 direct financial news URLs:")
        url_1 = st.text_input("URL 1", placeholder="https://finance.yahoo.com/news/...")
        url_2 = st.text_input("URL 2", placeholder="https://www.reuters.com/business/...")
        url_3 = st.text_input("URL 3", placeholder="https://www.bloomberg.com/news/...")

        direct_urls = [u.strip() for u in [url_1, url_2, url_3] if u.strip()]
        if direct_urls:
            urls_to_process.extend(direct_urls)

    with ingest_tab2:
        st.caption("Batch process multiple URLs from file or text:")
        uploaded_file = st.file_uploader("Upload URLs text file (.txt)", type=["txt"])
        
        batch_text_input = st.text_area(
            "Or paste URLs (one per line):",
            placeholder="https://finance.yahoo.com/news/apple...\nhttps://finance.yahoo.com/news/nvidia...",
            height=120,
        )

        # Sample URLs prefill button
        if st.button("📂 Load Sample Tech Earnings URLs", use_container_width=True):
            sample_path = Path(__file__).parent / "sample_data" / "sample_urls.txt"
            if sample_path.exists():
                batch_text_input = sample_path.read_text(encoding="utf-8")
                st.info("Loaded sample URLs into batch processor.")
                parsed_sample = parse_urls_from_text(batch_text_input)
                urls_to_process.extend(parsed_sample)

        if uploaded_file is not None:
            file_content = uploaded_file.read().decode("utf-8")
            parsed_file_urls = parse_urls_from_text(file_content)
            urls_to_process.extend(parsed_file_urls)
            st.caption(f"Found {len(parsed_file_urls)} valid URLs in uploaded file.")

        if batch_text_input.strip():
            parsed_text_urls = parse_urls_from_text(batch_text_input)
            urls_to_process.extend(parsed_text_urls)

    # Deduplicate preserving order
    urls_to_process = list(dict.fromkeys(urls_to_process))

    if urls_to_process:
        st.info(f"Ready to ingest {len(urls_to_process)} article(s).")

    # Ingestion Actions
    col_act1, col_act2 = st.columns(2)
    with col_act1:
        start_ingest = st.button("⚡ Ingest & Index", type="primary", use_container_width=True)
    with col_act2:
        reset_index = st.button("🗑️ Reset DB", use_container_width=True)

    if reset_index:
        v_manager = VectorStoreManager(store_type=selected_db, api_key=api_key)
        v_manager.reset_index()
        st.session_state.indexed_articles = []
        st.session_state.total_chunks = 0
        st.session_state.history = []
        st.success(f"{selected_db} vector store cleared!")

    # Ingestion Execution Logic
    if start_ingest:
        if not api_key:
            st.error("Please enter a valid OpenAI API Key first!")
        elif not urls_to_process:
            st.error("No valid URLs provided to ingest. Add URLs above.")
        else:
            with st.status(f"Ingesting {len(urls_to_process)} articles into {selected_db}...", expanded=True) as status:
                # 1. Scraping
                st.write("🕷️ Fetching and parsing web articles...")
                scraper = NewsArticleScraper()
                progress_bar = st.progress(0.0)

                def update_progress(curr, tot, u, succ):
                    progress_bar.progress(curr / tot)
                    if succ:
                        st.write(f"✅ Ingested: [{extract_domain(u)}] {u[:50]}...")
                    else:
                        st.write(f"⚠️ Failed to fetch: {u[:50]}...")

                raw_documents = scraper.scrape_multiple_urls(urls_to_process, progress_callback=update_progress)

                if not raw_documents:
                    status.update(label="Scraping failed: no content could be retrieved from the URLs.", state="error")
                else:
                    st.write(f"📄 Successfully retrieved {len(raw_documents)} articles.")
                    
                    # 2. Text Splitting
                    st.write("✂️ Splitting content into semantically rich chunks...")
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        separators=["\n\n", "\n", ". ", " ", ""],
                    )
                    chunks = text_splitter.split_documents(raw_documents)
                    st.write(f"🧩 Created {len(chunks)} text chunks.")

                    # 3. Vector Database Embedding
                    st.write(f"🧠 Generating embeddings via {selected_embedding} and building {selected_db} index...")
                    try:
                        v_manager = VectorStoreManager(
                            store_type=selected_db,
                            embedding_model=selected_embedding,
                            api_key=api_key,
                        )
                        v_manager.build_from_documents(chunks)
                        
                        st.session_state.indexed_articles = [doc.metadata for doc in raw_documents]
                        st.session_state.total_chunks = len(chunks)
                        status.update(label=f"Successfully indexed into {selected_db}!", state="complete", expanded=False)
                        st.success(f"Indexed {len(raw_documents)} articles ({len(chunks)} chunks) into {selected_db}!")
                    except Exception as e:
                        status.update(label=f"Vector store error: {str(e)}", state="error")
                        st.error(f"Embedding error: {str(e)}")

# ---------------------------------------------------------
# Main Page: Dashboard & Research Interface
# ---------------------------------------------------------
st.markdown("# 📈 AlphaPulse | Equity News Research & RAG Assistant")
st.markdown(
    "Synthesize complex financial news into actionable equity research summaries, "
    "earnings takeaways, and market impact analysis backed by verified sources."
)

# Top Metric Banner
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Articles Ingested", len(st.session_state.indexed_articles))
with m2:
    st.metric("Vector Chunks", st.session_state.total_chunks)
with m3:
    st.metric("Vector Database", selected_db)
with m4:
    st.metric("Active LLM", selected_llm)

st.divider()

# Check if index exists on disk or in session
v_manager = VectorStoreManager(
    store_type=selected_db,
    embedding_model=selected_embedding,
    api_key=api_key,
)
has_index = v_manager.has_existing_index() or st.session_state.total_chunks > 0

if not has_index:
    st.info(
        "💡 **Getting Started:** Add news article URLs in the left sidebar (either via Direct URLs or Batch Process), "
        "and click **'⚡ Ingest & Index'** to build your financial intelligence vector database.",
        icon="ℹ️",
    )

# ---------------------------------------------------------
# Research Query Section
# ---------------------------------------------------------
st.markdown("### 🔎 Conduct Equity Research")

# Quick Prompt Chips
st.caption("Quick Queries:")
quick_cols = st.columns(3)
with quick_cols[0]:
    if st.button("📊 Revenue & Earnings Highlights", use_container_width=True):
        st.session_state.last_query = "What are the key revenue drivers, financial figures, and earnings results reported?"
with quick_cols[1]:
    if st.button("⚠️ Headwinds, Margins & Risks", use_container_width=True):
        st.session_state.last_query = "What cost pressures, margin headwinds, and macro risks did management highlight?"
with quick_cols[2]:
    if st.button("🎯 Guidance & Growth Catalysts", use_container_width=True):
        st.session_state.last_query = "What is the forward-looking guidance, CapEx plan, and strategic growth catalysts?"

user_query = st.text_input(
    "Ask a research question regarding the indexed news articles:",
    value=st.session_state.last_query,
    placeholder="e.g., What are the details of the fourth quarter earnings and iPhone sales growth?",
)

run_query = st.button("🔍 Generate Research Analysis", type="primary")

if run_query:
    if not api_key:
        st.error("Please provide an OpenAI API Key in the sidebar first.")
    elif not user_query.strip():
        st.warning("Please enter a research question.")
    elif not has_index:
        st.error("No articles have been indexed yet. Ingest articles from the sidebar first.")
    else:
        with st.spinner("Analyzing indexed financial reports and synthesizing answer..."):
            try:
                rag_engine = EquityRAGEngine(
                    vector_store_manager=v_manager,
                    llm_model=selected_llm,
                    api_key=api_key,
                )
                start_time = time.time()
                result = rag_engine.answer_question(user_query, top_k=top_k)
                elapsed_time = round(time.time() - start_time, 2)

                # Append to history
                st.session_state.history.insert(0, {
                    "query": user_query,
                    "answer": result["answer"],
                    "sources": result["sources"],
                    "time": elapsed_time,
                })

            except Exception as e:
                st.error(f"Error executing research query: {str(e)}")

# ---------------------------------------------------------
# Results Display
# ---------------------------------------------------------
if st.session_state.history:
    latest = st.session_state.history[0]

    st.markdown("---")
    res_col1, res_col2 = st.columns([7, 3])

    with res_col1:
        st.markdown(f"### 📋 Research Findings: *'{latest['query']}'*")
        st.caption(f"⚡ Synthesized in {latest['time']}s using {selected_llm} & {selected_db}")
        st.markdown(latest["answer"])

        # Export research report
        report_text = f"""# Equity News Research Report
**Query**: {latest['query']}
**Model**: {selected_llm} | **Vector Store**: {selected_db}
**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Analysis Findings
{latest['answer']}

## Source Citations
"""
        for i, s in enumerate(latest["sources"], 1):
            report_text += f"\n{i}. [{s['title']}]({s['url']}) (Domain: {s['domain']})\n   Excerpt: {s['excerpt']}\n"

        st.download_button(
            label="📥 Download Research Brief (.md)",
            data=report_text,
            file_name=f"equity_research_{int(time.time())}.md",
            mime="text/markdown",
        )

    with res_col2:
        st.markdown("### 📚 Source Attribution")
        st.caption("Direct references retrieved from the vector index:")

        if not latest["sources"]:
            st.write("No direct source chunks retrieved.")
        else:
            for idx, src in enumerate(latest["sources"], 1):
                score_str = f"Score: {src['relevance_score']}" if src.get("relevance_score") is not None else ""
                st.markdown(f"""
                <div class="source-card">
                    <span class="source-badge">{src['domain']}</span>
                    <span class="source-badge badge-score">{score_str}</span>
                    <br/>
                    <a class="source-title" href="{src['url']}" target="_blank">🔗 {src['title']}</a>
                </div>
                """, unsafe_allow_html=True)
                with st.expander(f"Inspect Context Excerpt #{idx}"):
                    st.write(src["excerpt"])

    # ---------------------------------------------------------
    # Previous Research Queries in Session
    # ---------------------------------------------------------
    if len(st.session_state.history) > 1:
        st.markdown("---")
        st.markdown("### 📜 Session Research History")
        for prev_idx, item in enumerate(st.session_state.history[1:], start=1):
            with st.expander(f"Query: {item['query']} ({item['time']}s)"):
                st.markdown(item["answer"])
                st.caption(f"Referenced {len(item['sources'])} sources.")

# ---------------------------------------------------------
# Currently Indexed Articles Overview
# ---------------------------------------------------------
if st.session_state.indexed_articles:
    st.markdown("---")
    st.markdown("### 📑 Currently Ingested Articles in Memory")
    articles_table = []
    for art in st.session_state.indexed_articles:
        articles_table.append({
            "Article Title": art.get("title", "Untitled"),
            "Publisher / Domain": art.get("domain", "web"),
            "Source URL": art.get("source", "#"),
        })
    st.dataframe(articles_table, use_container_width=True)
