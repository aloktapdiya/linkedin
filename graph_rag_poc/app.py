from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Graph RAG",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Lazy imports (avoid crashing on missing optional deps at startup)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_config():
    from config import Config
    return Config()


def _get_pipeline(config):
    from src.pipeline import GraphRAGPipeline
    return GraphRAGPipeline(config)


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        "pipeline": None,
        "built": False,
        "graph_stats": {},
        "chat_history": [],
        "benchmark_results": None,
        "provider_name": "—",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🔗 Graph RAG")
    st.caption("Knowledge-graph-augmented retrieval")

    st.divider()

    # -- Provider status --
    st.subheader("Inference engine")
    config = _load_config()

    # Check Ollama live
    def _check_ollama() -> bool:
        try:
            import requests
            r = requests.get(
                f"{config.ollama_base_url.rstrip('/')}/api/tags", timeout=2
            )
            return r.status_code == 200
        except Exception:
            return False

    ollama_ok = _check_ollama()
    gemini_key = config.gemini_api_key or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    anthropic_key = config.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY", "")

    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("Ollama")
    with col2:
        if ollama_ok:
            st.success(f"✓ {config.ollama_model}", icon=None)
        else:
            st.warning("offline", icon=None)

    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("Gemini")
    with col2:
        if gemini_key:
            st.success(f"✓ {config.gemini_model}", icon=None)
        else:
            st.error("no key", icon=None)

    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("Claude")
    with col2:
        if anthropic_key:
            st.success(f"✓ {config.claude_model}", icon=None)
        else:
            st.error("no key", icon=None)

    if st.session_state.provider_name != "—":
        st.info(f"Active: **{st.session_state.provider_name}**")

    st.divider()

    # -- Index controls --
    st.subheader("Index")

    dataset_path = st.text_input(
        "Dataset (JSON)",
        value="data/benchmark/hotpotqa_sample.json",
        help="Path to a HotpotQA-style JSON file",
    )

    if st.button("⚙️ Build Index", use_container_width=True, type="primary"):
        if not os.path.exists(dataset_path):
            st.error(f"File not found: {dataset_path}")
        else:
            with st.spinner("Building knowledge graph…"):
                try:
                    pipeline = _get_pipeline(config)
                    with open(dataset_path) as f:
                        dataset = json.load(f)

                    # flatten to passages
                    passages: List[Dict[str, Any]] = []
                    seen: set = set()
                    for item in dataset:
                        for title, sentences in item.get("context", {}).items():
                            text = " ".join(sentences)
                            key = f"{title}::{text[:60]}"
                            if key not in seen:
                                seen.add(key)
                                passages.append(
                                    {
                                        "id": f"doc_{len(passages)}",
                                        "title": title,
                                        "text": text,
                                        "source": "dataset",
                                    }
                                )

                    pipeline.build(passages)

                    gb = pipeline._graph_builder
                    graph = gb.graph if hasattr(gb, "graph") else None
                    st.session_state.graph_stats = {
                        "passages": len(gb.get_passage_nodes()),
                        "entities": len(gb.get_entity_nodes()),
                        "nodes": graph.number_of_nodes() if graph else "?",
                        "edges": graph.number_of_edges() if graph else "?",
                    }
                    st.session_state.pipeline = pipeline
                    st.session_state.built = True
                    # capture which provider was used
                    try:
                        st.session_state.provider_name = pipeline.generator._get_provider().name
                    except Exception:
                        st.session_state.provider_name = "unknown"
                    st.success("Index built!")
                except RuntimeError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Build failed: {exc}")

    if st.session_state.built:
        gs = st.session_state.graph_stats
        st.markdown("**Graph stats**")
        mcol1, mcol2 = st.columns(2)
        mcol1.metric("Nodes", gs.get("nodes", "?"))
        mcol2.metric("Edges", gs.get("edges", "?"))
        mcol1.metric("Passages", gs.get("passages", "?"))
        mcol2.metric("Entities", gs.get("entities", "?"))

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------
tab_chat, tab_bench = st.tabs(["💬 Chat", "📊 Benchmark"])

# ===========================================================================
# TAB 1 — Chat
# ===========================================================================
with tab_chat:
    st.header("Ask the knowledge graph")

    if not st.session_state.built:
        st.info("Build the index first using the sidebar.")
    else:
        # Render chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("context"):
                    with st.expander(
                        f"Retrieved {len(msg['context'])} passage(s)", expanded=False
                    ):
                        for i, passage in enumerate(msg["context"], 1):
                            title = passage.get("title") or passage.get("id", f"passage-{i}")
                            text = passage.get("text", "")
                            st.markdown(f"**{i}. {title}**")
                            st.caption(text[:300] + ("…" if len(text) > 300 else ""))
                            if i < len(msg["context"]):
                                st.divider()

        # Chat input
        if question := st.chat_input("Ask a question about the indexed documents…"):
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        result = st.session_state.pipeline.query(question)
                        answer = result["answer"]
                        context = result.get("context", [])
                    except Exception as exc:
                        answer = f"Error: {exc}"
                        context = []

                st.markdown(answer)
                if context:
                    with st.expander(f"Retrieved {len(context)} passage(s)", expanded=False):
                        for i, passage in enumerate(context, 1):
                            title = passage.get("title") or passage.get("id", f"passage-{i}")
                            text = passage.get("text", "")
                            st.markdown(f"**{i}. {title}**")
                            st.caption(text[:300] + ("…" if len(text) > 300 else ""))
                            if i < len(context):
                                st.divider()

            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer, "context": context}
            )

        if st.session_state.chat_history:
            if st.button("🗑️ Clear chat", use_container_width=False):
                st.session_state.chat_history = []
                st.rerun()

# ===========================================================================
# TAB 2 — Benchmark
# ===========================================================================
with tab_bench:
    st.header("Benchmark evaluation")

    bcol1, bcol2 = st.columns([2, 1])
    with bcol1:
        bench_path = st.text_input(
            "Benchmark dataset (JSON)",
            value="data/benchmark/hotpotqa_sample.json",
            key="bench_path",
        )
    with bcol2:
        subset_size = st.slider("Questions", min_value=5, max_value=100, value=20, step=5)

    run_bench = st.button(
        "▶ Run Benchmark",
        use_container_width=True,
        type="primary",
        disabled=not os.path.exists(bench_path),
    )

    if run_bench:
        if not os.path.exists(bench_path):
            st.error(f"File not found: {bench_path}")
        else:
            progress_bar = st.progress(0, text="Loading dataset…")
            status_box = st.empty()
            results_placeholder = st.empty()

            try:
                from benchmark.evaluator import BenchmarkEvaluator

                with open(bench_path) as f:
                    dataset = json.load(f)
                dataset = dataset[:subset_size]

                # Build pipeline if not already built
                if not st.session_state.built:
                    progress_bar.progress(5, text="Building index…")
                    pipeline = _get_pipeline(config)
                    passages: List[Dict[str, Any]] = []
                    seen: set = set()
                    for item in dataset:
                        for title, sentences in item.get("context", {}).items():
                            text = " ".join(sentences)
                            key = f"{title}::{text[:60]}"
                            if key not in seen:
                                seen.add(key)
                                passages.append(
                                    {
                                        "id": f"doc_{len(passages)}",
                                        "title": title,
                                        "text": text,
                                        "source": "dataset",
                                    }
                                )
                    pipeline.build(passages)
                    st.session_state.pipeline = pipeline
                    st.session_state.built = True
                else:
                    pipeline = st.session_state.pipeline

                evaluator = BenchmarkEvaluator(config)
                raw_results: List[Dict[str, Any]] = []

                for idx, item in enumerate(dataset):
                    pct = int(10 + 88 * (idx / len(dataset)))
                    progress_bar.progress(pct, text=f"Question {idx + 1}/{len(dataset)}…")
                    out = pipeline.query(item["question"])
                    out["ground_truth"] = item["answer"]
                    raw_results.append(out)

                progress_bar.progress(99, text="Computing metrics…")
                evaluation = evaluator.evaluate(raw_results)
                st.session_state.benchmark_results = evaluation
                progress_bar.progress(100, text="Done!")

            except RuntimeError as exc:
                progress_bar.empty()
                st.error(str(exc))
                st.stop()
            except Exception as exc:
                progress_bar.empty()
                st.error(f"Benchmark failed: {exc}")
                st.stop()

    # Display stored results
    if st.session_state.benchmark_results:
        ev = st.session_state.benchmark_results
        summary = ev.get("summary", {})

        st.subheader("Summary")
        m1, m2, m3 = st.columns(3)
        m1.metric("Exact Match", f"{summary.get('exact_match', 0):.1f}%")
        m2.metric("F1 Score", f"{summary.get('f1', 0):.1f}%")
        m3.metric("Questions", summary.get("num_examples", len(ev.get("detailed", []))))

        # Per-example table
        rows = ev.get("detailed", [])
        if rows:
            import pandas as pd

            df = pd.DataFrame(
                [
                    {
                        "Question": r.get("question", "")[:80],
                        "Predicted": r.get("predicted", "")[:60],
                        "Ground Truth": r.get("ground_truth", "")[:60],
                        "EM": "✓" if r.get("exact_match") else "✗",
                        "F1": f"{r.get('f1', 0):.2f}",
                    }
                    for r in rows
                ]
            )
            st.subheader("Per-question results")
            st.dataframe(df, use_container_width=True, hide_index=True)

        # Save button
        results_dir = config.results_dir
        os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(results_dir, "benchmark_results.json")
        if st.button("💾 Save results to JSON"):
            with open(out_path, "w") as f:
                json.dump(ev, f, indent=2)
            st.success(f"Saved → {out_path}")
