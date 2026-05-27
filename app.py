import altair as alt
import streamlit as st

from rag_pipeline import answer_query, prepare_vector_store


st.set_page_config(
    page_title="Query Aware RAG Assistant",
    layout="wide",
)


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;500&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

        /* Sidebar collapse button — show >> when sidebar is open */
        [data-testid="stSidebarCollapseButton"] button::before {
            content: ">>";
            font-family: 'DM Mono', monospace;
            font-size: 0.85rem;
            color: #f0ece4;
        }
        [data-testid="stSidebarCollapseButton"] button svg,
        [data-testid="stSidebarCollapseButton"] button span {
            display: none !important;
        }

        /* Sidebar expand button (when sidebar is closed) — show << */
        [data-testid="stSidebarCollapsedControl"] button::before {
            content: "<<";
            font-family: 'DM Mono', monospace;
            font-size: 0.85rem;
            color: #f0ece4;
        }
        [data-testid="stSidebarCollapsedControl"] button svg,
        [data-testid="stSidebarCollapsedControl"] button span {
            display: none !important;
        }

        /* Upload — show a single custom-styled browse button */
        [data-testid="stFileUploaderDropzone"] {
            background: transparent !important;
            border: none !important;
            padding: 0 !important;
        }

        [data-testid="stFileUploaderDropzoneInstructions"],
        [data-testid="stFileUploaderDropzone"] small,
        [data-testid="stFileUploaderDropzone"] svg {
            display: none !important;
        }

        [data-testid="stFileUploader"] > label,
        div[data-testid="stFileUploaderFile"] {
            display: none !important;
        }

        [data-testid="stFileUploaderDropzone"] > div > button {
            background: #E05A4E !important;
            color: transparent !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.55rem 1.2rem !important;
            font-size: 0 !important;
            width: 100% !important;
            position: relative !important;
            cursor: pointer !important;
            min-height: 2.5rem !important;
        }

        [data-testid="stFileUploaderDropzone"] > div > button::after {
            content: "Upload Files";
            position: absolute !important;
            top: 50% !important;
            left: 50% !important;
            transform: translate(-50%, -50%) !important;
            font-size: 0.85rem !important;
            font-family: 'Outfit', sans-serif !important;
            font-weight: 500 !important;
            color: #ffffff !important;
            visibility: visible !important;
            white-space: nowrap !important;
        }

        [data-testid="stFileUploaderDropzone"] > div > button:hover {
            background: #c94e43 !important;
        }

        html, body, .stApp {
            font-family: 'Outfit', sans-serif !important;
        }

        .stApp {
            background: #0f0e0c;
            color: #f0ece4;
        }

        [data-testid="stSidebar"] {
            background: #131210;
            border-right: 1px solid rgba(255, 255, 255, 0.07);
        }

        [data-testid="stSidebar"] * {
            font-family: 'Outfit', sans-serif !important;
        }

        .material-symbols-rounded,
        .material-icons {
            font-family: 'Material Symbols Rounded', 'Material Icons' !important;
        }

        [data-testid="stHeader"] {
            background: rgba(15, 14, 12, 0.92);
            backdrop-filter: blur(8px);
        }

        .block-container {
            padding-top: 4.5rem;
            padding-bottom: 2.5rem;
            max-width: 1020px;
        }

        div[data-testid="stMetric"] {
            background: #1a1916;
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 10px;
            padding: 0.8rem 1rem;
            min-height: 82px;
        }

        div[data-testid="stMetricLabel"] {
            font-size: 0.7rem !important;
            color: #7a7268 !important;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        div[data-testid="stMetricValue"] {
            font-family: 'DM Mono', monospace !important;
            font-size: 1.25rem !important;
            color: #f0ece4 !important;
        }

        div[data-testid="stTextInputRootElement"] > div {
            background: #1a1916;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.09);
        }

        div[data-testid="stFileUploader"] section {
            background: #1a1916;
            border-radius: 10px;
        }

        div[data-testid="stExpander"] {
            background: #181714;
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 10px;
        }

        div[data-testid="stButton"] > button {
            background: #242220;
            color: #f0ece4;
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 10px;
            font-weight: 500;
            font-family: 'Outfit', sans-serif !important;
            transition: background 0.15s;
        }

        div[data-testid="stButton"] > button:hover {
            background: #2e2c28;
        }

        div[data-testid="stButton"] > button[kind="primary"] {
            background: #E05A4E;
            border-color: #E05A4E;
            color: white;
        }

        div[data-testid="stButton"] > button[kind="primary"]:hover {
            background: #c94e43;
        }

        div[data-testid="stButton"] button[kind="secondary"] {
            background: rgba(224, 90, 78, 0.15) !important;
            color: #E05A4E !important;
            border: 1px solid rgba(224, 90, 78, 0.3) !important;
            border-radius: 8px !important;
            padding: 0.3rem 0.5rem !important;
            font-size: 0.75rem !important;
            min-height: 0 !important;
            width: 100% !important;
        }

        div[data-testid="stButton"] button[kind="secondary"]:hover {
            background: rgba(224, 90, 78, 0.3) !important;
        }

        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #E05A4E, #f08078);
        }

        hr {
            border-color: rgba(255,255,255,0.07) !important;
        }

        .stCaption {
            color: #7a7268 !important;
        }

        .app-title {
            font-family: 'DM Serif Display', serif !important;
            font-size: 1.9rem;
            font-weight: 400;
            letter-spacing: -0.01em;
            color: #f0ece4;
            margin-bottom: 0;
        }

        .topline {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 0.8rem;
            margin-bottom: 1.25rem;
        }

        .live-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 0.28rem 0.7rem;
            border-radius: 999px;
            background: rgba(77, 184, 72, 0.12);
            color: #7ee87a;
            border: 1px solid rgba(77, 184, 72, 0.2);
            font-size: 0.74rem;
            font-weight: 500;
        }

        .live-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #7ee87a;
            display: inline-block;
        }

        .section-label {
            color: #6e6760;
            font-size: 0.7rem;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 500;
        }

        .panel {
            background: #181714;
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 12px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.75rem;
        }

        .verdict-pill {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.04em;
        }

        .verdict-supported {
            background: rgba(77, 184, 72, 0.15);
            color: #8af07a;
            border: 1px solid rgba(77, 184, 72, 0.2);
        }

        .verdict-partial {
            background: rgba(255, 184, 77, 0.14);
            color: #ffd089;
            border: 1px solid rgba(255, 184, 77, 0.2);
        }

        .verdict-unsupported, .verdict-unknown {
            background: rgba(224, 90, 78, 0.14);
            color: #ff9a9d;
            border: 1px solid rgba(224, 90, 78, 0.2);
        }

        .query-type-pill {
            display: inline-block;
            padding: 0.22rem 0.65rem;
            border-radius: 999px;
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            margin-left: 0.5rem;
        }

        .qtype-factual {
            background: rgba(79, 124, 255, 0.14);
            color: #7ea8ff;
            border: 1px solid rgba(79, 124, 255, 0.2);
        }

        .qtype-analytical {
            background: rgba(167, 110, 255, 0.14);
            color: #c99dff;
            border: 1px solid rgba(167, 110, 255, 0.2);
        }

        .qtype-multi_hop {
            background: rgba(255, 165, 60, 0.14);
            color: #ffbe6e;
            border: 1px solid rgba(255, 165, 60, 0.2);
        }

        .qtype-default {
            background: rgba(180, 180, 180, 0.12);
            color: #b0a89e;
            border: 1px solid rgba(180, 180, 180, 0.15);
        }

        .answer-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: #c8c0b5;
            font-size: 0.82rem;
            padding-bottom: 0.65rem;
            margin-bottom: 0.75rem;
            border-bottom: 1px solid rgba(255,255,255,0.07);
        }

        .answer-meta-left {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            flex-wrap: wrap;
        }

        .source-chip {
            display: inline-block;
            padding: 0.24rem 0.55rem;
            margin: 0.2rem 0.3rem 0 0;
            border-radius: 7px;
            background: #222019;
            border: 1px solid rgba(255,255,255,0.07);
            color: #e8e0d4;
            font-size: 0.72rem;
            font-family: 'DM Mono', monospace;
        }

        .source-sim {
            color: #E05A4E;
        }

        .citation-note {
            color: #7a7268;
            font-size: 0.78rem;
            margin-top: 0.6rem;
            font-style: italic;
        }

        .question-shell {
            background: #181714;
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 12px;
            padding: 0.65rem 0.75rem;
            margin-bottom: 1.1rem;
        }

        .perf-section-header {
            font-size: 0.72rem;
            font-weight: 600;
            color: #6e6760;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin: 1.2rem 0 0.6rem;
        }

        .graph-container {
            background: #181714;
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 12px;
            padding: 1rem 1rem 0.5rem;
            margin-top: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    defaults = {
        "store": None,
        "build_info": None,
        "history": [],
        "active_mode": "existing",
        "uploaded_files_list": [],
        "uploader_nonce": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_build_summary() -> None:
    build_info = st.session_state.get("build_info")
    if not build_info:
        st.info("Prepare a knowledge source from the sidebar to start asking questions.")
        return

    st.caption(
        f"Data source: {build_info.get('data_label', 'Unknown')} | "
        f"Status: {build_info.get('index_status', 'n/a')}"
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Documents", build_info.get("document_count", build_info.get("uploaded_document_count", 0)))
    col2.metric("Chunks in index", build_info.get("chunk_count", 0))
    col3.metric("Build time", f"{build_info.get('build_seconds', 0.0):.2f} s")
    col4.metric("Mode", build_info.get("mode", "-").title())


def _query_type_pill(query_type: str) -> str:
    qt = str(query_type).lower().replace("-", "_").strip()
    label_map = {
        "factual": "Factual",
        "analytical": "Analytical",
        "multi_hop": "Multi-hop",
    }
    label = label_map.get(qt, query_type.replace("_", " ").title() if query_type else "Unknown")
    css_class = f"qtype-{qt}" if qt in label_map else "qtype-default"
    return f'<span class="query-type-pill {css_class}">{label}</span>'


def render_result(result: dict, title: str = "Latest Answer") -> None:
    retrieval = result["retrieval"]
    generation = result["generation"]
    evaluation = generation["evaluation"]
    performance = generation.get("performance_metrics", {})
    evidence = generation.get("supporting_evidence", [])
    sources = result["sources"]

    query_type = retrieval.get("query_type") or result.get("query_type") or ""

    st.markdown(f'<div class="section-label">{title}</div>', unsafe_allow_html=True)

    metric_cols = st.columns(6)
    metric_cols[0].metric("Build time", _format_build_time())
    metric_cols[1].metric("Latency", f"{result['total_latency_ms']:.0f}ms")
    metric_cols[2].metric("Best Similarity", f"{max(retrieval['scores'], default=0.0):.3f}")
    metric_cols[3].metric("Mode", st.session_state.get("build_info", {}).get("mode", "-").title())
    metric_cols[4].metric("Top-k", retrieval["top_k"])
    metric_cols[5].metric("Confidence", f"{retrieval['confidence']:.3f}")

    answer_col, eval_col = st.columns([1.65, 0.95])
    verdict_class = _verdict_class(evaluation.get("verdict", "UNKNOWN"))

    with answer_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        primary_source = sources[0] if sources else "Unknown source"
        query_pill_html = _query_type_pill(query_type) if query_type else ""
        st.markdown(
            f'<div class="answer-meta">'
            f'<div class="answer-meta-left">'
            f'<span>Answer · {primary_source}</span>'
            f'{query_pill_html}'
            f'</div>'
            f'<span class="verdict-pill {verdict_class}">{evaluation["verdict"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.write(generation["answer"])

        source_comparison = sorted(
            generation["source_comparison"],
            key=lambda item: item["support_score"],
            reverse=True,
        )
        if source_comparison:
            chips = "".join(
                f'<span class="source-chip">Src {item["rank"]} <span class="source-sim">· {item["similarity"]:.3f}</span></span>'
                for item in source_comparison[:4]
            )
            st.markdown(chips, unsafe_allow_html=True)
        st.markdown(
            '<div class="citation-note">Inline citations map to the proof panels below.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with eval_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.72rem;font-weight:600;color:#6e6760;text-transform:uppercase;'
            'letter-spacing:0.08em;margin-bottom:0.6rem;">Trust metrics</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="verdict-pill {verdict_class}" style="margin-bottom:0.7rem;">'
            f'{evaluation["verdict"]}</div>',
            unsafe_allow_html=True,
        )
        rows = [
            ("LLM confidence", f"{evaluation['confidence']:.2f}"),
            ("Groundedness", f"{performance.get('groundedness_score', 0.0):.2f}"),
            ("Answer accuracy", f"{performance.get('answer_accuracy', 0.0):.2f}"),
            ("Hallucination risk", f"{performance.get('hallucination_risk', 0.0):.2f}"),
        ]
        for label, val in rows:
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:0.3rem 0;border-bottom:1px solid rgba(255,255,255,0.05);">'
                f'<span style="font-size:0.8rem;color:#9a9288;">{label}</span>'
                f'<span style="font-family:\'DM Mono\',monospace;font-size:0.82rem;color:#f0ece4;">{val}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<div style="font-size:0.76rem;color:#6e6760;margin-top:0.65rem;line-height:1.5;">'
            f'{evaluation["explanation"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="perf-section-header">Performance Summary</div>', unsafe_allow_html=True)
    perf_cols = st.columns(5)
    perf_cols[0].metric("Retrieval Confidence", f"{performance.get('retrieval_confidence', 0.0):.3f}")
    perf_cols[1].metric("Citation Coverage", f"{performance.get('citation_coverage', 0.0):.3f}")
    perf_cols[2].metric("Groundedness", f"{performance.get('groundedness_score', 0.0):.3f}")
    perf_cols[3].metric("Answer Accuracy", f"{performance.get('answer_accuracy', 0.0):.3f}")
    perf_cols[4].metric("Hallucination Risk", f"{performance.get('hallucination_risk', 0.0):.3f}")

    acc_val = float(performance.get("answer_accuracy", 0.0))
    gnd_val = float(performance.get("groundedness_score", 0.0))
    st.progress(acc_val, text=f"Answer accuracy  {acc_val:.0%}")
    st.progress(gnd_val, text=f"Groundedness  {gnd_val:.0%}")

    render_performance_graph(performance, retrieval, evaluation)

    if retrieval.get("query_expanded") and retrieval.get("expanded_query"):
        st.caption(f"Expanded query: {retrieval['expanded_query']}")

    if evidence:
        st.markdown('<div class="perf-section-header">Proof from retrieved sources</div>', unsafe_allow_html=True)
        for item in evidence:
            tags = ", ".join(item.get("matching_terms", [])) or "no strong lexical overlap"
            header = (
                f"Source {item['rank']} | {item['source']} | "
                f"support {item['support_score']:.3f} | similarity {item['similarity']:.3f}"
            )
            with st.expander(header, expanded=(item["rank"] == 1)):
                cited_color = "#7ee87a" if item["cited_in_answer"] else "#ff9a9d"
                st.markdown(
                    f'<span style="font-size:0.78rem;color:{cited_color};">'
                    f'{"✓ Cited in answer" if item["cited_in_answer"] else "✗ Not cited"}</span>',
                    unsafe_allow_html=True,
                )
                st.caption(f"Matching terms: {tags}")
                st.write(item["proof"])

    st.markdown('<div class="perf-section-header">Retrieved chunks</div>', unsafe_allow_html=True)
    for rank, (chunk, score) in enumerate(zip(retrieval["chunks"], retrieval["scores"]), start=1):
        header = f"{rank}. {chunk.get('source', 'Unknown source')} | similarity {score:.3f}"
        with st.expander(header, expanded=(rank == 1)):
            st.write(chunk["text"])

    st.markdown('<div class="perf-section-header">Answer and source comparison</div>', unsafe_allow_html=True)
    for item in generation["source_comparison"]:
        label = (
            f"Rank {item['rank']} | {item['source']} | "
            f"similarity {item['similarity']:.3f} | overlap {item['answer_overlap']:.3f} | "
            f"support {item['support_score']:.3f}"
        )
        with st.expander(label):
            cited_color = "#7ee87a" if item["cited_in_answer"] else "#ff9a9d"
            st.markdown(
                f'<span style="font-size:0.78rem;color:{cited_color};">'
                f'{"✓ Cited in answer" if item["cited_in_answer"] else "✗ Not cited"}</span>',
                unsafe_allow_html=True,
            )
            if item["matching_terms"]:
                st.caption("Matching terms: " + ", ".join(item["matching_terms"]))
            st.write(item["excerpt"])


def render_performance_graph(performance: dict, retrieval: dict, evaluation: dict) -> None:
    graph_data = [
        {"metric": "Retrieval confidence", "value": round(float(performance.get("retrieval_confidence", 0.0)), 4), "group": "Retrieval"},
        {"metric": "Citation coverage", "value": round(float(performance.get("citation_coverage", 0.0)), 4), "group": "Grounding"},
        {"metric": "Groundedness", "value": round(float(performance.get("groundedness_score", 0.0)), 4), "group": "Grounding"},
        {"metric": "Answer accuracy", "value": round(float(performance.get("answer_accuracy", 0.0)), 4), "group": "Answer"},
        {"metric": "Hallucination risk", "value": round(float(performance.get("hallucination_risk", 0.0)), 4), "group": "Risk"},
        {"metric": "LLM confidence", "value": round(float(evaluation.get("confidence", 0.0)), 4), "group": "Answer"},
        {"metric": "Retrieval mean", "value": round(float(retrieval.get("confidence", 0.0)), 4), "group": "Retrieval"},
    ]

    color_scale = alt.Scale(
        domain=["Retrieval", "Grounding", "Answer", "Risk"],
        range=["#4f7cff", "#4fd98b", "#E05A4E", "#ff6d4d"],
    )

    bars = (
        alt.Chart(alt.Data(values=graph_data))
        .mark_bar(
            cornerRadiusTopLeft=5,
            cornerRadiusTopRight=5,
            size=22,
        )
        .encode(
            x=alt.X(
                "metric:N",
                sort=None,
                axis=alt.Axis(
                    labelAngle=-25,
                    labelColor="#9a9288",
                    labelFontSize=11,
                    labelFont="Outfit",
                    title=None,
                    tickColor="transparent",
                    domainColor="rgba(255,255,255,0.1)",
                ),
            ),
            y=alt.Y(
                "value:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(
                    format=".0%",
                    labelColor="#9a9288",
                    labelFontSize=11,
                    labelFont="DM Mono",
                    title=None,
                    gridColor="rgba(255,255,255,0.06)",
                    gridDash=[3, 3],
                    tickColor="transparent",
                    domainColor="transparent",
                ),
            ),
            color=alt.Color(
                "group:N",
                scale=color_scale,
                legend=alt.Legend(
                    title=None,
                    labelColor="#9a9288",
                    labelFontSize=11,
                    labelFont="Outfit",
                    orient="top-right",
                    symbolType="circle",
                    symbolSize=80,
                ),
            ),
            tooltip=[
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("value:Q", title="Score", format=".3f"),
                alt.Tooltip("group:N", title="Category"),
            ],
        )
        .properties(height=230, title="")
    )

    text = bars.mark_text(
        align="center",
        baseline="bottom",
        dy=-4,
        fontSize=10,
        font="DM Mono",
        color="#c8c0b5",
    ).encode(text=alt.Text("value:Q", format=".2f"))

    chart = (
        (bars + text)
        .configure_view(
            strokeOpacity=0,
            fill="transparent",
        )
        .configure(
            background="transparent",
            font="Outfit",
        )
        .configure_axis(
            gridColor="rgba(255,255,255,0.06)",
        )
    )

    st.markdown('<div class="perf-section-header">Performance graph</div>', unsafe_allow_html=True)
    st.markdown('<div class="graph-container">', unsafe_allow_html=True)
    st.altair_chart(chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _verdict_class(verdict: str) -> str:
    normalized = str(verdict).upper()
    if normalized == "SUPPORTED":
        return "verdict-supported"
    if normalized == "PARTIALLY SUPPORTED":
        return "verdict-partial"
    if normalized in ("NOT SUPPORTED", "UNSUPPORTED"):
        return "verdict-unsupported"
    return "verdict-unknown"


def _format_build_time() -> str:
    build_seconds = float(st.session_state.get("build_info", {}).get("build_seconds", 0.0) or 0.0)
    if build_seconds <= 0:
        return "-"
    if build_seconds < 1:
        return f"{build_seconds * 1000:.0f}ms"
    return f"{build_seconds:.2f}s"


def render_uploaded_file_item(file, index: int) -> None:
    col1, col2 = st.columns([5, 1], vertical_alignment="center")
    with col1:
        st.markdown(
            f'<div style="background:#1a1916; border:1px solid rgba(255,255,255,0.07); '
            f'border-radius:8px; padding:0.4rem 0.7rem; font-size:0.78rem; '
            f'color:#f0ece4; font-family:DM Mono,monospace; overflow:hidden; '
            f'text-overflow:ellipsis; white-space:nowrap;">'
            f'📄 {file.name}</div>',
            unsafe_allow_html=True,
        )
    with col2:
        if st.button("✕", key=f"remove_{index}_{file.name}", help="Remove file"):
            st.session_state.uploaded_files_list.pop(index)
            st.session_state.uploader_nonce += 1
            st.rerun()


init_state()
inject_theme()

build_info = st.session_state.get("build_info") or {}
live_chunks = build_info.get("chunk_count", 0)
st.markdown(
    f'<div class="topline">'
    f'<div class="app-title">RAG Assistant</div>'
    f'<div class="live-pill"><span class="live-dot"></span>Live · {live_chunks} chunks</div>'
    f'</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="section-label">Knowledge Source</div>', unsafe_allow_html=True)
    data_mode = st.radio(
        "Choose data source",
        options=["existing", "upload"],
        format_func=lambda v: "Use existing indexed data" if v == "existing" else "Upload new documents",
        index=0 if st.session_state.active_mode == "existing" else 1,
        label_visibility="collapsed",
    )

    append_uploaded = False
    if data_mode == "upload":
        new_files = st.file_uploader(
            "Upload PDF, TXT, or DOCX files",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=f"upload_files_{st.session_state.uploader_nonce}",
        )

        # Add only new files that are not already in the list
        if new_files:
            existing_names = [f.name for f in st.session_state.uploaded_files_list]
            for f in new_files:
                if f.name not in existing_names:
                    st.session_state.uploaded_files_list.append(f)

        # Render uploaded files below the uploader, separate from the widget itself.
        for i, file in enumerate(st.session_state.uploaded_files_list):
            render_uploaded_file_item(file, i)

        append_uploaded = st.checkbox(
            "Append uploaded documents to the saved index", value=False
        )

    st.divider()
    st.markdown('<div class="section-label">Retrieval</div>', unsafe_allow_html=True)
    top_k = st.slider("Top-k chunks", min_value=1, max_value=10, value=5)
    use_expansion = st.checkbox("Use query expansion", value=True)

    if st.button("Prepare Source", use_container_width=True):
        with st.spinner("Preparing vector store…"):
            try:
                store, build_info = prepare_vector_store(
                    data_mode=data_mode,
                    uploaded_files=st.session_state.get("uploaded_files_list", []),
                    append_uploaded=append_uploaded,
                )
                st.session_state.store = store
                st.session_state.build_info = build_info
                st.session_state.active_mode = data_mode
                st.success("Knowledge source ready.")
            except Exception as exc:
                st.session_state.store = None
                st.session_state.build_info = None
                st.error(str(exc))

    if st.button("Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.uploaded_files_list = []
        st.session_state.uploader_nonce += 1
        st.rerun()

render_build_summary()

st.markdown('<div class="section-label">Question</div>', unsafe_allow_html=True)
st.markdown('<div class="question-shell">', unsafe_allow_html=True)
q_col, ask_col = st.columns([5.5, 1])
with q_col:
    question = st.text_input(
        "Question",
        placeholder="Ask anything about your documents…",
        label_visibility="collapsed",
    )
with ask_col:
    ask = st.button("Ask", type="primary", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

if ask:
    if not question.strip():
        st.warning("Enter a question first.")
    elif st.session_state.store is None:
        st.warning("Prepare a knowledge source before asking a question.")
    else:
        try:
            with st.status("Running RAG pipeline…", expanded=True) as status:
                st.write("Classifying query type and preparing retrieval settings…")
                st.write("Searching the vector index for relevant chunks…")
                result = answer_query(
                    query=question.strip(),
                    store=st.session_state.store,
                    top_k=top_k,
                    use_expansion=use_expansion,
                    api_key=None,
                )
                st.write("Generating answer with inline citations…")
                st.write("Calculating groundedness, confidence, and hallucination risk…")
                status.update(label="Answer ready", state="complete", expanded=False)
            st.session_state.history.insert(0, result)
        except Exception as exc:
            st.error(str(exc))

if st.session_state.history:
    render_result(st.session_state.history[0])

    if len(st.session_state.history) > 1:
        st.markdown('<div class="perf-section-header">Recent questions</div>', unsafe_allow_html=True)
        for item in st.session_state.history[1:]:
            with st.expander(item["query"]):
                render_result(item, title="Stored Answer")
