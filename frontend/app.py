"""
frontend/app.py
---------------
Streamlit dashboard for the AI Model Evaluation Platform.

Pages (via sidebar navigation):
  1. Upload Dataset    – upload a CSV file
  2. Run Evaluation    – select a dataset + model and launch evaluation
  3. Results & Metrics – view past evaluation runs and comparison charts

Run with:
    streamlit run frontend/app.py
"""

import os
import io
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AI Model Evaluation Platform",
    page_icon="🤖",
    layout="wide",
)

# ─── Utility helpers ──────────────────────────────────────────────────────────

def api_get(path: str) -> list | dict | None:
    """GET request to backend; returns parsed JSON or None on error."""
    try:
        resp = requests.get(f"{BACKEND_URL}{path}", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach the backend. Is the FastAPI server running?")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post_json(path: str, payload: dict) -> dict | None:
    """POST JSON body to backend; returns parsed response or None on error."""
    try:
        resp = requests.post(
            f"{BACKEND_URL}{path}",
            json=payload,
            timeout=300,   # evaluations can take a few minutes
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach the backend. Is the FastAPI server running?")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post_file(path: str, file_bytes: bytes, filename: str, name: str) -> dict | None:
    """POST multipart/form-data (file upload) to backend."""
    try:
        resp = requests.post(
            f"{BACKEND_URL}{path}",
            files={"file": (filename, file_bytes, "text/csv")},
            data={"name": name},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach the backend. Is the FastAPI server running?")
        return None
    except Exception as e:
        st.error(f"Upload error: {e}")
        return None


# ─── Sidebar navigation ───────────────────────────────────────────────────────

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/4616/4616734.png",
    width=80,
)
st.sidebar.title("AI Eval Platform")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📂 Upload Dataset", "🚀 Run Evaluation", "📊 Results & Metrics"],
)

st.sidebar.markdown("---")
st.sidebar.caption("Powered by FastAPI + Streamlit")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Home
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Home":
    st.title("🤖 AI Model Evaluation Platform")
    st.markdown(
        """
        Welcome! This platform lets you:

        | Step | Action |
        |------|--------|
        | 1️⃣  | **Upload** a CSV dataset (columns: `question`, `ground_truth_answer`) |
        | 2️⃣  | **Select** an LLM (Llama, Mistral, Gemma …) |
        | 3️⃣  | **Run** an automated evaluation |
        | 4️⃣  | **Compare** model performance via BLEU, ROUGE, Accuracy & Latency |
        """
    )

    # Backend health check
    st.markdown("### 🔌 Backend Status")
    health = api_get("/health")
    if health and health.get("status") == "ok":
        st.success("✅ Backend is online")
    else:
        st.error("❌ Backend is offline")

    # Quick stats
    st.markdown("### 📈 Quick Stats")
    col1, col2, col3 = st.columns(3)

    with col1:
        models = api_get("/models") or []
        st.metric("Available Models", len(models))

    with col2:
        datasets = api_get("/datasets") or []
        st.metric("Uploaded Datasets", len(datasets))

    with col3:
        results = api_get("/evaluation-results") or []
        st.metric("Evaluation Runs", len(results))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Upload Dataset
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📂 Upload Dataset":
    st.title("📂 Upload Evaluation Dataset")
    st.markdown(
        "Upload a **CSV file** with exactly two columns: "
        "`question` and `ground_truth_answer`."
    )

    # Sample CSV download
    sample_csv = "question,ground_truth_answer\n" \
                 "What is the capital of France?,Paris\n" \
                 "Who wrote Romeo and Juliet?,William Shakespeare\n" \
                 "What is 2 + 2?,4\n"
    st.download_button(
        "⬇️ Download sample CSV",
        data=sample_csv,
        file_name="sample_dataset.csv",
        mime="text/csv",
    )

    st.markdown("---")

    with st.form("upload_form"):
        dataset_name = st.text_input("Dataset Name", placeholder="e.g. General Knowledge QA")
        uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])
        submitted = st.form_submit_button("📤 Upload Dataset")

    if submitted:
        if not dataset_name:
            st.warning("Please enter a dataset name.")
        elif uploaded_file is None:
            st.warning("Please select a CSV file.")
        else:
            # Preview
            df_preview = pd.read_csv(io.BytesIO(uploaded_file.read()))
            uploaded_file.seek(0)

            st.markdown("**Preview (first 5 rows):**")
            st.dataframe(df_preview.head(5), use_container_width=True)

            with st.spinner("Uploading …"):
                result = api_post_file(
                    "/upload-dataset",
                    file_bytes=uploaded_file.read(),
                    filename=uploaded_file.name,
                    name=dataset_name,
                )

            if result:
                st.success(
                    f"✅ Dataset **{result['name']}** uploaded successfully! "
                    f"({result['row_count']} rows, ID: {result['id']})"
                )

    # Show existing datasets
    st.markdown("---")
    st.markdown("### 📋 Existing Datasets")
    datasets = api_get("/datasets")
    if datasets:
        df_ds = pd.DataFrame(datasets)
        st.dataframe(df_ds, use_container_width=True)
    else:
        st.info("No datasets uploaded yet.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Run Evaluation
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🚀 Run Evaluation":
    st.title("🚀 Run Model Evaluation")

    datasets = api_get("/datasets") or []
    models   = api_get("/models")   or []

    if not datasets:
        st.warning("No datasets found. Please upload a dataset first.")
    elif not models:
        st.warning("No models found. Check that the database was seeded correctly.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            dataset_options = {f"{d['name']} (ID {d['id']}, {d['row_count']} rows)": d["id"]
                               for d in datasets}
            selected_dataset_label = st.selectbox("Select Dataset", list(dataset_options.keys()))
            selected_dataset_id    = dataset_options[selected_dataset_label]

        with col2:
            model_options = {f"{m['name']} [{m['provider']}]": m["id"] for m in models}
            selected_model_label = st.selectbox("Select Model", list(model_options.keys()))
            selected_model_id    = model_options[selected_model_label]

        st.markdown("---")
        st.info(
            "💡 The evaluation is **synchronous** – the page will wait until all "
            "questions have been answered by the LLM. For large datasets this may "
            "take several minutes."
        )

        if st.button("▶️ Start Evaluation", type="primary"):
            with st.spinner("Running evaluation … please wait"):
                result = api_post_json(
                    "/evaluate-model",
                    {"dataset_id": selected_dataset_id, "model_id": selected_model_id},
                )

            if result:
                st.success(f"✅ Evaluation complete! (ID: {result['evaluation_id']})")

                metrics = result.get("metrics", {})
                if metrics:
                    st.markdown("### 📊 Evaluation Metrics")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Accuracy",       f"{metrics['accuracy']*100:.1f}%")
                    m2.metric("BLEU",           f"{metrics['bleu_score']:.4f}")
                    m3.metric("ROUGE-L",        f"{metrics['rougeL']:.4f}")
                    m4.metric("Avg Latency",    f"{metrics['avg_latency_ms']:.0f} ms")

                # Question-level results
                q_results = result.get("question_results", [])
                if q_results:
                    st.markdown("### 🔍 Per-Question Results")
                    df_q = pd.DataFrame(q_results)
                    st.dataframe(df_q, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Results & Metrics
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Results & Metrics":
    st.title("📊 Evaluation Results & Model Comparison")

    results = api_get("/evaluation-results")

    if not results:
        st.info("No evaluation runs found. Run an evaluation first.")
    else:
        df = pd.DataFrame(results)

        # ── Summary table ─────────────────────────────────────────────────────
        st.markdown("### 📋 All Evaluation Runs")
        display_cols = [
            "evaluation_id", "model_name", "dataset_name",
            "status", "total_questions",
            "accuracy", "bleu_score", "rouge1", "rouge2", "rougeL", "avg_latency_ms",
            "started_at",
        ]
        # Only show columns that actually exist
        display_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[display_cols], use_container_width=True)

        # ── Filter to completed runs with metrics ──────────────────────────────
        df_done = df[df["status"] == "done"].dropna(subset=["accuracy"])

        if df_done.empty:
            st.warning("No completed runs with metrics yet.")
        else:
            st.markdown("---")
            st.markdown("### 📈 Model Comparison Charts")

            # Aggregate by model (mean across datasets)
            agg = df_done.groupby("model_name", as_index=False).agg(
                accuracy=("accuracy", "mean"),
                bleu_score=("bleu_score", "mean"),
                rouge1=("rouge1", "mean"),
                rouge2=("rouge2", "mean"),
                rougeL=("rougeL", "mean"),
                avg_latency_ms=("avg_latency_ms", "mean"),
                runs=("evaluation_id", "count"),
            )

            tab1, tab2, tab3 = st.tabs(["Accuracy & BLEU", "ROUGE Scores", "Latency"])

            # ── Tab 1: Accuracy & BLEU ────────────────────────────────────────
            with tab1:
                col1, col2 = st.columns(2)

                with col1:
                    fig_acc = px.bar(
                        agg,
                        x="model_name",
                        y="accuracy",
                        color="model_name",
                        title="Accuracy by Model",
                        labels={"accuracy": "Accuracy (0–1)", "model_name": "Model"},
                        color_discrete_sequence=px.colors.qualitative.Bold,
                    )
                    fig_acc.update_layout(showlegend=False, yaxis_range=[0, 1])
                    st.plotly_chart(fig_acc, use_container_width=True)

                with col2:
                    fig_bleu = px.bar(
                        agg,
                        x="model_name",
                        y="bleu_score",
                        color="model_name",
                        title="BLEU Score by Model",
                        labels={"bleu_score": "BLEU (0–1)", "model_name": "Model"},
                        color_discrete_sequence=px.colors.qualitative.Pastel,
                    )
                    fig_bleu.update_layout(showlegend=False, yaxis_range=[0, 1])
                    st.plotly_chart(fig_bleu, use_container_width=True)

            # ── Tab 2: ROUGE ──────────────────────────────────────────────────
            with tab2:
                rouge_df = agg.melt(
                    id_vars="model_name",
                    value_vars=["rouge1", "rouge2", "rougeL"],
                    var_name="metric",
                    value_name="score",
                )
                fig_rouge = px.bar(
                    rouge_df,
                    x="model_name",
                    y="score",
                    color="metric",
                    barmode="group",
                    title="ROUGE Scores by Model",
                    labels={"score": "Score (0–1)", "model_name": "Model", "metric": "Metric"},
                    color_discrete_sequence=["#636EFA", "#EF553B", "#00CC96"],
                )
                fig_rouge.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(fig_rouge, use_container_width=True)

            # ── Tab 3: Latency ────────────────────────────────────────────────
            with tab3:
                fig_lat = px.bar(
                    agg,
                    x="model_name",
                    y="avg_latency_ms",
                    color="model_name",
                    title="Average Latency by Model (lower is better)",
                    labels={"avg_latency_ms": "Avg Latency (ms)", "model_name": "Model"},
                    color_discrete_sequence=px.colors.qualitative.Safe,
                )
                fig_lat.update_layout(showlegend=False)
                st.plotly_chart(fig_lat, use_container_width=True)

            # ── Radar chart – normalised multi-metric comparison ───────────────
            st.markdown("### 🕸️ Multi-Metric Radar Chart")

            metrics_for_radar = ["accuracy", "bleu_score", "rouge1", "rougeL"]
            fig_radar = go.Figure()

            for _, model_row in agg.iterrows():
                values = [model_row[m] for m in metrics_for_radar]
                values.append(values[0])  # close the polygon

                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metrics_for_radar + [metrics_for_radar[0]],
                    fill="toself",
                    name=model_row["model_name"],
                ))

            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="Model Performance Radar",
                showlegend=True,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # ── Raw data table ────────────────────────────────────────────────
            st.markdown("### 📐 Aggregated Metrics Table")
            st.dataframe(
                agg.style.format({
                    "accuracy":       "{:.3f}",
                    "bleu_score":     "{:.4f}",
                    "rouge1":         "{:.4f}",
                    "rouge2":         "{:.4f}",
                    "rougeL":         "{:.4f}",
                    "avg_latency_ms": "{:.1f} ms",
                }),
                use_container_width=True,
            )
