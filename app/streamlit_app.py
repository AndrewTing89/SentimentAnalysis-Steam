import os
import joblib
import streamlit as st
from google.cloud import aiplatform, bigquery
import subprocess, pathlib, tempfile
import pandas as pd

# ── CONFIG via env-vars ───────────────────────────────────────────────────────
PROJECT = os.getenv("PROJECT_ID", "sentiment-analysis-steam")
REGION  = os.getenv("REGION",     "us-central1")
EP_BERT = os.getenv("ENDPOINT_ID_DISTILBERT")
BUNDLE  = os.getenv(
    "LOGREG_BUNDLE_PATH",
    "models/best_tfidf_lr_negRecall_20250630-050145.joblib.gz"
)

# ── TAB LAYOUT ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Steam Sentiment", layout="wide")
tab1, tab2 = st.tabs(["🕹️ Classifier", "📊 Dashboard"])

# ── TAB 1: CLASSIFIER ─────────────────────────────────────────────────────────
with tab1:
    st.title("🎮 Steam Review Sentiment Demo")

    txt = st.text_area("Paste a review ↓", height=160)
    if st.button("Classify") and txt.strip():
        col1, col2 = st.columns(2)

        # DistilBERT QA
        with col1:
            st.subheader("DistilBERT (Vertex)")
            def bert_predict(text):
                if not EP_BERT:
                    return {"error": "ENDPOINT_ID_DISTILBERT not set"}
                aiplatform.init(project=PROJECT, location=REGION)
                ep = aiplatform.Endpoint(EP_BERT)
                return ep.predict(instances=[{"text": text}]).predictions[0]
            out = bert_predict(txt)
            st.write(out if "error" in out else
                     f"**{out['label']}** · {out['score']:.2%}")

        # Log-Reg QA
        with col2:
            st.subheader("Log-Reg (local)")
            _loaded = None
            def _ensure_local(path_or_gs):
                if path_or_gs.startswith("gs://"):
                    local = pathlib.Path(tempfile.gettempdir()) / pathlib.Path(path_or_gs).name
                    if not local.exists():
                        subprocess.check_call(["gsutil", "cp", path_or_gs, str(local)])
                    return str(local)
                return path_or_gs

            def logreg_predict(text):
                nonlocal _loaded
                if _loaded is None:
                    vec, clf = joblib.load(_ensure_local(BUNDLE))
                    _loaded = (vec, clf)
                vec, clf = _loaded
                p = clf.predict_proba(vec.transform([text]))[0]
                return {"label": "POSITIVE" if p[1]>=.5 else "NEGATIVE", "score": float(p[1])}

            out = logreg_predict(txt)
            st.write(out if "error" in out else
                     f"**{out['label']}** · {out['score']:.2%}")

# ── TAB 2: DASHBOARD ──────────────────────────────────────────────────────────
with tab2:
    st.header("📊 Review Sentiment Dashboard")

    # 1) Game selector
    GAMES = [
        "Counter-Strike: Global Offensive",
        "Apex Legends",
        "PUBG: BATTLEGROUNDS",
        # … your other titles …
    ]
    game = st.selectbox("Choose a game", GAMES)

    # 2) Query BQ
    client = bigquery.Client(project=PROJECT)
    sql = """
      SELECT
        SUM(CASE WHEN voted_up THEN 1 ELSE 0 END) AS positives,
        SUM(CASE WHEN NOT voted_up THEN 1 ELSE 0 END) AS negatives
      FROM `{project}.steam_reviews.top10_owned_steamcommunity`
      WHERE game_name = @game
    """.format(project=PROJECT)
    job = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("game", "STRING", game)
            ]
        )
    )
    df = job.to_dataframe()

    # 3) Render
    if df.empty:
        st.warning(f"No data for {game}.")
    else:
        counts = {"Positive": int(df.loc[0, "positives"]),
                  "Negative": int(df.loc[0, "negatives"])}
        chart_df = pd.DataFrame.from_dict(counts, orient="index", columns=["Count"])
        st.bar_chart(chart_df)