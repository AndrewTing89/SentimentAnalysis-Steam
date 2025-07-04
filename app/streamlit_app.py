import os
import joblib
import streamlit as st
from google.cloud import aiplatform, bigquery
import subprocess, pathlib, tempfile
import pandas as pd

# ── CONFIG via env-vars ───────────────────────────────────────────────────────
PROJECT   = os.getenv("PROJECT_ID", "sentiment-analysis-steam")
REGION    = os.getenv("REGION",     "us-central1")
EP_BERT   = os.getenv("ENDPOINT_ID_DISTILBERT")
BUNDLE    = os.getenv(
    "LOGREG_BUNDLE_PATH",
    "models/best_tfidf_lr_negRecall_20250630-050145.joblib.gz"
)
BQ_TABLE  = "sentiment-analysis-steam.steam_reviews.top10-owned-steamcommunity"

# ── DistilBERT (Vertex endpoint) ─────────────────────────────────────────────
def bert_predict(text: str):
    if not EP_BERT:
        return {"error": "ENDPOINT_ID_DISTILBERT not set"}
    aiplatform.init(project=PROJECT, location=REGION)
    endpoint = aiplatform.Endpoint(EP_BERT)
    response = endpoint.predict(instances=[{"text": text}])
    return response.predictions[0]

# ── Log-Reg helper (local, auto-downloads from GCS if needed) ────────────────
_loaded = None
def _ensure_local(path_or_gs: str) -> str:
    if path_or_gs.startswith("gs://"):
        local = pathlib.Path(tempfile.gettempdir()) / pathlib.Path(path_or_gs).name
        if not local.exists():
            subprocess.check_call(["gsutil", "cp", path_or_gs, str(local)])
        return str(local)
    return path_or_gs

def logreg_predict(text: str):
    global _loaded
    if _loaded is None:
        vec, clf = joblib.load(_ensure_local(BUNDLE))
        _loaded = (vec, clf)
    vec, clf = _loaded
    p = clf.predict_proba(vec.transform([text]))[0]  # [neg, pos]
    return {"label": "POSITIVE" if p[1]>=.5 else "NEGATIVE", "score": float(p[1])}

# ── BigQuery helper ───────────────────────────────────────────────────────────
def run_bigquery(sql: str) -> pd.DataFrame:
    client = bigquery.Client(project=PROJECT)
    job    = client.query(sql)
    return job.to_dataframe()

# ── Streamlit UI ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Steam Sentiment", layout="wide")
mode = st.sidebar.radio("Mode", ["Classification", "Dashboard"])

if mode == "Classification":
    st.title("🎮 Steam Review Sentiment Demo")
    txt = st.text_area("Paste a review ↓", height=160)
    if st.button("Classify") and txt.strip():
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("DistilBERT (Vertex)")
            out = bert_predict(txt)
            st.write(
                out if "error" in out
                else f"**{out['label']}** · {out['score']:.2%}"
            )
        with col2:
            st.subheader("Log-Reg (local)")
            out = logreg_predict(txt)
            st.write(
                out if "error" in out
                else f"**{out['label']}** · {out['score']:.2%}"
            )

else:
    st.title("📊 Steam Reviews Dashboard")
    # 1) get list of games
    df_games = run_bigquery(f"""
      SELECT DISTINCT game_name
      FROM `{BQ_TABLE}`
      ORDER BY game_name
    """)
    all_games = df_games["game_name"].tolist()
    selected = st.multiselect(
        "Select games to compare",
        all_games,
        default=all_games[:3]
    )

    if selected:
        # safely escape single quotes and build IN list
        quoted = ",".join(
            "'" + g.replace("'", "\\'") + "'" for g in selected
        )
        df = run_bigquery(f"""
          SELECT
            game_name,
            COUNTIF(voted_up) AS positive_count,
            COUNT(*) - COUNTIF(voted_up) AS negative_count
          FROM `{BQ_TABLE}`
          WHERE game_name IN ({quoted})
          GROUP BY game_name
          ORDER BY game_name
        """)
        # compute percentages
        df["total"]    = df["positive_count"] + df["negative_count"]
        df["pct_pos"] = (df["positive_count"] / df["total"] * 100).round(1)
        df["pct_neg"] = (df["negative_count"] / df["total"] * 100).round(1)

        st.subheader("Sentiment Breakdown")
        st.dataframe(
            df[["game_name","pct_pos","pct_neg"]],
            use_container_width=True
        )
        st.bar_chart(
            df.set_index("game_name")[["pct_pos","pct_neg"]],
            height=400
        )
    else:
        st.info("Pick at least one game to show its sentiment.")