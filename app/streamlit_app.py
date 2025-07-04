import os
import joblib
import streamlit as st
from google.cloud import aiplatform, bigquery
import subprocess, pathlib, tempfile
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
PROJECT   = os.getenv("PROJECT_ID",           "sentiment-analysis-steam")
REGION    = os.getenv("REGION",               "us-central1")
EP_BERT   = os.getenv("ENDPOINT_ID_DISTILBERT")
BUNDLE    = os.getenv("LOGREG_BUNDLE_PATH",   "models/best_tfidf_lr_negRecall_20250630-050145.joblib.gz")
BQ_TABLE  = "sentiment-analysis-steam.steam_reviews.top10-owned-steamcommunity"

# ── DistilBERT inference ─────────────────────────────────────────────────────
def bert_predict(text: str):
    if not EP_BERT:
        return {"error": "ENDPOINT_ID_DISTILBERT not set"}
    aiplatform.init(project=PROJECT, location=REGION)
    endpoint = aiplatform.Endpoint(EP_BERT)
    return endpoint.predict(instances=[{"text": text}]).predictions[0]

# ── LogReg inference (cached) ────────────────────────────────────────────────
_logreg = None
def logreg_predict(text: str):
    global _logreg
    if _logreg is None:
        # download from GCS if needed
        path = pathlib.Path(tempfile.gettempdir()) / pathlib.Path(BUNDLE).name
        if str(BUNDLE).startswith("gs://") and not path.exists():
            subprocess.check_call(["gsutil", "cp", BUNDLE, str(path)])
        vec, clf = joblib.load(path if path.exists() else BUNDLE)
        _logreg = (vec, clf)
    vec, clf = _logreg
    prob = clf.predict_proba(vec.transform([text]))[0][1]
    return {"label": "POSITIVE" if prob >= 0.5 else "NEGATIVE", "score": prob}

# ── BigQuery helper ───────────────────────────────────────────────────────────
def run_bigquery(query: str) -> pd.DataFrame:
    client = bigquery.Client(project=PROJECT)
    return client.query(query).to_dataframe()

# ── UI ─────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Steam Sentiment", layout="wide")
mode = st.sidebar.radio("Choose mode", ["Classify", "Dashboard"])

if mode == "Classify":
    st.header("🎮 Classify a Steam Review")
    txt = st.text_area("Paste your review here:", height=150)
    if st.button("Run"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("DistilBERT")
            out = bert_predict(txt)
            if "error" in out:
                st.error(out["error"])
            else:
                st.write(f"**{out['label']}** ({out['score']:.1%})")
        with col2:
            st.subheader("LogReg")
            out = logreg_predict(txt)
            st.write(f"**{out['label']}** ({out['score']:.1%})")

else:
    st.header("📊 Sentiment Dashboard")
    # pull distinct games
    df_games = run_bigquery(f"""
      SELECT DISTINCT game_name
      FROM `{BQ_TABLE}`
      ORDER BY game_name
    """)
    choices = df_games["game_name"].tolist()
    selected = st.multiselect("Pick games:", choices, default=choices[:3])

    if selected:
        # build safe IN list
        safe = ",".join("'" + g.replace("'", "''") + "'" for g in selected)
        df = run_bigquery(f"""
          SELECT
            game_name,
            COUNTIF(voted_up)   AS positives,
            COUNT(*) - COUNTIF(voted_up) AS negatives
          FROM `{BQ_TABLE}`
          WHERE game_name IN ({safe})
          GROUP BY game_name
        """)
        df["total"] = df["positives"] + df["negatives"]
        df["pct_pos"] = (df["positives"]/df["total"]*100).round(1)
        df["pct_neg"] = (df["negatives"]/df["total"]*100).round(1)

        st.dataframe(df[["game_name","pct_pos","pct_neg"]])
        st.bar_chart(df.set_index("game_name")[["pct_pos","pct_neg"]])

    else:
        st.info("Select at least one game above.")
#