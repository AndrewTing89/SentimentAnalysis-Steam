import os
import joblib
import streamlit as st
from google.cloud import aiplatform
import subprocess, pathlib, tempfile

# ── CONFIG via env-vars ───────────────────────────────────────────────────────
PROJECT  = os.getenv("PROJECT_ID", "sentiment-analysis-steam")
REGION   = os.getenv("REGION",     "us-central1")
EP_BERT  = os.getenv("ENDPOINT_ID_DISTILBERT")
BUNDLE   = os.getenv(
    "LOGREG_BUNDLE_PATH",
    "models/best_tfidf_lr_negRecall_20250630-050145.joblib.gz"
)

# ── DistilBERT (Vertex endpoint) ─────────────────────────────────────────────
def bert_predict(text: str):
    if not EP_BERT:
        return {"error": "ENDPOINT_ID_DISTILBERT not set"}

    aiplatform.init(project=PROJECT, location=REGION)
    endpoint = aiplatform.Endpoint(EP_BERT)
    response = endpoint.predict(instances=[{"text": text}])
    return response.predictions[0]

# ── Log-Reg helper (unchanged) ───────────────────────────────────────────────
# … your _ensure_local & logreg_predict here …

# ── Streamlit UI (unchanged) ────────────────────────────────────────────────
st.title("🎮 Steam Review Sentiment Demo")
txt = st.text_area("Paste a review ↓", height=160)
if st.button("Classify") and txt.strip():
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("DistilBERT (Vertex)")
        out = bert_predict(txt)
        st.write(out if "error" in out else
                 f"**{out['label']}** · {out['score']:.2%}")

    with col2:
        st.subheader("Log-Reg (local)")
        out = logreg_predict(txt)
        st.write(out if "error" in out else
                 f"**{out['label']}** · {out['score']:.2%}")
