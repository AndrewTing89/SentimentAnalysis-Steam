import os, joblib, streamlit as st, pathlib, subprocess, tempfile
from google.cloud import aiplatform

# ── CONFIG via env-vars ───────────────────────────────────────────────────────
PROJECT  = os.getenv("PROJECT_ID", "sentiment-analysis-steam")
REGION   = os.getenv("REGION",     "us-central1")
EP_BERT  = os.getenv("ENDPOINT_ID_DISTILBERT")  # 18-digit ID
BUNDLE   = os.getenv(
    "LOGREG_BUNDLE_PATH",
    "models/best_tfidf_lr_negRecall_20250630-050145.joblib.gz"
)

# ── Initialize Vertex client (once) ──────────────────────────────────────────
aiplatform.init(project=PROJECT, location=REGION)
_endpoint = aiplatform.Endpoint(EP_BERT)

def bert_predict(text: str):
    if not EP_BERT:
        return {"error": "ENDPOINT_ID_DISTILBERT not set"}
    # this automatically wraps your text in the right JSON
    response = _endpoint.predict(instances=[{"text": text}])
    # response.predictions is a list; take the first element
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
