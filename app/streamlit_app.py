import os, joblib, streamlit as st, google.auth, subprocess, pathlib, tempfile
from google.auth.transport.requests import AuthorizedSession

# ── CONFIG via env-vars ───────────────────────────────────────────────────────
PROJECT  = os.getenv("PROJECT_ID", "sentiment-analysis-steam")
REGION   = os.getenv("REGION",     "us-central1")
EP_BERT  = os.getenv("ENDPOINT_ID_DISTILBERT")              # 18-digit ID
BUNDLE   = os.getenv(
    "LOGREG_BUNDLE_PATH",
    "models/best_tfidf_lr_negRecall_20250630-050145.joblib.gz"
)

# ── DistilBERT (Vertex endpoint) ─────────────────────────────────────────────
def bert_predict(text: str):
    if not EP_BERT:
        return {"error": "ENDPOINT_ID_DISTILBERT not set"}
    url = (f"https://{REGION}-aiplatform.googleapis.com/v1/projects/"
           f"{PROJECT}/locations/{REGION}/endpoints/{EP_BERT}:predict")
    creds, _ = google.auth.default()
    r = AuthorizedSession(creds).post(url, json={"instances":[{"text": text}]})
    r.raise_for_status()
    return r.json()["predictions"][0]

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

# ── Streamlit UI ─────────────────────────────────────────────────────────────
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
