import os
import joblib
import streamlit as st
import requests, subprocess, pathlib, tempfile
from google.cloud import bigquery, aiplatform
from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account

# ── CONFIG via env-vars ───────────────────────────────────────────────────────
PROJECT      = os.getenv("PROJECT_ID", "sentiment-analysis-steam")
REGION       = os.getenv("REGION",     "us-central1")
EP_BERT      = os.getenv("ENDPOINT_ID_DISTILBERT")
BUNDLE       = os.getenv("LOGREG_BUNDLE_PATH", "models/best_tfidf_lr_negRecall_20250630-050145.joblib.gz")
BQ_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", None)

# ── AUTH / CLIENTS ────────────────────────────────────────────────────────────
@st.cache_resource
def get_bq_client():
    if BQ_CREDENTIALS:
        creds = service_account.Credentials.from_service_account_file(BQ_CREDENTIALS)
        return bigquery.Client(project=PROJECT, credentials=creds)
    return bigquery.Client(project=PROJECT)

@st.cache_data(ttl=600)
def fetch_game_list():
    client = get_bq_client()
    sql = f"""
      SELECT DISTINCT game_name
      FROM `{PROJECT}.steam_reviews.top10_owned_steamcommunity`
      ORDER BY game_name
    """
    return [row.game_name for row in client.query(sql).result()]

@st.cache_data(ttl=300)
def fetch_sentiment_for(games: list[str]):
    client = get_bq_client()
    # build safe IN-clause
    quoted = ",".join(f"'{g.replace(\"'\",\"\\'\")}'" for g in games)
    sql = f"""
      SELECT
        game_name,
        COUNTIF(voted_up)   AS positives,
        COUNT(*) - COUNTIF(voted_up) AS negatives
      FROM `{PROJECT}.steam_reviews.top10_owned_steamcommunity`
      WHERE game_name IN ({quoted})
      GROUP BY game_name
      ORDER BY game_name
    """
    return client.query(sql).to_dataframe()

# ── DistilBERT (Vertex endpoint) ─────────────────────────────────────────────
def bert_predict(text: str):
    if not EP_BERT:
        return {"error": "ENDPOINT_ID_DISTILBERT not set"}
    aiplatform.init(project=PROJECT, location=REGION)
    endpoint = aiplatform.Endpoint(EP_BERT)
    response = endpoint.predict(instances=[{"text": text}])
    return response.predictions[0]

# ── Log-Reg helper (local) ──────────────────────────────────────────────────
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
    p = clf.predict_proba(vec.transform([text]))[0]
    return {"label": "POSITIVE" if p[1]>=.5 else "NEGATIVE", "score": float(p[1])}

# ── Streamlit UI ─────────────────────────────────────────────────────────────
st.title("🎮 Steam Review Sentiment Demo")

txt = st.text_area("Paste a review ↓", height=160)
if st.button("Classify") and txt.strip():
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("DistilBERT (Vertex)")
        out = bert_predict(txt)
        st.write(out if "error" in out else f"**{out['label']}** · {out['score']:.2%}")
    with col2:
        st.subheader("Log-Reg (local)")
        out = logreg_predict(txt)
        st.write(out if "error" in out else f"**{out['label']}** · {out['score']:.2%}")

# ── Dashboard: BigQuery Aggregates ───────────────────────────────────────────
# ── DASHBOARD ────────────────────────────────────────────────────────────────
st.header("📊 Review Sentiment Dashboard")

@st.cache_data
def load_game_list() -> list[str]:
    client = bigquery.Client()
    sql = """
      SELECT DISTINCT game_name
      FROM `sentiment-analysis-steam.steam_reviews.top10-owned-steamcommunity`
      ORDER BY game_name
    """
    df = client.query(sql).to_dataframe()
    return df["game_name"].tolist()

@st.cache_data
def get_sentiment_stats(game_name: str) -> tuple[int,int,int]:
    client = bigquery.Client()
    sql = """
      SELECT
        voted_up AS positive,
        COUNT(1)   AS cnt
      FROM `sentiment-analysis-steam.steam_reviews.top10-owned-steamcommunity`
      WHERE game_name = @game_name
      GROUP BY positive
    """
    job_config = bigquery.QueryJobConfig(
      query_parameters=[
        bigquery.ScalarQueryParameter("game_name", "STRING", game_name)
      ]
    )
    df = client.query(sql, job_config=job_config).to_dataframe()
    total = int(df["cnt"].sum())
    pos   = int(df.loc[df["positive"] == True,  "cnt"].sum() or 0)
    neg   = int(df.loc[df["positive"] == False, "cnt"].sum() or 0)
    return pos, neg, total

games = load_game_list()
choice = st.selectbox("Select a game to explore:", ["—"] + games)
if choice != "—":
    pos, neg, total = get_sentiment_stats(choice)
    st.metric("Positive reviews", f"{pos}/{total} ({pos/total:.1%})")
    st.metric("Negative reviews", f"{neg}/{total} ({neg/total:.1%})")

    chart_df = pd.DataFrame({
      "Sentiment": ["Positive", "Negative"],
      "Count":     [pos, neg]
    }).set_index("Sentiment")
    st.bar_chart(chart_df)