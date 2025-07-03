# 🎮 Steam Sentiment Analysis

A full-stack demonstration of how to turn raw Steam user reviews into actionable sentiment scores—from initial exploration and classical baselines to Transformer fine-tuning and a serverless web demo on GCP.  

**Live demo & article:**  
[Medium write-up](https://medium.com/…) • [GitHub repo](https://github.com/AndrewTing89/SentimentAnalysis-Steam)

---

## 🚀 Features

- **Data Exploration & Sampling**  
  – Explored a 50 K-review sample locally; then scaled everything up to the full 565 K-review dataset (2.16 GB) for final training.  
- **Classical ML Baseline**  
  – Compared nine text-vectorizer + classifier pipelines with `RandomizedSearchCV`, picked the best TF–IDF + Logistic Regression, and trained it on the full dataset.  
- **Transformer Fine-Tuning**  
  – Fine-tuned DistilBERT on the 565 K reviews, saved the checkpoint in a private Hugging Face repo.  
- **Managed Serving**  
  – Registered and deployed both models on Vertex AI Endpoints (TF–IDF + LR & DistilBERT).  
- **Interactive Demo**  
  – Streamlit app showing side-by-side predictions, deployed serverless on Cloud Run with CI/CD.  

---

## 📁 Repository Structure

```text
.
├── Notebooks/
│   ├── 04_DistilBERT_API.ipynb       # register & deploy DistilBERT on Vertex AI
│   ├── 05_LogReg.ipynb              # train & save TF–IDF + Logistic Regression on full data
│   └── 06_Streamlit_Deployment.ipynb # containerize & test Streamlit app locally
│
├── models/                          # (not checked in) local storage for joblib bundles
│   └── best_tfidf_lr_negRecall_YYYYMMDD-HHMMSS.joblib.gz
│
├── src/
│   └── deploy_distilbert.py         # optional script for automated model upload & deploy
│
├── app/
│   ├── streamlit_app.py             # Streamlit front-end
│   └── requirements.txt             # Python dependencies
│
├── Dockerfile                       # builds the Streamlit container
├── README.md                        # this file
└── LICENSE                          # MIT License
```

> **Note:** The original EDA/cleaning and DistilBERT-training notebooks were lost, but all model weights are safely stored in the private HF repo `andrewting89/steam-distilbert`.

---

## 🔧 Getting Started

### Prerequisites

- A **Google Cloud** project with billing enabled  
- `gcloud` CLI + appropriate IAM roles for Vertex AI & Cloud Run  
- A **GitHub** PAT stored in GCP Secret Manager  
- A **Hugging Face** PAT (for private model repo)

### Local Setup

```bash
git clone https://github.com/AndrewTing89/SentimentAnalysis-Steam.git
cd SentimentAnalysis-Steam/app
pip install -r requirements.txt
```

Set environment variables for local testing:

```bash
export PROJECT_ID=sentiment-analysis-steam
export REGION=us-central1
export ENDPOINT_ID_DISTILBERT=<your-vertex-endpoint-id>
export LOGREG_BUNDLE_PATH=models/best_tfidf_lr_negRecall_*.joblib.gz
```

---

## 📡 Serving & Deployment

### 1. Register & Deploy DistilBERT on Vertex AI

_In Notebook 04 or via `src/deploy_distilbert.py`:_

```python
from google.cloud import aiplatform
aiplatform.init(project=PROJECT_ID, location=REGION)

# Upload the HF checkpoint
model = aiplatform.Model.upload(
  display_name="steam-distilbert-full",
  serving_container_image_uri="<HF_CPU_IMAGE_URI>",
  serving_container_environment_variables={
    "HF_MODEL_ID": "andrewting89/steam-distilbert",
    "HF_TASK":     "text-classification",
    "HUGGING_FACE_HUB_TOKEN": "<HF_TOKEN>",
  },
  sync=True,
)

# Deploy to endpoint
endpoint = aiplatform.Endpoint.list(
  filter='display_name="steam-sentiment-endpoint"', location=REGION
)[0]
model.deploy(
  endpoint           = endpoint,
  machine_type       = "n1-standard-4",
  min_replica_count  = 1,
  max_replica_count  = 1,
  traffic_percentage = 100,
  sync               = True,
)
```

### 2. Run & Test Streamlit Locally

```bash
cd app
streamlit run streamlit_app.py   --server.address 0.0.0.0 --server.port 8501
```

### 3. Deploy to Cloud Run

```bash
# Build & push container
gcloud builds submit --tag gcr.io/$PROJECT_ID/steam-sentiment-ui .

# Deploy service
gcloud run deploy steam-sentiment-ui   --image gcr.io/$PROJECT_ID/steam-sentiment-ui   --region $REGION   --platform managed   --allow-unauthenticated   --set-env-vars PROJECT_ID=$PROJECT_ID,REGION=$REGION,ENDPOINT_ID_DISTILBERT=$ENDPOINT_ID_DISTILBERT,LOGREG_BUNDLE_PATH=models/best_tfidf_lr_negRecall_*.joblib.gz
```

> **Demo Note:** To reduce costs, the Vertex AI endpoint for DistilBERT weights is scaled to zero when idle. If you’d like to see the live demo in action, please reach out to me so I can spin up the endpoint for you.

---

## 🤝 Contributing

Contributions welcome! Open an issue or submit a PR.

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
