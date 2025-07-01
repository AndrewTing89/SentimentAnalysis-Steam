FROM python:3.11-slim
WORKDIR /app

# install dependencies first (layer cache friendly)
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy UI code and tiny model bundle
COPY app/streamlit_app.py .
COPY models/*.joblib.gz

EXPOSE 8080
CMD ["streamlit", "run", "/app/streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8080"]
