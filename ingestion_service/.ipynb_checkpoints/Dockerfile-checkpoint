# ingestion_service/Dockerfile

FROM python:3.9-slim

WORKDIR /app
COPY . /app

# Install production dependencies.
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Use Gunicorn to run the Flask app, explicitly invoking a shell for ${PORT} expansion.
# This makes sure the ${PORT} environment variable is correctly substituted.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --timeout 600 ingestion_app:app"] # <-- ADD --timeout 600 HERE
