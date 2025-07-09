FROM python:3.9-slim

WORKDIR /app

# Install dbt and its BigQuery adapter
# Using ~1.6 for compatibility, adjust if newer dbt version is preferred
RUN pip install --no-cache-dir dbt-bigquery~=1.6 

# Copy your dbt project files into the container
COPY . /app/dbt_project

# Set the working directory to the dbt project folder within the container
WORKDIR /app/dbt_project

# Entrypoint for dbt commands (this will be overridden by Cloud Build)
ENTRYPOINT ["dbt"]