# ingestion_service/ingestion_app.py

import os
import pandas as pd
from flask import Flask, request, jsonify
from google.cloud import storage, bigquery
import datetime as dt
import requests
import json
import re
import io
import time

app = Flask(__name__)

# --- Configuration (will be pulled from environment variables in Cloud Run) ---
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
if not GCS_BUCKET_NAME:
    raise ValueError("GCS_BUCKET_NAME environment variable not set.")

GCS_DATA_FILE_PATH = os.environ.get("GCS_DATA_FILE_PATH", "steam_reviews_cleaned.csv") 

BQ_PROJECT_ID = os.environ.get("BQ_PROJECT_ID")
if not BQ_PROJECT_ID:
    raise ValueError("BQ_PROJECT_ID environment variable not set.")

BQ_DATASET_ID = os.environ.get("BQ_DATASET_ID", "steam_reviews")
BQ_RAW_TABLE_ID = os.environ.get("BQ_RAW_TABLE_ID", "raw_reviews")

REGION_ENV = os.environ.get("REGION", "US")

print(f"App config: GCS_BUCKET_NAME={GCS_BUCKET_NAME}, BQ_PROJECT_ID={BQ_PROJECT_ID}, REGION_ENV={REGION_ENV}")

# Initialize GCS client
print("Initializing GCS client...")
try:
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    print("GCS client initialized.")
except Exception as e:
    print(f"ERROR: Failed to initialize GCS client or access bucket: {e}")
    raise

# Initialize BigQuery client
print("Initializing BigQuery client...")
try:
    bq_client = bigquery.Client(project=BQ_PROJECT_ID)
    print("BigQuery client initialized.")
except Exception as e:
    print(f"ERROR: Failed to initialize BigQuery client: {e}")
    raise

# --- Define Schema for RAW Steam Reviews (MOVED TO GLOBAL SCOPE) ---
# This schema is now accessible by all functions in this module.
raw_reviews_schema = [
    bigquery.SchemaField("recommendationid", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("author", "RECORD", mode="NULLABLE", fields=[
        bigquery.SchemaField("steamid", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("num_games_owned", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("num_reviews", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("playtime_forever", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("playtime_last_two_weeks", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("playtime_at_review", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("last_played", "INT64", mode="NULLABLE"),
    ]),
    bigquery.SchemaField("language", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("review", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("timestamp_created", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("timestamp_updated", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("voted_up", "BOOL", mode="NULLABLE"),
    bigquery.SchemaField("votes_up", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("votes_funny", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("weighted_vote_score", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("comment_count", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("steam_purchase", "BOOL", mode="NULLABLE"),
    bigquery.SchemaField("received_for_free", "BOOL", mode="NULLABLE"),
    bigquery.SchemaField("written_during_early_access", "BOOL", mode="NULLABLE"),
    bigquery.SchemaField("steam_deck_review", "BOOL", mode="NULLABLE"),
    bigquery.SchemaField("app_id", "INT64", mode="NULLABLE"),
    bigquery.SchemaField("game_name", "STRING", mode="NULLABLE"),
]

# --- NEW ADDITION: Define the Basic Outbound Network Test Function ---
def test_outbound_network(test_url="https://www.google.com"):
    print(f"DEBUG: Performing outbound network test to {test_url}...")
    try:
        test_resp = requests.get(test_url, timeout=5) # Short timeout for a quick test
        test_resp.raise_for_status()
        print(f"DEBUG: Outbound network test SUCCESS: {test_url} responded with {test_resp.status_code}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Outbound network test FAILED for {test_url}: {type(e).__name__} - {e}")
        return False
    except Exception as e:
        print(f"ERROR: Unexpected error during outbound network test: {type(e).__name__} - {e}")
        return False

# --- NEW ADDITION: Call the network test during global initialization ---
# This will run when the Cloud Run container starts up, as part of the app's initial checks.
print("DEBUG: Running startup network test...")
if not test_outbound_network():
    raise Exception("Critical: Outbound network access failed during startup health check.")
print("DEBUG: Startup network test complete.")

# --- Helper Function for BigQuery Operations ---
def create_bq_dataset_and_table_if_not_exists():
    """Ensures the BigQuery dataset and raw_reviews table exist."""
    dataset_ref = bq_client.dataset(BQ_DATASET_ID)
    dataset = bigquery.Dataset(dataset_ref)
    
    dataset.location = REGION_ENV # Set dataset location based on environment variable
    print(f"Ensuring BigQuery dataset {BQ_DATASET_ID} exists in location {dataset.location}...")

    try:
        bq_client.get_dataset(dataset_ref)
        print(f"Dataset {BQ_DATASET_ID} already exists.")
    except Exception as e:
        try:
            bq_client.create_dataset(dataset)
            print(f"Created dataset {BQ_DATASET_ID}.")
        except Exception as create_e:
            print(f"ERROR: Failed to create BigQuery dataset {BQ_DATASET_ID}: {create_e}")
            raise

    table_ref = dataset_ref.table(BQ_RAW_TABLE_ID)
    table = bigquery.Table(table_ref)

    # Schema is now in global scope, so no need to define it here again
    print(f"Ensuring BigQuery table {BQ_RAW_TABLE_ID} exists...")
    try:
        bq_client.get_table(table_ref)
        print(f"Table {BQ_RAW_TABLE_ID} already exists.")
    except Exception as e:
        try:
            table.schema = raw_reviews_schema # Use the globally defined schema
            bq_client.create_table(table)
            print(f"Created table {BQ_RAW_TABLE_ID}.")
        except Exception as create_e:
            print(f"ERROR: Failed to create BigQuery table {BQ_RAW_TABLE_ID}: {create_e}")
            raise

# --- SteamSpy and Steam API Logic (from your notebook) ---

def owners_upper(o_str: str) -> int:
    try:
        return int(o_str.split("..")[-1].replace(",", ""))
    except:
        return 0

def get_top_10_steam_games():
    print("Fetching top 10 Steam games from SteamSpy...")
    url = "https://steamspy.com/api.php"
    resp = requests.get(url, params={"request":"top100forever"}, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    games = list(data.values())
    games_sorted = sorted(
        games,
        key=lambda g: owners_upper(g.get("owners", "")),
        reverse=True
    )
    top10 = games_sorted[:10]
    
    top10_games_list = []
    print("Top 10 Steam Games by all-time owners:")
    for g in top10:
        aid = g["appid"]
        name = g.get("name", "<unknown>")
        owners = g.get("owners", "<n/a>")
        print(f" • {aid}: {name} (owners ≈ {owners})")
        top10_games_list.append((aid, name))
    
    return top10_games_list
''' 
### original fetch_raw_recent_reviews
def fetch_raw_recent_reviews(
    appid: int,
    game_name: str,
    per_page: int = 10,
    max_pages: int = 1,
    pause: float = 0.2
) -> list:
    """Fetches recent reviews for a given appid and returns raw JSON records."""
    print(f"Fetching raw recent reviews for {game_name} (AppID: {appid})...")
    
    all_raw_reviews = []
    base_url = f"https://store.steampowered.com/appreviews/{appid}"
    params = {
        "json": "1",
        "language": "english",
        "filter": "recent",
        "num_per_page": str(per_page),
    }
    cursor = None
    page = 0

    while page < max_pages:
        page += 1
        if cursor:
            params["cursor"] = cursor

        try:
            resp = requests.get(base_url, params=params, headers={"User-Agent":"Mozilla/5.0"}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            reviews = data.get("reviews", [])
            if not reviews:
                print(f"→ {game_name}: no reviews on page {page}, stopping.")
                break

            print(f"→ {game_name}: page {page}/{max_pages}, got {len(reviews)} reviews.")

            for r in reviews:
                r["app_id"] = appid
                r["game_name"] = game_name
                all_raw_reviews.append(r)

            cursor = data.get("cursor", "")
            if not cursor:
                print(f"→ {game_name}: no next cursor; done.")
                break

        except requests.exceptions.RequestException as e:
            print(f"Error fetching reviews for {game_name} (AppID: {appid}): {e}")
            break
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for {game_name} (AppID: {appid}): {e}")
            break
        
        time.sleep(pause)

    print(f"✅ Done: fetched {len(all_raw_reviews):,} raw reviews for {game_name}")
    return all_raw_reviews
'''

def fetch_raw_recent_reviews(
    appid: int,
    game_name: str,
    per_page: int = 100,  # Keeping this smaller for debugging, can be 100 later
    max_pages: int = 5,  # <-- TEMPORARILY REDUCED TO 1 PAGE FOR FASTER DEBUGGING CYCLES
    pause: float = 0.2
) -> list:
    """Fetches recent reviews for a given appid and returns raw JSON records."""
    print(f"Fetching raw recent reviews for {game_name} (AppID: {appid})...")
    
    all_raw_reviews = []
    base_url = f"https://store.steampowered.com/appreviews/{appid}"
    params = {
        "json": "1",
        "language": "english",
        "filter": "recent",
        "num_per_page": str(per_page),
    }
    cursor = None
    page = 0

    while page < max_pages:
        page += 1
        if cursor:
            params["cursor"] = cursor

        try:
            # --- ADDED MORE SPECIFIC EXCEPTION HANDLING FOR REQUESTS ---
            resp = requests.get(base_url, params=params, headers={"User-Agent":"Mozilla/5.0"}, timeout=60) # <-- INCREASED TIMEOUT TO 60 SECONDS
            resp.raise_for_status() # This raises HTTPError for 4xx/5xx responses (e.g., 429 Too Many Requests)
            data = resp.json()
            reviews = data.get("reviews", [])
            if not reviews:
                print(f"→ {game_name}: no reviews on page {page}, stopping.")
                break

            print(f"→ {game_name}: page {page}/{max_pages}, got {len(reviews)} reviews.")

            for r in reviews:
                r["app_id"] = appid
                r["game_name"] = game_name
                all_raw_reviews.append(r)

            cursor = data.get("cursor", "")
            if not cursor:
                print(f"→ {game_name}: no next cursor; done.")
                break

        except requests.exceptions.Timeout as e:
            print(f"ERROR: Request Timeout for {game_name} (AppID: {appid}): {e}")
            break # Stop fetching for this game on timeout
        except requests.exceptions.ConnectionError as e:
            print(f"ERROR: Connection Error for {game_name} (AppID: {appid}): {e}")
            break # Stop fetching for this game on connection error
        except requests.exceptions.HTTPError as e: # This handles 4xx/5xx status codes
            print(f"ERROR: HTTP Error for {game_name} (AppID: {appid}): {e.response.status_code} - {e.response.text}")
            break # Stop fetching for this game on HTTP error (e.g., 429 Too Many Requests)
        except requests.exceptions.RequestException as e: # Catch any other requests-related error
            print(f"ERROR: General Request Error for {game_name} (AppID: {appid}): {e}")
            break # Stop fetching for this game on general request error
        except json.JSONDecodeError as e:
            print(f"ERROR: JSON Decode Error for {game_name} (AppID: {appid}): {e}")
            break # Stop fetching for this game on JSON decode error
        except Exception as e: # Catch any other unexpected errors
            print(f"ERROR: Unexpected error in fetching reviews for {game_name} (AppID: {appid}): {type(e).__name__} - {e}")
            break # Stop fetching for this game on any other error
        
        time.sleep(pause)

    print(f"✅ Done: fetched {len(all_raw_reviews):,} raw reviews for {game_name}")
    return all_raw_reviews

# --- Main Ingestion Logic ---
def ingest_and_update_data_to_bq():
    print("Starting data ingestion process to BigQuery...")
    
    create_bq_dataset_and_table_if_not_exists()

    # 1. Get Top 10 Games
    top10_games = get_top_10_steam_games()
    if not top10_games:
        print("No top 10 games found. Exiting ingestion.")
        return False

    # 2. Fetch Raw Reviews for all games
    all_raw_reviews_for_bq = []
    for appid, name in top10_games:
        reviews_for_game = fetch_raw_recent_reviews(
            appid=appid,
            game_name=name,
            max_pages=5,
            per_page=100,
            pause=0.2
        )
        all_raw_reviews_for_bq.extend(reviews_for_game)
    
    if not all_raw_reviews_for_bq:
        print("No new raw reviews fetched. BigQuery table not updated.")
        return True

    # --- Convert list of dicts to JSONL string for GCS upload ---
    jsonl_data = "\n".join([json.dumps(r, ensure_ascii=False) for r in all_raw_reviews_for_bq])
    
    # --- Construct GCS destination path with date partitioning ---
    current_date_str = dt.datetime.utcnow().strftime("%Y-%m-%d")
    current_timestamp_str = dt.datetime.utcnow().strftime("%H%M%S")
    gcs_jsonl_file_name = f"steam_reviews_raw_{current_timestamp_str}.jsonl"
    
    GCS_LANDING_ZONE_PREFIX = f"raw_data/steam_reviews_api_raw/{current_date_str}"
    gcs_destination_blob_path = f"{GCS_LANDING_ZONE_PREFIX}/{gcs_jsonl_file_name}"

    # 3. Upload Raw Reviews to GCS Landing Zone
    print(f"Uploading {len(all_raw_reviews_for_bq):,} raw reviews to GCS: gs://{GCS_BUCKET_NAME}/{gcs_destination_blob_path}")
    try:
        blob = bucket.blob(gcs_destination_blob_path)
        blob.upload_from_string(jsonl_data, content_type="application/jsonl")
        print("✅ Raw reviews uploaded to GCS successfully.")
    except Exception as e:
        print(f"ERROR: Failed to upload raw reviews to GCS: {e}")
        return False

    # 4. Load Raw Reviews from GCS into BigQuery Table
    print(f"Loading raw reviews from GCS into BigQuery table {BQ_DATASET_ID}.{BQ_RAW_TABLE_ID}...")
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND, # Append to existing table
        schema=raw_reviews_schema, # Use the schema defined earlier
        ignore_unknown_values=True, # Good for raw data ingestion
    )

    load_job = bq_client.load_table_from_uri(
        f"gs://{GCS_BUCKET_NAME}/{gcs_destination_blob_path}",
        f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_RAW_TABLE_ID}",
        job_config=job_config
    )
    
    try:
        load_job.result() # Wait for the job to complete
        print(f"✅ Loaded {load_job.output_rows} rows from GCS into BigQuery table {BQ_RAW_TABLE_ID}.")
        return True
    except Exception as e:
        print(f"ERROR: BigQuery load job failed: {e}")
        if load_job.errors:
            print("BigQuery job errors:", load_job.errors)
        return False

# --- Flask Endpoint for Cloud Run ---
@app.route('/', methods=['POST'])
def ingest_data_trigger():
    if request.method == 'POST':
        try:
            success = ingest_and_update_data_to_bq()
            if success:
                # In the next phase, you'd trigger your DBT transformation here (e.g., via Cloud Build)
                return jsonify({"status": "success", "message": "Raw data ingested to BigQuery."}), 200
            else:
                return jsonify({"status": "failed", "message": "Raw data ingestion to BigQuery failed."}), 500
        except Exception as e:
            print(f"Error during ingestion: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500
    return 'OK', 200

# ingestion_service/ingestion_app.py (at the very bottom)

if __name__ == '__main__':
    # For local testing only
    import io
    # Set dummy environment variables for local testing
    if "GCS_BUCKET_NAME" not in os.environ:
        os.environ["GCS_BUCKET_NAME"] = "steam-reviews-bucket-0" # <-- USE YOUR ACTUAL BUCKET NAME HERE!
    if "GCS_DATA_FILE_PATH" not in os.environ: # This one is for later, keep it
        os.environ["GCS_DATA_FILE_PATH"] = "steam_reviews_cleaned.csv"
    if "BQ_PROJECT_ID" not in os.environ:
        os.environ["BQ_PROJECT_ID"] = "sentiment-analysis-steam" # <-- USE YOUR ACTUAL PROJECT ID HERE!
    if "BQ_DATASET_ID" not in os.environ: # This one is for later, keep it
        os.environ["BQ_DATASET_ID"] = "steam_reviews"
    if "BQ_RAW_TABLE_ID" not in os.environ: # This one is for later, keep it
        os.environ["BQ_RAW_TABLE_ID"] = "raw_reviews"
    if "REGION" not in os.environ: # Add REGION for BQ dataset location check
        os.environ["REGION"] = "us-west1" # <-- Use your Cloud Run/Bucket region
    
    print("Running ingestion_app.py locally...")
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8081)))