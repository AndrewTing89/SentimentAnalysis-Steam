-- models/stg_kaggle_historical.sql
/*
    Purpose: Stages historical Kaggle review data.
             This model standardizes column names and types to match the API data,
             and generates a unique ID for reviews if one doesn't exist.

    Transformations Applied:
    - Generates 'recommendationid' using an MD5 hash of review_text and timestamp_created
      if an existing 'recommendationid' is not present in the Kaggle data.
    - Renames 'app_name' to 'app_name' (no change, just explicit for clarity).
    - Explicitly selects 'review_text', 'review_score', 'app_id' as they are.
    - Adds NULL for 'timestamp_created' and 'timestamp_updated' if these fields
      are not present in the Kaggle CSV/table, to ensure schema compatibility for UNION ALL.
    - Adds 'source_system' column with value 'kaggle'.
    - Adds 'dbt_updated_at' timestamp.

    Input: {{ source('steam_reviews', 'kaggle_historical_raw') }} (BigQuery: steam_reviews.kaggle_historical_raw table)
    Output: Staged Kaggle review data with standardized column names and types.
*/
SELECT
    -- Generate a recommendationid if one doesn't exist (using hash of text and creation timestamp)
    COALESCE(recommendationid, MD5(CONCAT(review_text, COALESCE(CAST(timestamp_created AS STRING), '')))) AS recommendationid, 
    app_id,
    app_name,
    review_text,
    review_score,
    NULL AS timestamp_created, -- Assuming Kaggle CSV doesn't have precise creation timestamps
    NULL AS timestamp_updated, -- Assuming Kaggle CSV doesn't have update timestamps
    'kaggle' as source_system,
    CURRENT_TIMESTAMP() as dbt_updated_at
FROM
    {{ source('steam_reviews', 'kaggle_historical_raw') }} -- Reads from steam_reviews.kaggle_historical_raw