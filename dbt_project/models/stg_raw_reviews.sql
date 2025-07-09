-- models/stg_raw_reviews.sql
/*
    Purpose: Stages raw review data ingested from Steam APIs.
             This model performs initial cleaning, renaming, and type casting
             to standardize the schema for combination with historical data.

    Transformations Applied:
    - Renames 'review' to 'review_text'.
    - Converts boolean 'voted_up' to INT64 (1 for True, 0 for False) and renames to 'review_score'.
    - Renames 'game_name' to 'app_name' for consistent naming across sources.
    - Adds 'source_system' column with value 'api'.
    - Adds 'dbt_updated_at' timestamp.

    Input: {{ source('steam_reviews', 'raw_reviews') }} (BigQuery: steam_reviews.raw_reviews table)
    Output: Staged API review data with standardized column names and types.
*/
SELECT
    recommendationid,
    review AS review_text,                 -- Rename 'review' to 'review_text'
    CAST(voted_up AS INT64) AS review_score, -- Convert BOOL 'voted_up' to INT64 (1 or 0)
    app_id,
    game_name AS app_name,                 -- Rename 'game_name' to 'app_name'
    timestamp_created,
    timestamp_updated,
    'api' as source_system,                -- Identify data source
    CURRENT_TIMESTAMP() as dbt_updated_at  -- When this record was processed by dbt
FROM
    {{ source('steam_reviews', 'raw_reviews') }} -- Reads from steam_reviews.raw_reviews
-- Add any basic filtering here if needed, e.g., WHERE language = 'english'