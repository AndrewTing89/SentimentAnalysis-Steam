-- models/combined_deduplicated_reviews.sql
/*
    Purpose: Combines standardized review data from all sources (API and Kaggle)
             and performs deduplication to ensure each unique review appears only once.

    Transformations Applied:
    - Unions (combines) all rows from stg_raw_reviews and stg_kaggle_historical.
    - Deduplicates records based on 'recommendationid' (the unique review identifier).
    - Prioritizes the most recent version of a review based on:
        1. 'timestamp_updated' (if available, newest first)
        2. 'timestamp_created' (if 'timestamp_updated' is null, newest first)
        3. 'dbt_updated_at' (if other timestamps are null, newest dbt processing time)
      This ensures the latest version of a review is kept.

    Inputs:
    - {{ ref('stg_raw_reviews') }} (Output of stg_raw_reviews.sql)
    - {{ ref('stg_kaggle_historical') }} (Output of stg_kaggle_historical.sql)
    Output: A single, deduplicated dataset containing all unique reviews from both sources.
*/
WITH combined_sources AS (
    SELECT * FROM {{ ref('stg_raw_reviews') }}      -- Includes API data
    UNION ALL                                        -- Combines rows from both sources
    SELECT * FROM {{ ref('stg_kaggle_historical') }} -- Includes Kaggle historical data
),
deduplicated AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY recommendationid            -- Deduplicate by the unique review ID
            -- Prioritize newer records: timestamp_updated (API), then timestamp_created, then dbt processing time
            ORDER BY COALESCE(timestamp_updated, 0) DESC, COALESCE(timestamp_created, 0) DESC, dbt_updated_at DESC 
        ) as rn
    FROM
        combined_sources
)
SELECT
    recommendationid,
    review_text,
    review_score,
    app_id,
    app_name,
    timestamp_created,
    timestamp_updated,
    source_system,
    dbt_updated_at
FROM
    deduplicated
WHERE rn = 1 -- Select only the latest (based on ORDER BY) unique record