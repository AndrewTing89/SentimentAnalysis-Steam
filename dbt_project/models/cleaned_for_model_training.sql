-- models/cleaned_for_model_training.sql
/*
    Purpose: Creates the final dataset explicitly for machine learning model training.
             It selects the exact columns needed by the LR model and applies final
             data quality filters.

    Transformations Applied:
    - Selects 'recommendationid', 'review_text', 'review_score', 'app_id', 'app_name'.
    - Filters out records where 'review_text' is NULL or empty.
    - Filters out records where 'review_score' is NULL.

    Input: {{ ref('combined_deduplicated_reviews') }} (Output of combined_deduplicated_reviews.sql)
    Output: A clean, model-ready BigQuery table. This table will be exported to GCS CSV
            for consumption by the LR model training job.
*/
SELECT
    recommendationid,
    review_text,
    review_score,
    app_id,   -- Including app_id and app_name for potential future analysis or features
    app_name
FROM
    {{ ref('combined_deduplicated_reviews') }} -- Reads from the combined and deduplicated model
WHERE
    review_text IS NOT NULL AND TRIM(review_text) != '' -- Ensure review text is not empty or just whitespace
    AND review_score IS NOT NULL                        -- Ensure review_score is present