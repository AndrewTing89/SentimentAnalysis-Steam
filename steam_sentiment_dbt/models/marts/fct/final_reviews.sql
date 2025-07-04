{{  
  config(
    materialized = "table"
  )  
}}

with raw as (
  select * from {{ ref('stg_steam_reviews') }}
)

select
  review,
  language,                                                   -- ← added!
  case when voted_up then 1 else 0 end    as sentiment,
  game_id                                  as game_id,
  game_name
from raw