{{ config(materialized='view') }}

with raw as (

  select *
  from {{ source('steam_reviews','top10_owned_steamcommunity') }}
)

select
  review,
  language,                    -- ← we added this
  voted_up,                    -- still raw BOOL here
  app_id     as game_id,       -- optional rename now or later
  game_name
from raw