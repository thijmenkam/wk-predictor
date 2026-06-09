# Polymarket raw data

This folder contains the manually curated `market_manifest.yaml`.

- Discovery and market metadata use the public Polymarket Gamma API.
- Pricing uses public, read-only Polymarket CLOB endpoints. It does not use authentication,
  API keys, trading, or order placement.
- CLOB `BUY` and `SELL` prices are stored as bid and ask. When both exist, their midpoint
  is the `chosen_probability`; otherwise the available side is used.
- `normalized_probability` divides priced outcomes by their market total. Raw bid, ask,
  midpoint, spread, and chosen probability remain unchanged.
- `world-cup-winner` is modeled as `event_binary_markets`: each team has a separate
  Yes/No market and only the YES token price is used as its raw win probability.
- Raw YES probabilities across teams may sum above or below 1. The
  `normalized_probability` column rescales mapped, priced teams for comparison only; it
  is not model calibration.
- Team-name normalization is configured in `entity_aliases.yaml`. Unknown entities stay
  in the processed output with a warning instead of being silently dropped.
- Search results are candidates for human review, not calibration inputs.
- Public web `/event/{slug}` URLs identify event slugs and may not identify Gamma market
  slugs. Start with `event_slug`, then run `polymarket-fetch-manifest` to create a
  `market_candidates.csv`.
- Fill `market_slug` only after human review and verification of resolution rules.
  Ordinary events resolve automatically only when exactly one priceable market exists.
  Entries with `structure: event_binary_markets` intentionally process all priceable
  binary Yes/No markets in the event.
- Raw prediction-market prices are not bookmaker odds. Liquidity and spread affect their
  quality, and illiquid or wide-spread markets require caution.
- These prices are not used for model calibration or predictions yet. Explicit
  market-versus-model comparison comes later.
- No credentials, private keys, or automated market selection belong in this workflow.
