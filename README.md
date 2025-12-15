# MarketMinds

Predict next-day stock direction (up/down) using financial news headlines + market context.

## Pipeline
1) Ingest headline dataset (Kaggle)
2) Fetch OHLCV from Yahoo Finance
3) Align headlines to trading days
4) Clean + dedupe text
5) Label next-day return (up/down)
6) Baselines: market-only, TF-IDF text-only
7) FinBERT embeddings (frozen) + fused model
8) Rolling time-series evaluation + interpretability (SHAP)
