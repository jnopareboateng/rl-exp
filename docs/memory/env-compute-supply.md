---
name: env-compute-supply
description: 'Kaggle gives 6 GPU-hours/week (resets 2026-10-24); no local GPU; Colab needs a browser accept'
type: env
updated: 2026-10-17
---
Kaggle API reports 21,600 GPU-seconds per week and 72,000 TPU-seconds for the account (refresh 2026-10-24 00:00, pay-to-scale off). The vLLM stack does not support P100. There is no local GPU. The Colab MCP connects only after the user accepts it in a signed-in browser tab.

**Why:** measured through the Kaggle quota call and colab_connection_status; the full study needs far more than one Kaggle week.
**How to apply:** run cheapest-first with gates; check colab_connection_status before assuming a Colab GPU; budget Kaggle sessions in whole GPU-hours.
