---
name: env-compute-supply
description: 'Kaggle: 2 x T4, 6 GPU-h/week, batch runs only; Colab is primary and needs a browser accept'
type: env
updated: 2026-10-17
---
Kaggle gives this account 21,600 GPU-seconds per week (refresh 2026-10-24) on two Tesla T4 (15 GB each, driver R580); the A100 shape request is ignored and P100 is unsupported by the vLLM stack. Use batch kernels (`kaggle kernels push -t SECONDS`) with a dataset for code and data; interactive API sessions leak quota. There is no local GPU. The user chose Colab as the primary GPU route on 2026-10-17; the Colab MCP connects only after the user accepts it in a signed-in browser tab.

**Why:** measured through the quota call and a batch run's environment dump. Thinking-mode RL does not fit on T4s (about 1k tokens per second decode); answer-only RL does.
**How to apply:** cap every run with a timeout; check colab_connection_status before assuming a Colab GPU; read the runtime's GPU before sizing a run.

Related: [[gotcha-kaggle-session-billing]]
