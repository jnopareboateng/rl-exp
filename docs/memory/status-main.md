---
name: status-main
description: 'In flight: A2/baselines measured on Colab; RL pack and SFT fixes under review; prereg-v1 not tagged'
type: status
updated: 2026-10-17
---
Done: repo, plan v5, contracts, CPU pipeline and controls, sampling and SFT packs, Val baselines (4B, 9B, 27B-FP8, ICL, name-given, A2 SFT). Colab A100 with Drive mounted is the compute route.
In flight: SFT-pack fixes after two independent reviews; the answer-only RL pack (own loop, HF generation); A1 trace SFT; identification diagnostic (profile to drug name) at 4B, 9B, 27B.
Not done: primary-endpoint and decoding decisions (plan section 2, open), prereg-v1 tag, A3 and A5 results, any Test sampling.

**Why:** records where multi-session work stands so a fresh session can resume.
**How to apply:** delete this entry when the work merges into main-line results.
