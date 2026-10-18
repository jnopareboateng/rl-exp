| Arm | Decoding | Valid answers | Delivered similarity | Samples per answer | Tokens per answer | Delivered lift | Lift, all samples | Mean similarity, all samples | Core match | Exact match |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | deterministic, copied to eight samples | 1.000 | 0.184 | 1.00 | 0.0 | +0.000 | +0.000 | 0.184 | 0.000 | 0.000 |
| C1 | deterministic, copied to eight samples | 1.000 | 0.168 | 1.00 | 0.0 | +0.014 | +0.014 | 0.168 | 0.000 | 0.000 |
| C2 | deterministic, copied to eight samples | 1.000 | 0.077 | 1.00 | 0.0 | +0.000 | +0.000 | 0.077 | 0.000 | 0.000 |
| C3 | deterministic, copied to eight samples | 1.000 | 0.199 | 1.00 | 0.0 | +0.000 | +0.000 | 0.199 | 0.000 | 0.000 |
| 4B untrained, no-think | T=0.6, k=8, no-think | 0.136 | 0.073 | 5.48 | 761.0 | +0.006 | +0.003 | 0.019 | 0.000 | 0.000 |
| 4B + 3 retrieved examples | T=0.6, k=8, no-think | 0.530 | 0.140 | 2.40 | 359.8 | +0.013 | +0.011 | 0.087 | 0.000 | 0.000 |
| 4B answer-only SFT (epoch 1) | T=1.0, k=8, think mode | 0.513 | 0.131 | 2.00 | 92.2 | -0.004 | +0.002 | 0.071 | 0.000 | 0.000 |
| 4B RL, standard reward (R5-step-50) | T=0.3, k=8, think mode | 0.990 | 0.186 | 1.00 | 34.3 | +0.009 | +0.008 | 0.184 | 0.000 | 0.000 |
| A2 with a shuffle profile | T=1.0, k=8, think mode | 0.528 | 0.133 | 1.89 | 84.4 | +0.001 | +0.001 | 0.070 | 0.000 | 0.000 |
| A2 with a blank profile | T=1.0, k=8, think mode | 0.745 | 0.137 | 1.00 | 28.0 | +0.000 | +0.000 | 0.117 | 0.003 | 0.000 |
| RL-full with a shuffle profile | T=0.3, k=8, think mode | 0.989 | 0.181 | 1.00 | 34.8 | +0.005 | +0.003 | 0.177 | 0.000 | 0.000 |
| RL-full with a blank profile | T=0.3, k=8, think mode | 1.000 | 0.163 | 1.00 | 26.0 | +0.000 | +0.000 | 0.166 | 0.000 | 0.000 |
| 4B name | T=1.0, k=8, no-think | 0.220 | 0.112 | 3.58 | 275.0 | +0.003 | +0.001 | 0.026 | 0.000 | 0.000 |
| 4B ident | T=1.0, k=8, no-think | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

Test-T split, 41 drugs. Similarity is the Tanimoto similarity (Morgan count vectors) to the reference structure. Delivered similarity and lift are those of the first valid, finished answer among up to eight samples of each prompt (0 if none), with its cost in samples and generated tokens per answer (prompts are not counted). Lift is the similarity to the drug's own structure minus the mean similarity to the other drugs' structures. Core match and exact match are the share of samples with the reference's scaffold or connectivity (core) or its connectivity (exact), averaged over drugs. n/a: fewer than eight samples per prompt, so no first-valid value.
