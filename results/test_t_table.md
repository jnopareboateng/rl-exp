# Arms table

## Test-T

| Arm | Group | Decoding | Valid answers | Delivered similarity (95% interval) | Samples per answer | Tokens per answer | Delivered lift | Lift, all samples | Mean similarity, all samples | Core match | Exact match |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | Control | deterministic, copied to eight samples | 1.000 | 0.184 [0.169, 0.198] | 1.00 | 0.0 | +0.000 | +0.000 | 0.184 | 0.000 | 0.000 |
| C1 | Control | deterministic, copied to eight samples | 1.000 | 0.168 [0.151, 0.184] | 1.00 | 0.0 | +0.014 | +0.014 | 0.168 | 0.000 | 0.000 |
| C2 | Control | deterministic, copied to eight samples | 1.000 | 0.077 [0.067, 0.088] | 1.00 | 0.0 | +0.000 | +0.000 | 0.077 | 0.000 | 0.000 |
| C3 | Control | deterministic, copied to eight samples | 1.000 | 0.199 [0.181, 0.216] | 1.00 | 0.0 | +0.000 | +0.000 | 0.199 | 0.000 | 0.000 |
| A0n: 4B untrained, no-think | 4B arm | T=0.6, k=8, no-think | 0.136 | 0.073 [0.053, 0.092] | 5.48 | 761.0 | +0.006 | +0.003 | 0.019 | 0.000 | 0.000 |
| A4: 4B + 3 retrieved examples | 4B arm | T=0.6, k=8, no-think | 0.530 | 0.140 [0.125, 0.155] | 2.40 | 359.8 | +0.013 | +0.011 | 0.087 | 0.000 | 0.000 |
| A2: 4B answer-only SFT (epoch 1) | 4B arm | T=1.0, k=8, think mode | 0.513 | 0.131 [0.118, 0.144] | 2.00 | 92.2 | -0.004 | +0.002 | 0.071 | 0.000 | 0.000 |
| RL-full: 4B RL, standard reward (R5-step-50) | 4B arm | T=0.3, k=8, think mode | 0.990 | 0.186 [0.170, 0.201] | 1.00 | 34.3 | +0.009 | +0.008 | 0.184 | 0.000 | 0.000 |
| A2-shuffle: A2 with a shuffle profile | Diagnostic | T=1.0, k=8, think mode | 0.528 | 0.133 [0.120, 0.146] | 1.89 | 84.4 | +0.001 | +0.001 | 0.070 | 0.000 | 0.000 |
| A2-blank: A2 with a blank profile | Diagnostic | T=1.0, k=8, think mode | 0.745 | n/a (shared draws) | n/a (shared draws) | n/a (shared draws) | n/a (shared draws) | +0.000 | 0.117 | 0.003 | 0.000 |
| RL-full-shuffle: RL-full with a shuffle profile | Diagnostic | T=0.3, k=8, think mode | 0.989 | 0.181 [0.163, 0.198] | 1.00 | 34.8 | +0.005 | +0.003 | 0.177 | 0.000 | 0.000 |
| RL-full-blank: RL-full with a blank profile | Diagnostic | T=0.3, k=8, think mode | 1.000 | n/a (shared draws) | n/a (shared draws) | n/a (shared draws) | n/a (shared draws) | +0.000 | 0.166 | 0.000 | 0.000 |
| name-4B: 4B name | Diagnostic | T=1.0, k=8, no-think | 0.220 | 0.112 [0.095, 0.130] | 3.58 | 275.0 | +0.003 | +0.001 | 0.026 | 0.000 | 0.000 |

Test-T split, 41 drugs. Similarity is the Tanimoto similarity (Morgan count vectors) to the reference structure. Delivered similarity and lift are those of the first valid, finished answer among up to eight samples of each prompt (0 if none), with its cost in samples and generated tokens per answer (prompts are not counted, and the arm with retrieved examples reads a longer one than the others). The bracket is a 95% interval from resampling groups of related drugs: descriptive, not a test. Lift is the similarity to the drug's own structure minus the mean similarity to the other drugs' structures. Core match and exact match are the share of samples with the reference's scaffold or connectivity (core) or its connectivity (exact), averaged over drugs. n/a: fewer than eight samples per prompt, so no first-valid value.
n/a (shared draws): a run with the profile removed shows every prompt the same text, so its draws repeat across prompts and its first valid answer is one shared molecule; read its mean similarity and valid answers.

Best constant answer (one fixed structure chosen from the training drugs): mean similarity 0.184, delivered similarity 0.184.
C3, the reward-optimal constant (what a policy that ignores the profile can at best learn): delivered similarity 0.199, mean similarity 0.199.

Identification (profile in, drug name out; a sample counts when its first answer line equals the drug's name):

| Arm | Decoding | Name match | Pass at k | k | Answered | Drugs |
|---|---|---:|---:|---:|---:|---:|
| ident-4B: 4B ident | T=1.0, k=8, no-think | 0.000 | 0.000 | 8 | 0.761 | 41 |
