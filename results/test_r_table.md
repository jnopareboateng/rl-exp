# Arms table

## Test-R

| Arm | Group | Decoding | Valid answers | Delivered similarity (95% interval) | Samples per answer | Tokens per answer | Delivered lift | Lift, all samples | Mean similarity, all samples | Core match | Exact match |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | Control | deterministic, copied to eight samples | 1.000 | 0.160 [0.134, 0.188] | 1.00 | 0.0 | +0.000 | +0.000 | 0.160 | 0.000 | 0.000 |
| C1 | Control | deterministic, copied to eight samples | 1.000 | 0.166 [0.143, 0.187] | 1.00 | 0.0 | +0.038 | +0.038 | 0.166 | 0.000 | 0.000 |
| C2 | Control | deterministic, copied to eight samples | 1.000 | 0.078 [0.064, 0.093] | 1.00 | 0.0 | +0.000 | +0.000 | 0.078 | 0.000 | 0.000 |
| C3 | Control | deterministic, copied to eight samples | 1.000 | 0.159 [0.135, 0.187] | 1.00 | 0.0 | +0.000 | +0.000 | 0.159 | 0.000 | 0.000 |
| A0n: 4B untrained, no-think | 4B arm | T=0.6, k=8, no-think | 0.309 | 0.090 [0.071, 0.108] | 4.06 | 421.9 | +0.013 | +0.007 | 0.041 | 0.000 | 0.000 |
| A4: 4B + 3 retrieved examples | 4B arm | T=0.6, k=8, no-think | 0.567 | 0.138 [0.116, 0.156] | 2.15 | 290.9 | +0.024 | +0.018 | 0.087 | 0.000 | 0.000 |
| A2: 4B answer-only SFT (epoch 1) | 4B arm | T=1.0, k=8, think mode | 0.524 | 0.133 [0.120, 0.150] | 2.12 | 98.4 | +0.017 | +0.008 | 0.071 | 0.001 | 0.000 |
| RL-full: 4B RL, standard reward (R5-step-50) | 4B arm | T=0.3, k=8, think mode | 0.983 | 0.168 [0.147, 0.193] | 1.02 | 34.3 | +0.013 | +0.013 | 0.166 | 0.001 | 0.000 |
| A2-shuffle: A2 with a shuffle profile | Diagnostic | T=1.0, k=8, think mode | 0.546 | 0.122 [0.107, 0.139] | 2.01 | 93.5 | +0.004 | +0.002 | 0.067 | 0.000 | 0.000 |
| A2-blank: A2 with a blank profile | Diagnostic | T=1.0, k=8, think mode | 0.744 | n/a (shared draws) | n/a (shared draws) | n/a (shared draws) | n/a (shared draws) | +0.000 | 0.102 | 0.000 | 0.000 |
| RL-full-shuffle: RL-full with a shuffle profile | Diagnostic | T=0.3, k=8, think mode | 0.982 | 0.156 [0.134, 0.180] | 1.02 | 33.1 | +0.000 | +0.000 | 0.154 | 0.000 | 0.000 |
| RL-full-blank: RL-full with a blank profile | Diagnostic | T=0.3, k=8, think mode | 1.000 | n/a (shared draws) | n/a (shared draws) | n/a (shared draws) | n/a (shared draws) | +0.000 | 0.153 | 0.000 | 0.000 |
| name-4B: 4B name | Diagnostic | T=1.0, k=8, no-think | 0.283 | 0.099 [0.078, 0.120] | 3.66 | 286.2 | +0.008 | +0.003 | 0.034 | 0.000 | 0.000 |

Test-R split, 30 drugs. Similarity is the Tanimoto similarity (Morgan count vectors) to the reference structure. Delivered similarity and lift are those of the first valid, finished answer among up to eight samples of each prompt (0 if none), with its cost in samples and generated tokens per answer (prompts are not counted, and the arm with retrieved examples reads a longer one than the others). The bracket is a 95% interval from resampling groups of related drugs: descriptive, not a test. Lift is the similarity to the drug's own structure minus the mean similarity to the other drugs' structures. Core match and exact match are the share of samples with the reference's scaffold or connectivity (core) or its connectivity (exact), averaged over drugs. n/a: fewer than eight samples per prompt, so no first-valid value.
n/a (shared draws): a run with the profile removed shows every prompt the same text, so its draws repeat across prompts and its first valid answer is one shared molecule; read its mean similarity and valid answers.

Best constant answer (one fixed structure chosen from the training drugs): mean similarity 0.160, delivered similarity 0.160.
C3, the reward-optimal constant (what a policy that ignores the profile can at best learn): delivered similarity 0.159, mean similarity 0.159.

Identification (profile in, drug name out; a sample counts when its first answer line equals the drug's name):

| Arm | Decoding | Name match | Pass at k | k | Answered | Drugs |
|---|---|---:|---:|---:|---:|---:|
| ident-4B: 4B ident | T=1.0, k=8, no-think | 0.027 | 0.072 | 8 | 0.860 | 30 |
