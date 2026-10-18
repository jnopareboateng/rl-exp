| Arm | Decoding | Valid answers | Delivered similarity | Samples per answer | Tokens per answer | Delivered lift | Lift, all samples | Mean similarity, all samples | Core match | Exact match |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 constant answer (best Train reference) | none | 1.000 | 0.176 | 1.00 | 0.0 | +0.000 | +0.000 | 0.176 | 0.000 | 0.000 |
| C1 copy the structure of the nearest Train profile (TF-IDF) | none | 1.000 | 0.174 | 1.00 | 0.0 | +0.022 | +0.022 | 0.174 | 0.000 | 0.000 |
| C2 constant scaffold answer | none | 1.000 | 0.076 | 1.00 | 0.0 | +0.000 | +0.000 | 0.076 | 0.000 | 0.000 |
| C3 reward-optimal constant (chosen on Train from Train + ChEMBL pool) | none | 1.000 | 0.187 | 1.00 | 0.0 | +0.000 | +0.000 | 0.187 | 0.000 | 0.000 |
| 4B untrained | T=1.0, k=8 | 0.158 | 0.087 | 4.90 | 377.0 | +0.014 | +0.005 | 0.025 | 0.008 | 0.000 |
| 4B + 3 retrieved examples (A4) | T=1.0, k=8 | 0.490 | 0.139 | 2.30 | 200.6 | +0.005 | +0.014 | 0.079 | 0.023 | 0.000 |
| 4B + 3 retrieved examples (A4) | T=0.6, k=8 | 0.681 | 0.160 | 1.66 | 139.5 | +0.021 | +0.022 | 0.118 | 0.014 | 0.000 |
| 4B + 3 retrieved examples (A4) | T=0.3, k=8 | 0.730 | 0.154 | 1.58 | 151.5 | +0.019 | +0.018 | 0.123 | 0.014 | 0.000 |
| 4B RL-full step 10 | T=0.3, k=8, think mode | 0.984 | 0.163 | 1.02 | 35.3 | +0.014 | +0.014 | 0.160 | 0.016 | 0.000 |
| 4B RL-full step 10 | T=0.6, k=8, think mode | 0.951 | 0.150 | 1.12 | 42.8 | +0.003 | +0.008 | 0.145 | 0.032 | 0.000 |
| 4B RL-full step 10 | T=1.0, k=8, think mode | 0.842 | 0.149 | 1.24 | 37.8 | +0.004 | +0.007 | 0.125 | 0.023 | 0.000 |
| 4B RL-full step 20 | T=0.3, k=8, think mode | 0.997 | 0.175 | 1.00 | 29.9 | +0.012 | +0.015 | 0.177 | 0.000 | 0.000 |
| 4B RL-full step 20 | T=0.6, k=8, think mode | 0.984 | 0.168 | 1.05 | 35.7 | +0.006 | +0.014 | 0.175 | 0.003 | 0.000 |
| 4B RL-full step 20 | T=1.0, k=8, think mode | 0.947 | 0.173 | 1.03 | 30.5 | +0.014 | +0.009 | 0.163 | 0.009 | 0.000 |
| 4B RL-full step 30 | T=0.3, k=8, think mode | 0.997 | 0.178 | 1.00 | 30.9 | +0.003 | +0.010 | 0.185 | 0.000 | 0.000 |
| 4B RL-full step 30 | T=0.6, k=8, think mode | 0.985 | 0.180 | 1.02 | 36.0 | +0.005 | +0.011 | 0.181 | 0.000 | 0.000 |
| 4B RL-full step 30 | T=1.0, k=8, think mode | 0.967 | 0.181 | 1.02 | 32.8 | +0.010 | +0.010 | 0.175 | 0.002 | 0.000 |
| 4B RL-full step 40 | T=0.3, k=8, think mode | 0.977 | 0.186 | 1.14 | 35.7 | +0.006 | +0.004 | 0.181 | 0.000 | 0.000 |
| 4B RL-full step 40 | T=0.6, k=8, think mode | 0.977 | 0.189 | 1.02 | 30.8 | +0.008 | +0.005 | 0.181 | 0.000 | 0.000 |
| 4B RL-full step 40 | T=1.0, k=8, think mode | 0.986 | 0.187 | 1.00 | 31.3 | +0.010 | +0.004 | 0.179 | 0.000 | 0.000 |
| 4B RL-full replicate (seed 1) step 40 | T=0.3, k=8, think mode | 0.982 | 0.183 | 1.03 | 40.9 | +0.009 | +0.005 | 0.175 | 0.000 | 0.000 |
| 4B RL-full replicate (seed 1) step 40 | T=0.6, k=8, think mode | 0.993 | 0.183 | 1.00 | 38.6 | +0.011 | +0.007 | 0.180 | 0.002 | 0.000 |
| 4B RL-full replicate (seed 1) step 40 | T=1.0, k=8, think mode | 0.989 | 0.187 | 1.02 | 39.8 | +0.016 | +0.009 | 0.179 | 0.005 | 0.000 |
| 4B RL-full replicate (seed 1) step 50 | T=0.3, k=8, think mode | 1.000 | 0.195 | 1.00 | 32.3 | +0.013 | +0.010 | 0.193 | 0.009 | 0.000 |
| 4B RL-full replicate (seed 1) step 50 | T=0.6, k=8, think mode | 0.997 | 0.193 | 1.00 | 31.6 | +0.011 | +0.011 | 0.192 | 0.009 | 0.000 |
| 4B RL-full replicate (seed 1) step 50 | T=1.0, k=8, think mode | 0.982 | 0.191 | 1.02 | 34.4 | +0.009 | +0.009 | 0.187 | 0.009 | 0.000 |
| 4B RL-full replicate (seed 1) step 60 | T=0.3, k=8, think mode | 1.000 | 0.193 | 1.00 | 42.3 | +0.006 | +0.008 | 0.195 | 0.000 | 0.000 |
| 4B RL-full replicate (seed 1) step 60 | T=0.6, k=8, think mode | 1.000 | 0.193 | 1.00 | 43.0 | +0.005 | +0.008 | 0.194 | 0.000 | 0.000 |
| 4B RL-full replicate (seed 1) step 60 | T=1.0, k=8, think mode | 0.993 | 0.192 | 1.00 | 43.0 | +0.006 | +0.006 | 0.190 | 0.000 | 0.000 |
| 4B A2 (answer-only SFT, epoch 1) | T=1.0, k=8, think mode | 0.556 | 0.156 | 1.54 | 61.6 | +0.020 | +0.005 | 0.079 | 0.004 | 0.000 |
| 4B A2 | T=0.6, k=8, think mode | 0.757 | 0.144 | 1.25 | 55.3 | +0.005 | +0.007 | 0.110 | 0.011 | 0.000 |
| 4B A2 | T=0.3, k=8, think mode | 0.811 | 0.137 | 1.25 | 69.5 | +0.005 | +0.008 | 0.117 | 0.003 | 0.000 |
| 4B RL-full step 40, sampler seed 1 | T=0.6, k=8, think mode | 0.977 | 0.189 | 1.07 | 33.2 | +0.008 | +0.005 | 0.181 | 0.000 | 0.000 |
| 4B RL-full step 40, sampler seed 2 | T=0.6, k=8, think mode | 0.980 | 0.186 | 1.04 | 32.7 | +0.005 | +0.004 | 0.182 | 0.000 | 0.000 |
| 4B RL-contrastive step 10 (run cut short) | T=0.6, k=8, think mode | 0.961 | 0.122 | 1.05 | 27.0 | +0.005 | +0.010 | 0.127 | 0.046 | 0.000 |
| 4B RL-contrastive step 10 (run cut short) | T=1.0, k=8, think mode | 0.885 | 0.123 | 1.08 | 27.0 | +0.005 | +0.008 | 0.115 | 0.045 | 0.000 |
| 4B RL-contrastive step 20 (run cut short) | T=0.6, k=8, think mode | 0.994 | 0.113 | 1.00 | 22.9 | +0.010 | +0.009 | 0.114 | 0.021 | 0.000 |
| 4B RL-contrastive step 20 (run cut short) | T=1.0, k=8, think mode | 0.968 | 0.120 | 1.01 | 24.5 | +0.010 | +0.007 | 0.109 | 0.044 | 0.000 |
| 4B A2, sampler seed 1 | T=1.0, k=8, think mode | 0.550 | 0.151 | 1.57 | 65.2 | +0.011 | +0.001 | 0.075 | 0.004 | 0.000 |
| 4B A2, sampler seed 2 | T=1.0, k=8, think mode | 0.559 | 0.150 | 1.54 | 64.2 | +0.018 | +0.002 | 0.077 | 0.004 | 0.000 |
| 4B A2 shown another drug's profile (shuffled) | T=1.0, k=8, think mode | 0.565 | 0.131 | 1.76 | 69.2 | -0.007 | -0.002 | 0.074 | 0.002 | 0.000 |
| 4B A2 with a blank profile | T=1.0, k=8, think mode | 0.721 | 0.168 | 1.00 | 28.0 | +0.000 | +0.000 | 0.114 | 0.000 | 0.000 |
| 4B RL-full step 40 shown another drug's profile (shuffled) | T=0.6, k=8, think mode | 0.977 | 0.183 | 1.00 | 30.1 | +0.002 | -0.004 | 0.173 | 0.000 | 0.000 |
| 4B RL-full step 40 with a blank profile | T=0.6, k=8, think mode | 1.000 | 0.089 | 1.00 | 104.0 | +0.000 | +0.000 | 0.165 | 0.000 | 0.000 |
| 4B A1 (trace SFT, epoch 2) | T=1.0, k=4, think mode, 8192 tokens | 0.328 | n/a | n/a | n/a | n/a | +0.003 | 0.048 | 0.000 | 0.000 |
| 4B untrained | T=0.6, k=8 | 0.254 | 0.099 | 4.34 | 351.5 | +0.011 | +0.008 | 0.042 | 0.013 | 0.000 |
| 4B untrained | T=0.3, k=8 | 0.268 | 0.097 | 4.81 | 534.0 | +0.027 | +0.013 | 0.048 | 0.020 | 0.000 |
| 9B untrained | T=1.0, k=8 | 0.417 | 0.159 | 2.34 | 155.8 | +0.020 | +0.009 | 0.069 | 0.019 | 0.000 |
| 9B untrained | T=0.6, k=8 | 0.629 | 0.166 | 1.77 | 91.5 | +0.028 | +0.015 | 0.108 | 0.013 | 0.000 |
| 9B untrained | T=0.3, k=8 | 0.564 | 0.161 | 2.04 | 169.2 | +0.024 | +0.015 | 0.099 | 0.020 | 0.000 |
| Qwen3.5-27B-FP8 untrained | T=1.0, k=8 | 0.673 | 0.186 | 1.52 | 99.9 | +0.045 | +0.027 | 0.121 | 0.048 | 0.000 |
| Qwen3.8-27B-FP8 untrained | T=1.0, k=4 | 0.615 | n/a | n/a | n/a | n/a | +0.019 | 0.103 | 0.022 | 0.000 |
| 4B told the drug name | T=1.0, k=8 | 0.328 | 0.141 | 3.25 | 157.6 | +0.009 | +0.002 | 0.043 | 0.000 | 0.000 |
| 9B told the drug name | T=1.0, k=8 | 0.421 | 0.155 | 2.24 | 89.8 | +0.010 | +0.008 | 0.069 | 0.021 | 0.000 |
| Qwen3.5-27B-FP8 told the drug name | T=1.0, k=2 | 0.540 | n/a | n/a | n/a | n/a | +0.025 | 0.096 | 0.000 | 0.000 |
| Qwen3.8-27B-FP8 told the drug name | T=1.0, k=2 | 0.596 | n/a | n/a | n/a | n/a | +0.009 | 0.097 | 0.025 | 0.000 |
| 4B A2 near-greedy | T=0.01, k=1, think mode | 0.750 | n/a | n/a | n/a | n/a | +0.012 | 0.128 | 0.000 | 0.000 |
| 9B untrained near-greedy | T=0.01, k=1 | 0.600 | n/a | n/a | n/a | n/a | +0.008 | 0.098 | 0.013 | 0.000 |
| 4B untrained near-greedy | T=0.01, k=1 | 0.279 | n/a | n/a | n/a | n/a | +0.005 | 0.044 | 0.000 | 0.000 |

Val split, 20 drugs. Similarity is the Tanimoto similarity (Morgan count vectors) to the reference structure. Delivered similarity and lift are those of the first valid, finished answer among up to eight samples of each prompt (0 if none), with its cost in samples and generated tokens per answer (prompts are not counted). Lift is the similarity to the drug's own structure minus the mean similarity to the other drugs' structures. Core match and exact match are the share of samples with the reference's scaffold or connectivity (core) or its connectivity (exact), averaged over drugs. n/a: fewer than eight samples per prompt, so no first-valid value.
