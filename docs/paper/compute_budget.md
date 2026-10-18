| Stage | Runs or steps | GPU hours | Note |
|---|---|---|---|
| SFT A2 (answer-only, 3 epochs) | 129 steps | 0.15 | SFT; status complete |
| SFT A1 (reasoning traces, 2 epochs) | 84 steps | 0.42 | SFT; status complete |
| RL-full pilot R1 (40 steps) | 40 steps | 0.80 | RL; status complete; includes 3 min of Val evaluations |
| RL-contrastive R3 (cut by a runtime loss) | 29 steps | 0.59 | RL; last record 'running': the runtime was lost; includes 4 min of Val evaluations |
| RL-full replicate R5 (seed 1) | 63 steps | 1.46 | RL; status stopped_collapse; includes 10 min of Val evaluations |
| RL-validity R2 | 40 steps | 0.73 | RL; status complete; includes 6 min of Val evaluations |
| 27B untrained (Qwen3.5 and Qwen3.8) | 2 runs | 0.57 | vLLM; 0.47 h of it generation |
| 4B untrained and prompt probes | 11 runs | 0.57 | vLLM; 0.43 h of it generation |
| 9B untrained | 5 runs | 0.22 | vLLM; 0.10 h of it generation |
| A1 Val sampling | 1 runs | 0.36 | vLLM; 0.52 h of it generation |
| A2 Val sampling | 6 runs | 0.16 | vLLM; 0.09 h of it generation |
| A4 Val sampling | 3 runs | 0.17 | vLLM; 0.13 h of it generation |
| Diagnostics: identification and name-given | 8 runs | 0.34 | vLLM; 0.15 h of it generation |
| Diagnostics: shuffled and blank profile | 6 runs | 0.15 | vLLM; 0.04 h of it generation |
| RL checkpoints: Val re-sampling | 21 runs | 0.44 | vLLM; 0.19 h of it generation |
| **Total logged** | | **7.12** | training 4.15 h, sampling 2.97 h |

Colab runtime sessions (as of 2026-10-18T02:00:00Z):

| Runtime | GPU | Hours | How it ended |
|---|---|---|---|
| 2026-10-17T12:10:00Z to 2026-10-17T20:35:00Z | Colab A100-SXM4-40GB (runtime 1) | 8.42 | disappeared; adapters and Val evaluations were on Drive, resumable RL checkpoints were not |
| 2026-10-17T20:57:00Z to running | Colab A100-SXM4-40GB (runtime 2) | 5.05 | rebuilt from the bootstrap bundle; running |
| quota to quota | Kaggle T4 quota | 0.29 | a 14-minute probe session billed before it was cancelled, plus one 3.3-minute smoke run |

Run directories without a meta.json, not counted (20): colab-s1/a1-val-partial, colab-s1/rlfull-s20-t0.6, colab-s1/rlfull-s20-t1.0, colab-s1/rlfull-s30-t0.6, colab-s1/rlfull-s30-t1.0, colab-s1/rlfull-s40-t0.6, colab-s1/rlfull-s40-t1.0, colab-s1/strict-a-direct, colab-s1/strict-a2-val, colab-s1/strict-a4-icl, colab-s1/strict-b27-direct, colab-s1/strict-b9-t0.6, colab-s1/strict-rlfull-s40-t0.6, colab-s1/val-control_constant, colab-s1/val-control_constant_scaffold, colab-s1/val-control_copy_nearest, colab-s1/val8-control_constant, colab-s1/val8-control_constant_pool, colab-s1/val8-control_constant_scaffold, colab-s1/val8-control_copy_nearest
