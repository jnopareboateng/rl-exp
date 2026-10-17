# frontier-rl: survey of frontier-lab RL stacks (2026-10-05)

Twelve parallel research agents (Sonnet), one per track. Each cloned the relevant repos and read
loss files, rollout loops, launch scripts and configs, then checked them against papers and model
cards. Every claim in the track files carries an evidence tag (read-in-code / paper-summarized /
secondary / inferred), and each track lists what it could not read.

Total: ~78,000 words across 12 files in `tracks/`.

## Tracks

| File | Scope |
|---|---|
| `tracks/01-moonshot-kimi.md` | k1.5 → Kimi-Researcher → K2 → K2.5 → K3; checkpoint-engine weight-broadcast code walk |
| `tracks/02-thudm-slime-glm.md` | slime read at HEAD; GLM-4.5 → 5.3 recipes; SAO; how to plug a custom oracle in |
| `tracks/03-deepseek.md` | GRPO origins, R1, V3.2 stabilizers, Math-V2 verifier/meta-verifier, Prover, V4 |
| `tracks/04-alibaba-qwen.md` | GSPO + SAPO, Qwen pipelines, Qwen3-Coder-Next, Tongyi DeepResearch, ROLL, ArenaRL |
| `tracks/05-bytedance-seed-verl.md` | verl architecture (V1 trainer, agent loop, reward loop); VC-PPO → VAPO → DAPO |
| `tracks/06-open-frameworks.md` | 19 frameworks compared; slow-oracle fit scored; recommendation for bio |
| `tracks/07-western-labs.md` | OpenAI, Anthropic, GDM, Meta, NVIDIA, AI2, Mistral, xAI, Microsoft |
| `tracks/08-deep-research-agents.md` | 11 agent-RL repos; observation masking, credit assignment, rubric rewards |
| `tracks/09-other-global-labs.md` | MiniMax, Ant, StepFun, Meituan, Xiaomi, Kuaishou, Shanghai AI Lab, Tencent, Baidu + more |
| `tracks/10-algorithms-rewards.md` | Objectives side by side; reward design; ScaleRL; pass@k debate; reward hacking |
| `tracks/11-rl-infra.md` | Rollout engines, weight sync, async/staleness, mismatch, MoE, sandbox fleets |
| `tracks/12-rl-for-biology-sota.md` | What RL in biology/chemistry exists today, plus the technique × domain gap matrix |
| `tracks/13-xiaomi-mimo-verl.md` | Addendum: XiaomiMiMo/verl — MiMo-V2.6 agentic RL code; credit assignment and reward hygiene |

## Synthesis

Published as an HTML report (see the thread for the link). Its sections: the convergent recipe,
lab-by-lab records, objectives side by side, the five recurring systems problems, framework
selection for expensive oracles, long-horizon agent RL, the state of RL in biology with the gap
matrix, and twelve ranked openings for transfer.

## Headline findings

1. The objective converged: critic-free, group-baseline policy gradient, little or no KL,
   clip-higher ~0.28, token-level loss, zero-variance groups dropped.
2. Moonshot is the exception — a squared-log-ratio mirror-descent surrogate, never PPO/GRPO.
3. Train/inference mismatch is the dominant engineering problem; every stack ships corrections.
4. Asynchrony is default; staleness is an explicitly budgeted quantity.
5. Environment supply, not compute, is the bottleneck; labs synthesize environments at scale.
6. Verifiers are being replaced by trained judges with meta-verifier audits.
7. Consolidation moved from mixed-domain RL to on-policy distillation of specialists.

## Caveats

- Paper figures were mostly obtained through a summarizing fetch tool; exact numbers are
  as-reported, not independently re-derived.
- Several pages would not render and several arXiv fetches were rate-limited; search budgets ran
  out on tracks 07, 09 and 12.
- Gap-matrix "gap" cells mean nothing was found, not that nothing exists.
