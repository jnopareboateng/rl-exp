---
name: gotcha-base-model-verbosity
description: 'Qwen3.5-4B rambles under the dataset's xhigh prompt (0% valid); a direct-answer prompt works'
type: gotcha
updated: 2026-10-17
---
With the dataset's system prompt ("Reasoning effort is set to xhigh ...") Qwen3.5-4B never answers briefly: 99.8% of no-think samples hit a 512-token cap with no SMILES, and in think mode only 8% close their reasoning within 8,192 tokens (0% valid answers). With the system turn replaced by "Reply with only the SMILES string of the drug described. No explanation." 82% finish within 256 tokens, 17% are valid SMILES, core@1 is under 1% and exact@1 is 0 on Val (20 drugs, k=8).

**Why:** measured in the first A100 session (aggregate metrics in results/). The untrained model has almost no knowledge of these structures, so the first lift must come from teaching the output format.
**How to apply:** answer-only arms use the direct-answer system text (rlexp.arms.DIRECT_SYSTEM); report the untrained model under both prompts; do not budget thinking-mode runs at 8k tokens without a trained model.

Related: [[decision-controls-first-design]]
