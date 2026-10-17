---
name: gotcha-blank-profile-shared-draws
description: 'Blank-profile runs give every row one prompt, so draws repeat across rows; read mean_tanimoto'
type: gotcha
updated: 2026-10-17
---
The blank-profile control (`rlexp.arms blank`, profile replaced by "Drug: unknown.") yields one identical prompt for every row (174 characters on Val, Test-R and Test-T). With the sampler's seeded draws, sample j is then the same molecule on every row, so the run holds only k distinct answers. Its fv_tanimoto is the Tanimoto of ONE shared first answer (A2 blank 0.168, RL-full step 40 blank 0.089 on Val, both with fv_tokens degenerate and fv_samples 1.000) and mostly reflects which molecule that draw was.

**Why:** seen on Val 2026-10-17: RL-full blank fv_tanimoto 0.089 against mean_tanimoto 0.165 over the same 464 samples; the bootstrap interval treats the 20 drugs as independent draws, which they are not here.
**How to apply:** report blank-profile runs by mean_tanimoto and validity (all k draws), never by fv_tanimoto; lift is 0 by construction; it is a description of the model's unconditional answer, not a test.

Related: [[decision-endpoint-fv-tanimoto]]
