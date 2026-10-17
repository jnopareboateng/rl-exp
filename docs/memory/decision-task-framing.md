---
name: decision-task-framing
description: 'SFT answers equal the real approved drug (98-99%): task is structure recovery, scored exactly'
type: decision
updated: 2026-10-17
---
The task is structure recovery from a pharmacology profile. The SFT answer's connectivity equals the ChEMBL structure of the named drug in 98-99% of the 942 rows whose name resolved (94-95% of all 980 single-drug rows); stereo-exact in about 89%.
References come from ChEMBL and the SFT answers, not from parsing the prompt.

**Why:** measured with scripts/resolve_refs.py. The reference traces were written with the answer known, so they are rationalisations, not evidence of deduction. A keyword survey finds no formula, MW, logP, TPSA or HBD/HBA term in 75-93% of prompts.
**How to apply:** score against the known structure (connectivity primary, stereo reported); do not build a constraint-satisfaction reward; do not read SFT traces as proof that a model can deduce structures.
