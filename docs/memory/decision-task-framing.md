---
name: decision-task-framing
description: 'SFT answers equal the real approved drug (98-99%): task is structure recovery, scored exactly'
type: decision
updated: 2026-10-17
---
The task is structure recovery from a pharmacology profile. Row level, strict connectivity key (first InChIKey block): the SFT answer equals the ChEMBL structure of the named drug in 98-99% of the 942 rows whose name resolved and answer parsed (94-95% of all 980 single-drug rows). Stereo, same rows: full InChIKey equal in 94% (888), identical canonical isomeric SMILES in 89% (842).
`split_report.json` `sft_agrees` (338 of 339) is a different count: per drug, tolerant rule (equal connectivity or equal formula and heavy-atom skeleton, so tautomers agree). The ChEMBL release behind `data/curated/chembl_refs.json` was not recorded (the later pool used ChEMBL_37). References come from ChEMBL and the SFT answers, not from parsing the prompt.

**Why:** measured with scripts/resolve_refs.py (connectivity) and recomputed with its recipe for stereo. The traces were written with the answer known, so they are rationalisations, not evidence of deduction. A keyword survey finds no formula, MW, logP, TPSA or HBD/HBA term in 75-93% of prompts.
**How to apply:** score against the known structure (connectivity primary, stereo reported); quote the 98-99% as row-level and never as the drug-level figure; do not build a constraint-satisfaction reward; do not read SFT traces as proof that a model can deduce structures.
