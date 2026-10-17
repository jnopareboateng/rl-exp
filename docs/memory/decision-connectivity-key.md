---
name: decision-connectivity-key
description: 'Exact match = strict InChIKey first block plus curated alternates; looser keys equate real isomers'
type: decision
updated: 2026-10-17
---
Exact match is judged on `chem.conn`, the first block of the standard InChIKey, plus `ref_alt_smiles`: other human-curated registry forms of the same compound with an equal `chem.skeleton`. `chem.inchikey` (full key) feeds stereo_match, reported and never rewarded.
Supersedes: tautomer-tolerant skeleton key (2026-10-17).

**Why:** the skeleton key equated delta-8 and delta-9 THC and norethindrone and norethynodrel (a reviewer's counterexamples, reward 1.0), while five approved drugs differ from ChEMBL only in hydrogen placement; curated alternates fix those five without merging real isomers. RDKit's canonical tautomer was tested and rejected: slow (up to 5.6 s per large molecule) and version-sensitive.
**How to apply:** compare molecules only through chem.conn plus the row's alternates; never add a model output to ref_alt_smiles; re-derive keys from ref_smiles in the scoring process.

Related: [[decision-reward-v2]]
