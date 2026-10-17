---
name: gotcha-answer-extractor
description: 'extract_answer misfires in 11 of 77 fixtures and cannot read SMILES under 8 characters'
type: gotcha
updated: 2026-10-17
---
rlexp.chem.extract_answer (last SMILES-like token after the last </think>) returns a wrong molecule instead of nothing in 11 of 77 fixture cases (token in a closing sentence, SMILES split across lines, trailing byproduct), and cannot extract a SMILES shorter than 8 characters or without a ring, branch, bond or digit character.

**Why:** found by tests/fixtures/answer_formats.json; references the extractor cannot recover from itself are excluded (unextractable_reference).
**How to apply:** on real rollouts, measure disagreement with a "last SMILES-only line" rule before prereg-v1 and switch if it exceeds 1%.
