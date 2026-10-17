---
name: gotcha-answer-extractor
description: 'extract_answer misfires in 11 of 77 fixtures; 2.2% of Val samples differ from the strict rule'
type: gotcha
updated: 2026-10-17
---
rlexp.chem.extract_answer (last SMILES-like token after the last </think>) returns a wrong molecule instead of nothing in 11 of 77 fixture cases (token in a closing sentence, SMILES split across lines, trailing byproduct), and cannot extract a SMILES shorter than 8 characters or without a ring, branch, bond or digit character.
On 10,304 real Val samples (results/extractor_audit.json) 2.2% differ from the plan's "last SMILES-only line" rule, above its 1% switch threshold. All 227 are in the 23 non-RL runs (3.0% of 7,520; 3.2% of the finished): 197 are tokens this reader takes from lines with other words, 26 are tiny answers only the strict rule reads, 4 (0.05%) are different molecules. A2 at T=1.0: 0.6%; untrained 4B 7-9%; A1 13%; the 6 RL-full checkpoint runs (2,784 samples): none.

**Why:** found by tests/fixtures/answer_formats.json and scripts/offline_audit.py. References the extractor cannot recover from itself are excluded (unextractable_reference).
**How to apply:** the keep-or-switch decision is a freeze item (docs/prereg-v1.md section 8). Switching drops answers written inside a sentence (weak arms lose validity); keeping accepts the 4 conflicts. Either way change docs/contracts.md, rlexp.chem and its tests in one commit, then rerun scripts/offline_audit.py.
