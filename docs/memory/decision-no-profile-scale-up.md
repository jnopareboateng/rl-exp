---
name: decision-no-profile-scale-up
description: 'No synthetic profiles for more training drugs: class facts predict structure like a constant'
type: decision
updated: 2026-10-17
---
The class-information ceiling (ChEMBL_37, 1,786 drugs) shows that nearest-neighbour prediction from ATC, mechanism, target and indication reaches mean Tanimoto 0.201 against 0.169 for a constant once neighbours respect the split rule (+0.032; +0.001 for Train-like drugs, +0.002 for drugs approved from 2015). More neighbours raise only the oracle (0.30 to 0.35). Scaling training drugs with LLM-written profiles was therefore dropped.

**Why:** a held-out drug has no close analogue by construction, and class-level information does not carry structure beyond a constant; profile-specific similarity needs recall of the specific drug (identification then SMILES). Source: results/class_ceiling.json, scripts/class_ceiling.py.
**How to apply:** do not spend generation budget on more profiles; scale model size (zero-RL sweep) rather than data; revisit only if free-text profiles are shown to carry structural cues beyond the structured facts.

Related: [[gotcha-core-endpoint-low-power]]
