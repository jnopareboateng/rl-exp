---
name: decision-no-profile-scale-up
description: 'Synthetic profiles for more training drugs are deprioritised: class facts add little over a constant'
type: decision
updated: 2026-10-17
---
Two retrieval rules (1-NN and 5-medoid) over coarsened ChEMBL class facts reach mean Tanimoto 0.201 against 0.169 for the best constant when only direct pairs are excluded (+0.032), but 0.163 (-0.006, interval up to +0.001) once the target's whole connected component is excluded as the real split does (Train targets -0.014). More neighbours raise only the oracle (0.30 to 0.35). Generating profiles for more training drugs was therefore deprioritised.

**Why:** this does NOT bound learned predictors, richer facts or free-text profiles, and failing to reject zero is not equivalence (pairwise Train interval up to +0.014); the nearest-profile copy control still shows lift +0.022. It only says the cheap structured-fact signal is small. Source: results/class_ceiling.json, scripts/class_ceiling.py.
**How to apply:** do not spend generation budget on more profiles until a free-text or learned predictor shows more than the TF-IDF control; scale model size (zero-RL sweep) first; never write that a profile-specific signal "requires recall of the drug".

Related: [[gotcha-core-endpoint-low-power]]
