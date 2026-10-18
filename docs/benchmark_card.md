# Benchmark card: an approved drug's structure from its pharmacology profile

This card describes the release tables that `scripts/make_release.py` builds in `release/`, which are published with the repository: reference structures and splits for 351 drugs, control answers, and the study's reference numbers (splits.csv, controls.csv,
values.csv, and a README with checksums). The profile texts belong to an in-house dataset and are not released. The protocol is frozen by the git tag prereg-v1 (2026-10-18).
The release tables hold no model score; the Test values of the controls are in section 6 and the full Test results are in the paper (`results/`, `docs/paper/draft.md`).

## 1. The task
A system reads a written pharmacology profile of one approved drug, with the drug's name withheld, and answers with the drug's structure as a SMILES string (a one-line text code for a molecule).
RDKit reads the answer, which is scored against the known structure in three ways: the same atom connectivity (exact), the same ring scaffold (core), and fingerprint similarity
(Tanimoto on Morgan count fingerprints). The study asks whether reinforcement learning (RL) beats cheaper alternatives. The tables are also useful without it, as references and controls
for any similarity-scored structure task.

## 2. Why it is hard to evaluate
- **A constant answer scores well.** One fixed molecule given for every drug delivers a similarity of 0.176 on the 20 Val drugs. The best constant chosen on the training drugs plus a public
  pool of approved drugs delivers 0.187, and the best constant that could have been picked with Val in view delivers 0.195. The Val-selected RL checkpoint (Qwen3.5-4B) also delivers 0.195:
  at the ceiling, not above it.
- **The constant's level is cohort-dependent.** The reward-optimal constant (C3) delivers 0.187 on Val, 0.199 on Test-T (the best of 1,828 candidate constants there) and 0.159 on Test-R (21st of 1,828; the best is 0.171),
  so 0.199 is not a fixed bar. On the temporal cohort (Test-T) the RL checkpoint delivers 0.186 against that 0.199 (the registered contrast does not reject, p = 0.061); on Test-R it delivers 0.168 against 0.159 (p = 0.28).
- **The similarity scale is compressed.** Random pairs of approved drugs average 0.141 (90th percentile 0.213, 99th 0.274), so 0.187 already sits at the 79th percentile of random pairs
  and 0.195 at the 83rd.
- **Nothing recovers a structure.** Exact match is 0 for every system on every split, and the core match (exact or the same ring scaffold) is at most 5% on Val and at most 0.3% on the two Test splits.
- So similarity alone cannot tell a system that reads the profile from one that learned a good generic molecule. The controls come first (docs/controls_checklist.md).

## 3. Data sources and licences
| Part | Source | Terms (checked 2026-10-18) |
|---|---|---|
| Reference structures, 333 of 351 drugs | ChEMBL parent structure found by drug name, cross-checked against the in-house dataset's answers | ChEMBL: CC BY-SA 3.0 (attribution, share-alike) |
| Reference structures, 18 of 351 | Resolved by hand from PubChem; metal complexes use the organic ligand (6 drugs) | NCBI places no restriction; submitters may claim rights |
| Approval years | ChEMBL first approval (285 drugs), openFDA (6), an audit of regulator pages for 43 candidates (42), hand dating (18) | ChEMBL as above; openFDA CC0; regulator pages: dates are facts, the page is named in the audit |
| PubChem record year and PubMed count | NCBI, matched by name and structure | NCBI: no restriction on use or distribution |
| Control C3 pool | ChEMBL_37, released 2026-05-01, 1,587 approved drugs; not copied | as ChEMBL |
| Profile texts, traces, samples, adapters | In-house dataset, 1,146 rows | not released; its licence is not recorded |

The release tables are licensed CC BY-SA 4.0: they adapt ChEMBL data (CC BY-SA 3.0), and that licence allows a later version for adaptations. Attribute ChEMBL when you use them. The code is MIT and the text CC BY 4.0.
The ChEMBL release behind the per-drug reference table was not recorded (only the C3 pool, ChEMBL_37). Not checked: the individual regulator pages beyond the fact that approval dates are facts, and the provenance and licence of the in-house dataset.

## 4. Splits
| Split | Drugs | Rows | Components | Rule |
|---|---|---|---|---|
| Train | 241 | 687 | 218 | the rest |
| Val | 20 | 58 | 16 | whole components, seeded draw |
| Test-R | 30 | 80 | 27 | whole components, seeded draw |
| Test-T | 41 | 100 | 41 | first approved anywhere in 2025 or 2026 |
| Buffer | 19 | 49 | 8 | other drugs in a Test-T drug's component; used nowhere |
| Excluded | 3 | 6 | | no usable reference (two unresolvable structures, one the answer reader cannot recover) |
- **Component rule.** Two drugs join one component if their bit Tanimoto is 0.4 or more or they share a ring scaffold, and a whole component goes to one split. No pair across Train, Val,
  Test-R and Test-T reaches 0.4 (largest 0.398) and none shares a scaffold.
- **Date rule.** The approval year comes from hand dating if there is one, else from the audit (the earliest approval by a listed regulator), else from ChEMBL or openFDA. The audit checked 43 drugs
  with a first US approval in 2025 or 2026; five had earlier approvals in Japan or China and left, and three more were dated by hand. The 41 Test-T drugs lie in 41 components, so each is its own unit.

## 5. Endpoint and cost
The one confirmatory number is the delivered similarity: for each profile, the similarity of the first valid, finished answer among up to 8 tries (0 if none), averaged per drug and then over drugs.
It is always read with its cost: mean tries per answer and mean generated tokens per answer. An answer cut off by the length cap is a failure. A constant costs 1 try and 0 tokens.
Exact and scaffold matches are reported but too rare to separate systems at these sizes. Statistics use components as the units: an exact sign-flip test, with bootstrap intervals as description.

## 6. Controls and their values (95% intervals)
| Control | Answer | Val (20 drugs) | Test-T (41 drugs) | Test-R (30 drugs) |
|---|---|---|---|---|
| C0 | the Train drug most similar on average to the other Train drugs (CHEMBL2105719) | 0.176 [0.145, 0.212] | 0.184 [0.169, 0.198] | 0.160 [0.134, 0.188] |
| C1 | the structure of the Train drug with the nearest profile text | 0.174 [0.152, 0.198] | 0.168 [0.151, 0.184] | 0.166 [0.143, 0.187] |
| C2 | the shortest structure among the most common Train ring scaffold (CHEMBL207538) | 0.076 [0.048, 0.125] | 0.077 [0.067, 0.088] | 0.078 [0.064, 0.093] |
| C3 | the best constant over Train plus the ChEMBL pool (CHEMBL941) | 0.187 [0.151, 0.229] | 0.199 [0.181, 0.216] | 0.159 [0.135, 0.187] |

The best constant with the split in view is 0.195 (Val), 0.199 (Test-T) and 0.171 (Test-R). The Test columns are the paper's locked-run values (results/test_t_baselines.json, results/test_r_baselines.json). Answers and rules are in release/controls.csv, numbers and their source files in release/values.csv. C1 needs the in-house profile text,
so only its rule is released.

## 7. What is not released, and why
Profile prompts, reasoning traces, model samples and trained adapters are not released: they come from an in-house dataset whose provenance and terms are not documented, and whether adapters trained on it can be shared is undecided.
The C3 pool is not copied; controls.csv records its checksum and release/README.md gives the rebuild commands. Model scores on Test-R and Test-T are reported in the paper, not in these tables.

## 8. Intended and out-of-scope uses
- Intended: calibrating and auditing similarity-scored structure tasks; reproducing the splits and control answers; comparing new systems on Val and on the date cohort against the controls.
- Out of scope: drug discovery or design claims; any clinical or medical use; judging whether a model has seen a molecule (approval date is not disclosure date); training on Test-R or
  Test-T labels, which are public here; reporting similarity without the constant and the cost beside it.

## 9. Known limitations
- Val has 20 drugs in 16 components, so intervals are wide (the constant's own is 0.07 to 0.08 wide). Test-T has 41. All evidence is one in-house dataset in four profile formats, and its
  traces were written with the answer known.
- Approval date is not disclosure date: 40 of the 41 Test-T drugs have a PubChem record created 3 to 20 years before approval (median 10.5), so Test-T is a temporal cohort, not unseen molecules.
- The Test run covers 4B models only; there is one training seed per arm (two RL runs); the RL checkpoint was picked from 21 Val candidates.
- Reference quirks: two combination products are represented by one component and carry that component's approval year; two radiolabelled DOTATATE drugs share one structure;
  6 of the 351 drugs (4 in the used splits) are metal or radiolabel complexes and use the organic ligand; 5 references have a recorded alternate tautomer; stereochemistry is reported but exact match ignores it. Eight ChEMBL ids have
  no ChEMBL structure to cross-check, and one id (gallium ga 68 dotatate) is withheld because ChEMBL's structure for it is another compound.
- PubChem and PubMed fields are proxies: 6 drugs have no CID (the reference is a free ligand); for 9 the CID differs from the reference in the stereo or isotope layer; PubMed counts for
  36 drugs (salt words dropped from the name) and 8 radiolabelled names may undercount (data/curated/exposure_notes.md).

## 10. How to cite
No paper or DOI exists yet. Until it does, cite the repository (jnopareboateng/rl-exp), the tag prereg-v1 and the checksums in release/README.md. Working title of the paper:
"Reinforcement learning did not beat a tuned constant answer at predicting the structures of newly approved drugs from their pharmacology profiles". Author and year fields will be added with the paper.

## 11. How to rerun the scoring
Setup: `uv venv && uv pip install -e ".[dev]"`. Rebuild the release (needs the repository's curated files, its results and the ChEMBL pool): `python scripts/make_release.py`.
With the release tables alone you can reproduce the control values and score your own answers drug by drug:
```python
import csv; from rlexp import chem
rows = list(csv.DictReader(open("release/splits.csv"))); c3 = next(r for r in csv.DictReader(open("release/controls.csv")) if r["control"] == "C3")["smiles"]
val = [chem.standardize(r["smiles"]) for r in rows if r["split"] == "val"]
print(sum(chem.tanimoto(chem.standardize(c3), m) for m in val) / len(val))   # 0.1865, the C3 row above
```
The full metrics (first-valid similarity with cost, bootstrap, sign-flip test) need the in-house row files: `python -m rlexp.metrics score`, then `python -m rlexp.metrics report`; see docs/contracts.md.
