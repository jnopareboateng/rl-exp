# data/

What is in this folder, where it came from and under what terms. Counts and rules are in `docs/benchmark_card.md`.

## Published here (CC BY-SA 4.0)
- `curated/`: the lookups and hand-resolved facts that the reference and split tables are built from, with the notes that explain them: `chembl_refs.json` (ChEMBL parent structure found by drug name),
  `manual_refs.json` (18 structures resolved by hand from PubChem), `approval_audit.json` (first approvals by regulator for the 43 candidates of the temporal cohort),
  `exposure.json` (PubChem record year and PubMed counts) and the `*_notes.md` files.
- `processed/refs.jsonl`: one row per drug group: ids, split, reference structure, approval year and source, exposure fields.
- `processed/split_report.json`: how the structure components and the splits were built, and their sizes.
- `../release/`: the same facts as three CSV files (splits, control answers, reference values) with a README and checksums.
- `MANIFEST.sha256`: hashes of every file that the pre-registration tags freeze, including the private files that are not in this repository.

## Not published
- `raw/`: the private dataset (1,146 rows, each a profile prompt, a reasoning trace and a final answer) and a ChEMBL download cache.
- `processed/`: the prompt files built from the private dataset (`train`, `val`, `test_r`, `test_t` and `arms/`).

They stay on the authors' machine. Without them you can still reproduce the control values and score your own answers against the published splits (`docs/benchmark_card.md`, section 11); the full pipeline
(`README.md`, Reproduce) needs them.

## Terms of the sources (checked 2026-10-18)
| Source | Used for | Terms |
|---|---|---|
| ChEMBL | reference structures, ChEMBL ids, first approvals, the control pool | CC BY-SA 3.0 Unported: attribution and share-alike. That licence allows a later version for adaptations, so the tables here are CC BY-SA 4.0. Cite the current ChEMBL paper. |
| openFDA | six first-marketing dates | CC0 1.0 (open.fda.gov/license) |
| PubChem and PubMed (NCBI) | record creation year, PubMed counts, 18 structures resolved by hand | NCBI places no restriction on use or distribution; submitters may claim rights, which NCBI does not assess (ncbi.nlm.nih.gov/home/about/policies) |
| Regulator pages (FDA, EMA, MHRA, PMDA, NMPA) | approval dates for 43 candidates | dates are facts; `approval_audit.json` names the page consulted for each |

## One caution
`curated/approval_audit_notes.md` lists "structure disclosed before approval: not seen" for the 43 audited drugs. That records only that no regulator page showed a disclosure. PubChem records of 40 of the 41
temporal-cohort drugs were created 3 to 20 years before approval (median 10.5), so read the exposure fields and Section 1 of the paper, not that note. The file is left as it was because the pre-registration manifest hashes it.
