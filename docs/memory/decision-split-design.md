---
name: decision-split-design
description: 'Splits by components (Tanimoto>=0.4 or same scaffold); Test-T = approved 2025-26, a temporal cohort'
type: decision
updated: 2026-10-17
---
Drugs form components when Tanimoto >= 0.4 or they share an identical non-empty Murcko scaffold; whole components go to one split. Test-T = first approved 2025-2026 (43 single drugs, 108 rows); neighbours of Test-T go to a buffer; Test-R is 30 drugs, Val 20, Train the rest.

**Why:** approval date is not disclosure date (a structure can be public years earlier), so Test-T is a temporal cohort, not proof the molecule is unseen. PubChem creation year and PubMed count are recorded as exposure covariates.
**How to apply:** report Test-T and Test-R separately and never pooled; stratify Test-T by exposure; do not call it recall-proof.
