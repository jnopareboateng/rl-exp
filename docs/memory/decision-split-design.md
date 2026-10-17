---
name: decision-split-design
description: 'Splits by components (Tanimoto>=0.4 or same scaffold); Test-T = approved 2025-26, a temporal cohort'
type: decision
updated: 2026-10-17
---
Drugs form components when Tanimoto >= 0.4 or they share an identical non-empty Murcko scaffold; whole components go to one split. Test-T = first approved anywhere in 2025-2026: 41 single drugs, 100 rows (26 in 2025, 15 in 2026), after an audit of 43 candidates of which 5 had earlier approvals in Japan or China. The 41 lie in 41 different components (8 of them also hold the 19 buffer drugs), so each Test-T drug is its own exchangeable unit. Test-R is 30 drugs (80 rows), Val 20 (58), Train 241 (687), 3 excluded.

**Why:** approval date is not disclosure date: PubChem records of 40 of the 41 cohort drugs were created 3 to 20 years before approval (median 10.5), so Test-T is a temporal cohort, not proof the molecule is unseen. PubChem creation year and PubMed count are recorded as exposure covariates.
**How to apply:** report Test-T and Test-R separately and never pooled; stratify Test-T by exposure; never call Test-T recall-proof or unseen; count components, not drugs, when a split holds related drugs (Val, Test-R).
