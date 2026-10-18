---
name: gotcha-approval-audit-wording
description: 'approval_audit_notes.md says 'not seen' but PubChem records predate approval by years'
type: gotcha
updated: 2026-10-18
---
`data/curated/approval_audit_notes.md` lists "structure disclosed before approval: not seen" for all 43 audited drugs. That records only that no regulator page showed a disclosure; PubChem records of 40 of the 41 Test-T drugs were created 3 to 20 years before approval (median 10.5), so Test-T is a temporal cohort, not a set of unseen molecules.

**Why:** quoting the notes as evidence of non-disclosure would be wrong, and `data/` is not edited after the tag.
**How to apply:** cite the exposure fields (`pubchem_create_year`, `pubmed_count` in `refs.jsonl` and `release/splits.csv`) and the paper's Section 1, never the audit notes, for what was public before approval.
