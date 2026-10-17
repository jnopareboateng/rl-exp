---
name: decision-connectivity-key
description: 'conn = formula + heavy-atom skeleton hash (tautomer-tolerant); stereo_match uses the full InChIKey'
type: decision
updated: 2026-10-17
---
Exact match is judged on `chem.conn`: a hash of the InChI formula and heavy-atom skeleton, ignoring hydrogen placement, charge, stereo and isotopes. `chem.inchikey` (full standard InChIKey) feeds stereo_match, which is reported and never rewarded.

**Why:** five approved drugs (nintedanib, omadacycline, eravacycline, copanlisib, sarecycline) differ from their ChEMBL entry only in hydrogen placement; the InChIKey first block would score a correct answer in the other tautomer as a miss. The skeleton key unified all five and merged no two different drugs among 348 references.
**How to apply:** compare molecules only through chem.conn; never parse an InChIKey by hand; report stereo-exact separately.

Related: [[decision-reward-v2]]
