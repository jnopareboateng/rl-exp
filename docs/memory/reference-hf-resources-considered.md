---
name: reference-hf-resources-considered
description: 'microsoft/msr-acc, HuggingFaceBio, BioMysteryBench reviewed 2026-10-17: peripheral, not used'
type: reference
updated: 2026-10-17
---
Reviewed and parked: microsoft/msr-acc (coupled-cluster atomization energies), HuggingFaceBio (DNA language models plus a TRL SeqQA reward regression fixture), Anthropic/BioMysteryBench-full (90 agentic bioinformatics problems, gated CC-BY-4.0). None supplies drug-structure data.

**Why:** different targets and modalities; the hand-checked reward fixture and BioMysteryBench's anonymisation are design precedents only.
**How to apply:** cite as related work in docs/paper/related-work.md; revisit only if the study adds tool use or a physics-based property reward.
