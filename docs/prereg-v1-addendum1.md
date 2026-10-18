# Pre-registration v1, addendum 1 (secondary analyses)

Status: committed and tagged `prereg-v1-a1` on 2026-10-18, before any sample of a language-model arm of the locked run was scored or read. The tag `prereg-v1` and everything it fixed are untouched. This addendum ADDS analyses and names the hypotheses they test; it changes no arm, no decoding setting, no checkpoint, no endpoint, no comparator, no selection and no split, and it removes nothing.
The locked sampling run `runs/final` was started under the frozen spec before this file existed, so nothing here could have influenced which samples exist. The only Test answers that existed when this file was committed are the CPU controls C0 to C3, which `docs/prereg-v1.md` section 3 specifies and which are deterministic functions of Train; no sample of a model arm had been scored or read.
Read with `docs/prereg-v1.md` (the protocol), `docs/contracts.md` (the exact definitions) and `docs/plan.md` section 0 (the review log).

## 1. Why this exists

The confirmatory endpoint fv_tanimoto is maximised by an answer that ignores the drug: the reward-optimal constant C3 delivers 0.1865 on Val, the best constant that any candidate of the Train-plus-pool set could have been delivers 0.1954, and the Val-selected RL checkpoint delivers 0.1949. An endpoint on which a constant is near the ceiling cannot distinguish an arm that reads the profile from an arm that has learned a good generic answer.
The protocol already anticipated this with the lift diagnostic and the shuffled-profile runs (section 7), but it left both unregistered as tests and gave lift, which is biased toward zero for a near-constant policy, the central role. This addendum names four secondary hypotheses, each with its test and its claim, and adds one readout on which a drug-independent policy scores exactly chance by construction.

## 2. What is added

**(a) The rank readout** (`scripts/rank_readout.py`, defined in `docs/contracts.md`). The candidate set is the N distinct drugs of the split, each by its reference structure, fixed by the split and not by the arm. For every prompt row we take the first valid, closed, finished answer among the first 8 samples (the endpoint's own policy), rank the N references by count-vector Morgan Tanimoto to that answer, and record the mid-rank of the row's own drug. A row with no valid answer takes the worst rank, and the share of rows that had one is reported beside the result. Per drug we average over its rows, then over drugs.
Two measures are reported: **top-1** (the own reference is strictly nearest) and **mean percentile rank** (rank / N; lower is better).

Its property, which is why it is here: any policy whose output does not depend on the drug gives every row the same ranking, so its top-1 is exactly 1/N and its mean percentile rank is exactly (N+1)/2N, whatever molecule it emits. Both chance values are printed beside every result. On Val (N = 20, chance 0.050 and 0.525) the three drug-independent arms land on them exactly: C0 0.050 / 0.525, C3 0.050 / 0.525, and the blank-profile RL run 0.050 / 0.525. The readout is therefore not gameable by a constant, which the endpoint is.

**(b) The random-pair calibration** (`scripts/random_pair_calibration.py`). The distribution of count-vector Tanimoto between the 332 distinct reference structures of the four used splits (54,946 pairs; reference structures only, no model output, no prompt, no sample): mean 0.141, median 0.141, p75 0.180, p90 0.213, p99 0.274. A delivered similarity of 0.195 sits at the 83rd percentile, so about 17% of random pairs of approved drugs are at least that similar to each other. This is a property of the drug population, not a result of any arm or split.

**(c) The four secondary hypotheses of section 4**, with their tests, and the split of the protocol's section 7 criterion into two separately named claims (section 3 here).

## 3. The word "profile": two claims, not one

`docs/prereg-v1.md` section 7 required, for a claim that an arm uses the profile, all of: fv_lift above zero with a bootstrap interval excluding zero; fv_lift above the nearest-profile control C1 on the same split; and a matched shuffled-profile run with lower fv_tanimoto. The second condition does not test whether an arm uses the profile, it tests whether the arm uses it better than TF-IDF nearest-profile copying. Conflating the two makes the criterion unachievable for a weak but real effect and unreportable either way.
It is therefore split, with no change to what must be measured:

- **P1, "the arm's output depends on the profile".** Requires the matched shuffled-profile run to score lower on the same prompts, by the paired cluster-level sign-flip test at alpha 0.05 on fv_tanimoto, AND the same direction on the rank readout's mean percentile rank (shuffled worse: a higher percentile rank; its own p-value is reported beside it). The lift is reported beside it as a diagnostic, not as a condition. P1 can be claimed only for arms with a shuffled-profile run on Test: A2 and RL-full.
- **P2, "the arm uses the profile better than nearest-profile copying".** Requires P1 and fv_lift above C1's fv_lift on the same split with a component-bootstrap interval for the difference excluding zero.

A claim of P1 does not imply P2, and neither implies structure recovery. Both are secondary (section 5). Where the paper previously said "no arm qualifies", it now says which of P1 and P2 each arm meets.

## 4. Secondary hypotheses

Each is tested on Test-T and reported on Test-R as exploratory, from the frozen samples of `runs/final`, with the exact two-sided cluster-level sign-flip test of `docs/prereg-v1.md` section 4 (components are the exchangeable units; in Test-T each of the 41 drugs is its own unit), alpha 0.05, wins, losses and ties shown.

- **S1 (profile dependence of the RL arm).** The Val-selected RL arm (A5 = RL-full) on the real profile against its own shuffled-profile run, on fv_tanimoto and on the rank readout's mean percentile rank. Claim if the shuffled run is worse on both (lower fv_tanimoto, higher mean percentile rank): "the arm's output depends on the profile" (P1). Val, descriptive and already selected on: fv_tanimoto 0.1949 against 0.1831, difference +0.012 [+0.003, +0.020], 13 wins 7 losses, p = 0.034; mean percentile rank 0.473 against 0.528, difference -0.055, 12 wins 6 losses 2 ties, p = 0.027.
- **S2 (profile dependence of the SFT arm).** A2 on the real profile against its own shuffled-profile run, same two measures and same test. Val: fv_tanimoto 0.1557 against 0.1315, 13 wins 7 losses, p = 0.022; mean percentile rank 0.396 against 0.566, difference -0.170, 16 wins 4 losses, p = 0.0045.
- **S3 (does the RL step change profile dependence?).** A2 against the Val-selected RL arm on the rank readout's mean percentile rank, both on the real profile. This is a two-sided question and the Val data do not answer it: A2 0.396 against 0.473, difference -0.077, 11 wins 9 losses, p = 0.169, and A2's advantage is partly decoding (A2 scores 0.498 at T=0.3 and 0.508 at T=0.6 against 0.396 at its selected T=1.0). It is registered because the direction matters for the paper's reading of what RL did, and because 41 drugs is the first cohort with a chance of resolving it. No claim is permitted from a non-significant result in either direction.
- **S4 (calibration).** The delivered similarity of every arm is reported with its percentile in the random-pair distribution of section 2(b). This is descriptive, has no test and no claim attached, and is computed from reference structures only.

## 5. Nothing here is confirmatory

The one confirmatory contrast remains the Val-selected A5 against C3 on Test-T by fv_tanimoto (`docs/prereg-v1.md` section 4). S1 to S4, the rank readout, the calibration, every Test-R number and every other comparison are secondary or exploratory. No p-value in this addendum is adjusted for multiplicity, and the four hypotheses plus two measures each are enough tests that a single result at p just under 0.05 should be read as suggestive, not established; the paper says so where it reports them.
A secondary result cannot rescue a failed confirmatory contrast, and failing to reject is not evidence of no effect. `scripts/rank_readout.py` writes `"confirmatory": false` into every file it produces and prints it in every table it writes.

## 6. Size references: Val-only mechanism controls

The Test run of `prereg-v1` is 4B only (deviation D4, the project owner's scope decision) and this addendum does not widen it: no 9B or 27B arm is sampled on a Test split in this study. The untrained Qwen3.5-27B is nonetheless the only size reference whose Val fv_lift clears both parts of the old section 7 criterion (0.045, interval [0.011, 0.075], against C1's 0.022), and it has no shuffled-profile run, so whether a model of that size uses the profile is untested.
For the 4B arms at their Val-selected decoding none clears both parts; one grid cell that was not selected, the untrained 4B at T=0.3, does (0.027, interval [0.005, 0.049], fv_tanimoto 0.097).

The shuffled-profile Val runs of the untrained Qwen3.5-27B and of the untrained 9B are therefore made as Val-only mechanism controls (`docs/prereg-v1.md` section 3 permits them after the tag), unconditionally, at the decoding setting of the matching real-profile Val run, and are read with the same paired test and the same rank readout as S1 and S2. They are exploratory, enter no confirmatory or secondary hypothesis, and their result is reported whatever it is.
A Test run of either model is named in the paper as a follow-up and is not part of this study.

## 7. What this addendum does not do

It does not change the arms, the decoding settings, the checkpoint, the comparator set, the selection, the endpoint, the splits, the cohort, the data manifest or the frozen spec; it adds no arm to the confirmatory contrast; it does not rescore any arm with a different setting; and it is written and tagged before any sample of a model arm is scored or read.
The `prereg-v1` tag, its three hashes and `runs/final` are unchanged. Any later change is a new addendum under a new tag, reported as a deviation.
