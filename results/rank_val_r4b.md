# Rank readout on Val

The answer ranks the 20 Val drugs by similarity to itself; the table shows where the row's own drug lands. Lower percentile rank is better. One fixed answer that is valid on every row scores exactly 0.525 on percentile rank and at most 0.050 on top-1, whatever it is; failures and tied nearest references only lower it. A policy whose answers vary without using the drug scores those values in expectation (Val spread, sd 0.02 to 0.035 on the percentile rank), so read the number of distinct answers beside it. Rows with no valid answer among their first eight samples take the worst rank and lower the valid-answer share. Nothing in this table is confirmatory.

| Arm | Top-1 | Mean percentile rank | Valid answers | Distinct answers | Drugs |
|---|---:|---:|---:|---:|---:|
| r4bs40-t0.3 | 0.087 | 0.490 | 1.000 | 4 | 20 |
| r4bs40-blank-t0.3 | 0.050 | 0.525 | 1.000 | 1 | 20 |
| r4bs40-shuffle-t0.3 | 0.037 | 0.553 | 1.000 | 4 | 20 |
| chance (one fixed answer, valid on every row) | 0.050 | 0.525 | | 1 | 20 |

### r4bs40-t0.3 minus r4bs40-shuffle-t0.3 (Val)

20 common drugs. Wins count drugs for which the first arm is better.

| Measure | A | B | Difference | Wins / losses / ties | p (sign-flip) | Components |
|---|---:|---:|---:|---:|---:|---:|
| Mean percentile rank | 0.490 | 0.553 | -0.063 | 7 / 1 / 12 | 0.0781 | 16 |
| Top-1 | 0.087 | 0.037 | +0.050 | 1 / 0 / 19 | 1.0000 | 16 |

