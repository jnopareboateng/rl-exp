# Rank readout on Val

The answer ranks the 20 Val drugs by similarity to itself; the table shows where the row's own drug lands. Lower percentile rank is better. One fixed answer that is valid on every row scores exactly 0.525 on percentile rank and at most 0.050 on top-1, whatever it is; failures and tied nearest references only lower it. A policy whose answers vary without using the drug scores those values in expectation (Val spread, sd 0.02 to 0.035 on the percentile rank), so read the number of distinct answers beside it. Rows with no valid answer among their first eight samples take the worst rank and lower the valid-answer share. Nothing in this table is confirmatory.

| Arm | Top-1 | Mean percentile rank | Valid answers | Distinct answers | Drugs |
|---|---:|---:|---:|---:|---:|
| b9-t0.6 | 0.113 | 0.424 | 0.988 | 57 | 20 |
| b9-shuffle-t0.6 | 0.013 | 0.598 | 0.963 | 56 | 20 |
| b27-direct | 0.196 | 0.379 | 1.000 | 58 | 20 |
| b27-shuffle-t1.0 | 0.017 | 0.561 | 1.000 | 58 | 20 |
| chance (one fixed answer, valid on every row) | 0.050 | 0.525 | | 1 | 20 |

### b9-t0.6 minus b9-shuffle-t0.6 (Val)

20 common drugs. Wins count drugs for which the first arm is better.

| Measure | A | B | Difference | Wins / losses / ties | p (sign-flip) | Components |
|---|---:|---:|---:|---:|---:|---:|
| Mean percentile rank | 0.424 | 0.598 | -0.175 | 15 / 5 / 0 | 0.0326 | 16 |
| Top-1 | 0.113 | 0.013 | +0.100 | 5 / 1 / 14 | 0.1250 | 16 |

### b27-direct minus b27-shuffle-t1.0 (Val)

20 common drugs. Wins count drugs for which the first arm is better.

| Measure | A | B | Difference | Wins / losses / ties | p (sign-flip) | Components |
|---|---:|---:|---:|---:|---:|---:|
| Mean percentile rank | 0.379 | 0.561 | -0.182 | 14 / 4 / 2 | 0.0475 | 16 |
| Top-1 | 0.196 | 0.017 | +0.179 | 6 / 0 / 14 | 0.1250 | 16 |

