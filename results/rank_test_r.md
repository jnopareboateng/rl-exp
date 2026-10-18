# Rank readout on Test-R

The answer ranks the 30 Test-R drugs by similarity to itself; the table shows where the row's own drug lands. Lower percentile rank is better. One fixed answer that is valid on every row scores exactly 0.517 on percentile rank and at most 0.033 on top-1, whatever it is; failures and tied nearest references only lower it. A policy whose answers vary without using the drug scores those values in expectation (Val spread, sd 0.02 to 0.035 on the percentile rank), so read the number of distinct answers beside it. Rows with no valid answer among their first eight samples take the worst rank and lower the valid-answer share. Nothing in this table is confirmatory.

| Arm | Top-1 | Mean percentile rank | Valid answers | Distinct answers | Drugs |
|---|---:|---:|---:|---:|---:|
| C0 | 0.033 | 0.517 | 1.000 | 1 | 30 |
| C1 | 0.158 | 0.396 | 1.000 | 58 | 30 |
| C2 | 0.033 | 0.517 | 1.000 | 1 | 30 |
| C3 | 0.033 | 0.517 | 1.000 | 1 | 30 |
| A0n | 0.086 | 0.582 | 0.706 | 51 | 30 |
| A4 | 0.067 | 0.438 | 0.936 | 73 | 30 |
| A2 | 0.044 | 0.430 | 0.983 | 78 | 30 |
| RL-full | 0.056 | 0.463 | 1.000 | 41 | 30 |
| A2-shuffle | 0.053 | 0.517 | 0.992 | 78 | 30 |
| RL-full-shuffle | 0.033 | 0.520 | 1.000 | 41 | 30 |
| A2-blank | 0.033 | 0.517 | 1.000 | 1 | 30 |
| RL-full-blank | 0.033 | 0.517 | 1.000 | 1 | 30 |
| chance (one fixed answer, valid on every row) | 0.033 | 0.517 | | 1 | 30 |

### RL-full minus RL-full-shuffle (Test-R)

30 common drugs. Wins count drugs for which the first arm is better.

| Measure | A | B | Difference | Wins / losses / ties | p (sign-flip) | Components |
|---|---:|---:|---:|---:|---:|---:|
| Mean percentile rank | 0.462 | 0.520 | -0.057 | 16 / 10 / 4 | 0.0840 | 27 |
| Top-1 | 0.056 | 0.033 | +0.022 | 2 / 0 / 28 | 0.5000 | 27 |

### A2 minus A2-shuffle (Test-R)

30 common drugs. Wins count drugs for which the first arm is better.

| Measure | A | B | Difference | Wins / losses / ties | p (sign-flip) | Components |
|---|---:|---:|---:|---:|---:|---:|
| Mean percentile rank | 0.430 | 0.517 | -0.087 | 19 / 10 / 1 | 0.0647 | 27 |
| Top-1 | 0.044 | 0.053 | -0.008 | 3 / 2 / 25 | 0.9375 | 27 |

### A2 minus RL-full (Test-R)

30 common drugs. Wins count drugs for which the first arm is better.

| Measure | A | B | Difference | Wins / losses / ties | p (sign-flip) | Components |
|---|---:|---:|---:|---:|---:|---:|
| Mean percentile rank | 0.430 | 0.462 | -0.033 | 17 / 11 / 2 | 0.5142 | 27 |
| Top-1 | 0.044 | 0.056 | -0.011 | 4 / 3 / 23 | 0.8750 | 27 |

