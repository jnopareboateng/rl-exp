# Rank readout on Test-T

The answer ranks the 41 Test-T drugs by similarity to itself; the table shows where the row's own drug lands. Lower percentile rank is better. One fixed answer that is valid on every row scores exactly 0.512 on percentile rank and at most 0.024 on top-1, whatever it is; failures and tied nearest references only lower it. A policy whose answers vary without using the drug scores those values in expectation (Val spread, sd 0.02 to 0.035 on the percentile rank), so read the number of distinct answers beside it. Rows with no valid answer among their first eight samples take the worst rank and lower the valid-answer share. Nothing in this table is confirmatory.

| Arm | Top-1 | Mean percentile rank | Valid answers | Distinct answers | Drugs |
|---|---:|---:|---:|---:|---:|
| C0 | 0.024 | 0.512 | 1.000 | 1 | 41 |
| C1 | 0.045 | 0.427 | 1.000 | 64 | 41 |
| C2 | 0.024 | 0.512 | 1.000 | 1 | 41 |
| C3 | 0.024 | 0.512 | 1.000 | 1 | 41 |
| A0n | 0.027 | 0.697 | 0.552 | 54 | 41 |
| A4 | 0.058 | 0.501 | 0.914 | 89 | 41 |
| A2 | 0.030 | 0.547 | 1.000 | 100 | 41 |
| RL-full | 0.048 | 0.475 | 1.000 | 47 | 41 |
| A2-shuffle | 0.037 | 0.503 | 1.000 | 100 | 41 |
| RL-full-shuffle | 0.057 | 0.484 | 1.000 | 48 | 41 |
| A2-blank | 0.024 | 0.512 | 1.000 | 1 | 41 |
| RL-full-blank | 0.024 | 0.512 | 1.000 | 1 | 41 |
| chance (one fixed answer, valid on every row) | 0.024 | 0.512 | | 1 | 41 |

### RL-full minus RL-full-shuffle (Test-T)

41 common drugs. Wins count drugs for which the first arm is better.

| Measure | A | B | Difference | Wins / losses / ties | p (sign-flip) | Components |
|---|---:|---:|---:|---:|---:|---:|
| Mean percentile rank | 0.475 | 0.484 | -0.010 | 18 / 19 / 4 | 0.5876 | 41 |
| Top-1 | 0.048 | 0.057 | -0.009 | 1 / 2 / 38 | 0.5000 | 41 |

### A2 minus A2-shuffle (Test-T)

41 common drugs. Wins count drugs for which the first arm is better.

| Measure | A | B | Difference | Wins / losses / ties | p (sign-flip) | Components |
|---|---:|---:|---:|---:|---:|---:|
| Mean percentile rank | 0.547 | 0.503 | +0.043 | 24 / 17 / 0 | 0.2655 | 41 |
| Top-1 | 0.030 | 0.037 | -0.007 | 2 / 3 / 36 | 0.8750 | 41 |

### A2 minus RL-full (Test-T)

41 common drugs. Wins count drugs for which the first arm is better.

| Measure | A | B | Difference | Wins / losses / ties | p (sign-flip) | Components |
|---|---:|---:|---:|---:|---:|---:|
| Mean percentile rank | 0.547 | 0.475 | +0.072 | 15 / 25 / 1 | 0.0242 | 41 |
| Top-1 | 0.030 | 0.048 | -0.017 | 1 / 2 / 38 | 0.7500 | 41 |

