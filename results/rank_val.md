# Rank readout on Val

The answer ranks the 20 Val drugs by similarity to itself; the table shows where the row's own drug lands. Lower percentile rank is better. One fixed answer that is valid on every row scores exactly 0.525 on percentile rank and at most 0.050 on top-1, whatever it is; failures and tied nearest references only lower it. A policy whose answers vary without using the drug scores those values in expectation (Val spread, sd 0.02 to 0.035 on the percentile rank), so read the number of distinct answers beside it. Rows with no valid answer among their first eight samples take the worst rank and lower the valid-answer share. Nothing in this table is confirmatory.

| Arm | Top-1 | Mean percentile rank | Valid answers | Distinct answers | Drugs |
|---|---:|---:|---:|---:|---:|
| val8-control_constant | 0.050 | 0.525 | 1.000 | 1 | 20 |
| val8-control_constant_pool | 0.050 | 0.525 | 1.000 | 1 | 20 |
| val8-control_copy_nearest | 0.129 | 0.439 | 1.000 | 49 | 20 |
| b4-t0.6 | 0.050 | 0.623 | 0.679 | 38 | 20 |
| a4-t0.6 | 0.150 | 0.464 | 0.975 | 56 | 20 |
| a2-val | 0.188 | 0.396 | 1.000 | 58 | 20 |
| rlfull5-s50-t0.3 | 0.100 | 0.473 | 1.000 | 26 | 20 |
| a2-shuffle-t1.0 | 0.000 | 0.566 | 1.000 | 58 | 20 |
| r5s50-shuffle-t0.3 | 0.071 | 0.528 | 1.000 | 23 | 20 |
| r5s50-blank-t0.3 | 0.050 | 0.525 | 1.000 | 1 | 20 |
| b9-t0.6 | 0.113 | 0.424 | 0.988 | 57 | 20 |
| b27-direct | 0.196 | 0.379 | 1.000 | 58 | 20 |
| a2-t0.3 | 0.092 | 0.498 | 1.000 | 53 | 20 |
| a2-t0.6 | 0.092 | 0.508 | 1.000 | 56 | 20 |
| chance (one fixed answer, valid on every row) | 0.050 | 0.525 | | 1 | 20 |

### rlfull5-s50-t0.3 minus r5s50-shuffle-t0.3 (Val)

20 common drugs. Wins count drugs for which the first arm is better.

| Measure | A | B | Difference | Wins / losses / ties | p (sign-flip) | Components |
|---|---:|---:|---:|---:|---:|---:|
| Mean percentile rank | 0.473 | 0.528 | -0.055 | 12 / 6 / 2 | 0.0270 | 16 |
| Top-1 | 0.100 | 0.071 | +0.029 | 2 / 0 / 18 | 0.5000 | 16 |

### a2-val minus a2-shuffle-t1.0 (Val)

20 common drugs. Wins count drugs for which the first arm is better.

| Measure | A | B | Difference | Wins / losses / ties | p (sign-flip) | Components |
|---|---:|---:|---:|---:|---:|---:|
| Mean percentile rank | 0.396 | 0.566 | -0.170 | 16 / 4 / 0 | 0.0045 | 16 |
| Top-1 | 0.188 | 0.000 | +0.188 | 8 / 0 / 12 | 0.0078 | 16 |

### a2-val minus rlfull5-s50-t0.3 (Val)

20 common drugs. Wins count drugs for which the first arm is better.

| Measure | A | B | Difference | Wins / losses / ties | p (sign-flip) | Components |
|---|---:|---:|---:|---:|---:|---:|
| Mean percentile rank | 0.396 | 0.473 | -0.077 | 11 / 9 / 0 | 0.1692 | 16 |
| Top-1 | 0.188 | 0.100 | +0.087 | 6 / 3 / 11 | 0.3398 | 16 |

### a2-t0.3 minus rlfull5-s50-t0.3 (Val)

20 common drugs. Wins count drugs for which the first arm is better.

| Measure | A | B | Difference | Wins / losses / ties | p (sign-flip) | Components |
|---|---:|---:|---:|---:|---:|---:|
| Mean percentile rank | 0.498 | 0.473 | +0.025 | 8 / 12 / 0 | 0.6792 | 16 |
| Top-1 | 0.092 | 0.100 | -0.008 | 2 / 3 / 15 | 0.8750 | 16 |

### a2-t0.6 minus rlfull5-s50-t0.3 (Val)

20 common drugs. Wins count drugs for which the first arm is better.

| Measure | A | B | Difference | Wins / losses / ties | p (sign-flip) | Components |
|---|---:|---:|---:|---:|---:|---:|
| Mean percentile rank | 0.508 | 0.473 | +0.035 | 9 / 11 / 0 | 0.4756 | 16 |
| Top-1 | 0.092 | 0.100 | -0.008 | 2 / 3 / 15 | 0.8750 | 16 |

