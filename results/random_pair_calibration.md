# What a similarity score means: random pairs of approved drugs

Count-vector Morgan Tanimoto (the endpoint's own similarity) between every pair of the 332 distinct reference structures of train, val, test_r, test_t (54,946 pairs): mean 0.141, standard deviation 0.058.

| Percentile | Similarity |
|---|---:|
| 5 | 0.046 |
| 25 | 0.102 |
| 50 | 0.141 |
| 75 | 0.180 |
| 90 | 0.213 |
| 95 | 0.233 |
| 97.5 | 0.251 |
| 99 | 0.274 |
| 99.5 | 0.293 |
| 99.9 | 0.389 |

| Score | Percentile of random pairs | Share of random pairs at least this similar |
|---|---:|---:|
| 0.1760 | 72.7 | 27.3% |
| 0.1740 | 71.6 | 28.4% |
| 0.0760 | 13.1 | 86.9% |
| 0.1865 | 78.7 | 21.3% |
| 0.1954 | 83.1 | 16.9% |
| 0.1949 | 82.8 | 17.2% |

A delivered similarity should be read against this scale: a score whose percentile is low is a score many unrelated pairs of approved drugs reach.

