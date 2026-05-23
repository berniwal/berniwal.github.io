Success rate (exact recovery) at increasing verifier-call budgets:

| target | method | 100k | 200k | 500k | 1M | 2M |
|---|---|---|---|---|---|---|
| easy | Random search | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| easy | Evolution (GP) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| easy | Greedy RL | 0.75 | 0.75 | 0.80 | 0.80 | 0.90 |
| easy | Risk-seeking RL (DSR) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| easy | Entropic RL (J_beta) | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 |
| medium | Random search | 0.50 | 0.75 | 1.00 | 1.00 | 1.00 |
| medium | Evolution (GP) | 0.85 | 0.90 | 0.95 | 1.00 | 1.00 |
| medium | Greedy RL | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 |
| medium | Risk-seeking RL (DSR) | 0.55 | 0.70 | 0.90 | 1.00 | 1.00 |
| medium | Entropic RL (J_beta) | 0.45 | 0.45 | 0.45 | 0.45 | 0.45 |
| harder | Random search | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| harder | Evolution (GP) | 0.00 | 0.00 | 0.05 | 0.05 | 0.05 |
| harder | Greedy RL | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| harder | Risk-seeking RL (DSR) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| harder | Entropic RL (J_beta) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
