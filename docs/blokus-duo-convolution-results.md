# Spatial-representation pilot results

The bounded screen **failed** its declared criterion. Both actors started from random weights and received 1,600 games against frontier with identical PPO settings. This is a small single-seed screen, not a general verdict about convolutional architectures.

| Actor | Parameters | Training transitions | Training seconds | Selected update | Random score | Largest score | Frontier score |
|---|---|---|---|---|---|---|---|
| additive | 126,043 | 27,835 | 35.7 | 100 | 95.25% | 53.00% | 44.00% |
| conv | 52,978 | 27,706 | 152.3 | 100 | 94.25% | 57.75% | 18.00% |

CNN frontier score change relative to the additive control: -26.00% (-26.00 percentage points). Required: at least +5 points and at least 90% random score rate. Final tests contain 200 fresh games per opponent, balanced by seat.

The convolutional actor was more expensive at this budget and did not demonstrate a benefit. A more expressive model can need different amounts of experience or optimization; this experiment did not tune those factors and cannot explain the cause. It provides no reason to replace the stronger previously trained additive policies.

Implementation checks passed: candidate-to-native orientation/anchor mapping in both perspectives; invariant cell ordering; legal padding; on-policy log probabilities; finite PPO updates; architecture-aware checkpoint round trips. Full suite: 52 Python tests; the Rust engine remains unchanged with 20 previously passing tests.

[Declared protocol](blokus-duo-convolution-protocol.md) · [Previous failed frozen-opponent probe](blokus-duo-frozen-league-results.md)

The protocol specified stopping automatic continuation after this failed screen, following the failed frozen-opponent transfer experiment. Earlier successful policies, all code, and negative evidence are preserved for review.
