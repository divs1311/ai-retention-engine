# Model card: `churn_classifier`

The machine-readable version (with curves and full importances) is written to `artifacts/churn_model_card.json` on every `retention-engine train`, and is served at `GET /v1/model`. Figures below are from the default configuration (`seed=42`, 6,000 users).

## Intended use
Rank currently active users of a quick-commerce app by their risk of not ordering in the next 30 days, so a retention team can prioritise outreach by value at risk. **Not** intended for pricing, credit, or any decision with legal or financial consequences for an individual.

## Definition
| | |
|---|---|
| Label | `1` if the user places **no order in the 30 days** after the snapshot |
| Population | Users with ≥1 order in the 30 days before the snapshot and ≥7 days of tenure |
| Snapshots | Monthly, from day 90 of the data until the last date with a fully observed label window |
| Validation | Out-of-time: last 2 snapshots held out (≈2,550 user-snapshots) |

## Features (24)
Recency, tenure, orders and spend windows (30/90 days), AOV, frequency momentum, mean/std inter-order gap, overdue ratio, late-delivery rate and count, average delivery time, last-order-late flag, discount share, average category width, distinct categories, weekend share, promo-acquired flag, city, acquisition channel, device OS.

## Candidates & selection
| Model | ROC-AUC | PR-AUC | Brier | ECE | Lift @10% | Recall @20% |
|---|---:|---:|---:|---:|---:|---:|
| **HistGradientBoosting** ✔ | 0.851 | **0.788** | 0.147 | **0.017** | **2.41×** | 45.3% |
| Logistic regression | **0.853** | 0.780 | 0.147 | 0.021 | 2.37× | 45.2% |
| *Recency ranking (rule)* | 0.817 | 0.721 | n/a | n/a | 2.19× | 41.3% |

Selected on PR-AUC. At the same budget as the legacy `inactive > 14 days` rule (29% of users flagged): **precision 73.6% → 79.0%, recall 56.7% → 60.9%.**

## Most important features (permutation, Δ PR-AUC)
1. `recency_days`
2. `tenure_days`
3. `overdue_ratio`
4. `discount_share_90d`
5. `std_gap_days`
6. `late_rate_90d`

## Limitations
* Trained and evaluated on **simulated** data. The simulator's causal structure (engagement → hazard) is a modelling assumption, and real data will be noisier.
* Recency dominates, as it does in most real churn problems. The model's added value is in the tails (top-decile precision 91%) and in calibrated probabilities for value-at-risk ranking.
* Calibration is checked on the held-out window only. Monitor it after deployment.
* The model predicts churn, not *persuadability*. High-risk users aren't necessarily the ones a campaign can save. See the uplift-modelling item on the roadmap.

## Ethical considerations
Reason codes and incentives are rule-based and auditable. No protected attributes are used as features. City is used and could proxy for socioeconomic status, so review tier distributions by city before using scores for anything beyond outreach prioritisation.
