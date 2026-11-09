# Architecture & design decisions

This document explains *why* the system is built the way it is. The README covers *what* it does.

## Layers

| Layer | Module | Responsibility |
|---|---|---|
| Config | `config.py` | One typed `Settings` object, overridable by `RE_*` env vars / `.env` |
| Data | `data/` | Simulator, schema contracts, IO (Parquet/CSV) |
| Analytics | `analytics/` | Cohorts, RFM, KPIs, which are pure functions of the order table |
| Churn | `churn/` | Point-in-time features, training/selection, evaluation, reason codes |
| CRM | `crm/` | Playbook → briefs → providers → guardrails → campaigns |
| Orchestration | `pipeline.py` | The one place that wires layers together; reused by CLI, API and UI |
| Interfaces | `cli.py`, `api/`, `app/` | Thin. No business logic lives in them. |

Every interface calls `pipeline.ensure_artifacts()`, which loads data, retrains the model if it's missing or the data fingerprint changed, and rescores if scores are older than the model. So the CLI, API and dashboard can't disagree with each other.

## Decisions

### 1. Simulate behaviour, not outcomes
The v0 generator sampled "retained this month?" from a fixed curve, so retention was independent of anything observable and any churn model built on it would have been theatre. The new simulator models latent engagement and makes churn hazard a function of it. Late deliveries, promo-driven acquisition and narrow baskets therefore *cause* churn in the data, and the model has something real to find. Parameters were tuned until the cohort curve matched typical quick-commerce shapes (M1 ≈ 40%, M6 ≈ 19%).

### 2. Censoring-aware cohorts
A cohort acquired in May can't have a "month 3" yet. Showing that cell as 0% (or computing it from a half-finished month) creates fake drop-offs on the heatmap's right edge. Unobservable cells are `NaN`, the trailing partial month is excluded by default, and the headline curve is a size-weighted average over observed cells only.

### 3. Churn definition and population
"Churn" only means something relative to a horizon and a population. We predict **no order in the next 30 days among users active in the last 30 days**. A wider population (e.g. 60 days) inflates AUC because it includes people who have already left. That's easy to predict and useless to act on. The narrower definition gives a harder, more honest problem with a ~38% base rate.

### 4. Out-of-time validation
Users appear in many monthly snapshots, and behaviour drifts over time. Random K-fold would leak future behaviour into training and the same user across folds. We hold out the most recent fully labelled snapshots, train only on earlier ones, and report those metrics. Then we refit on everything for production scoring.

### 5. Always compare against the rule you're replacing
The card reports the model next to (a) ranking by recency and (b) the original `inactive > 14 days` rule *at the same outreach budget*. A model that can't beat a one-line rule shouldn't ship, and the same-budget comparison is the one a growth lead cares about.

### 6. Prioritise by value at risk, not probability
A 90% churn risk on a ₹200/quarter user matters less than 50% on a ₹30k/quarter regular. The queue sorts by `p(churn) × trailing-90d spend`. That only works if probabilities are calibrated, so ECE and a calibration plot are first-class metrics.

### 7. Reason codes over SHAP
Per-user explanations need to be readable by a CRM manager and usable as LLM context. Deterministic, feature-threshold reason codes ("3 late deliveries in 30 days") do both, cost nothing at scoring time, and avoid a heavy dependency. Global importance comes from permutation importance on held-out data.

### 8. LLMs write copy; policy owns money
Letting a model decide who gets a discount turns a margin decision into a sampling decision. The playbook assigns incentives deterministically, and the guardrails reject any draft that mentions a different offer. The model's job is reduced to what it's good at: tone, specificity and channel fit.

### 9. Guardrails, retry, fallback
LLM output is untrusted input. Every draft is validated (length, spam phrases, caps, emoji, merge tokens, offer consistency). Violations go back to the model as feedback once, and then a vetted template is used. Provider errors (timeouts, auth, malformed JSON) take the same fallback path, so the batch never fails halfway.

### 10. Audience-level generation
Per-user LLM calls scale cost with users and make review impossible. Per-audience generation (stage × reason) yields ~20 messages a human can approve, personalised at send time with merge tokens.

## Extending

* **New provider:** subclass `crm.providers.BaseProvider`, implement `generate()`, and register it in `get_provider()`.
* **New feature:** add it in `churn/features.py`, append it to `NUMERIC_FEATURES`, and add a reason rule in `churn/explain.py` if it's actionable. The leakage test covers it automatically.
* **Real data:** export `users` / `orders` matching `data/schema.py` into `RE_DATA_DIR` and run `retention-engine train`.
