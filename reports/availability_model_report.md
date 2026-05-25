# Nextbike Brno Availability Modeling Report

Generated at: `2026-05-25T10:00:13+02:00`

## Executive Summary

We now have an end-to-end local pipeline: poll GBFS, normalize to DuckDB, build a station-level ML dataset, evaluate simple baselines, and train first sklearn tabular models.

The current data now includes evening, overnight, and early morning behavior. The strongest baseline is still simple persistence: assume a station's 30-minute future emptiness equals its current emptiness. That baseline is hard to beat because many stations remain stable over a 30-minute horizon.

Best baseline by F1: `persistence_empty_now` with F1 `0.9498`.

Best trained model by F1: `hist_gradient_boosting` with F1 `0.9459`.

Interpretation: The best trained model does not beat persistence on this split. That makes persistence the operational baseline to beat. We still need daytime, afternoon, and weekend data before judging model class quality.

## Data Snapshot

| Metric | Value |
|---|---|
| `collection_runs` | `809` |
| `first_collected_at` | `2026-05-24 18:32:12+02:00` |
| `latest_collected_at` | `2026-05-25 09:59:57+02:00` |
| `station_rows` | `240273` |
| `free_bike_rows` | `443058` |
| `distinct_stations` | `297` |
| `bikes_available_min_avg_max` | `542/564.22/573` |
| `db_size_mb` | `23.01` |

## Dataset

Target: `empty_future`, meaning whether a station is empty at the first snapshot between 30 and 40 minutes after `collected_at`.

Build settings:

- horizon minutes: `30`
- max target delay minutes: `40`
- lag tolerance minutes: `10`

| Metric | Value |
|---|---|
| `rows` | `229581` |
| `stations` | `297` |
| `first_collected_at` | `2026-05-24 20:17:28+02:00` |
| `latest_collected_at` | `2026-05-25 09:29:04+02:00` |
| `empty_future_rate` | `0.3982` |
| `avg_abs_change` | `0.0888` |
| `changed_rate` | `0.0707` |
| `null_lag_5m` | `2970` |
| `null_lag_15m` | `6831` |

Feature engineering started in `build-dataset`:

- current station state: `bikes_now`, `docks_now`, `empty_now`
- lag features: `bikes_lag_5m`, `bikes_lag_15m`
- deltas: `bikes_delta_5m`, `bikes_delta_15m`
- cyclic time features: `hour_sin`, `hour_cos`, `weekday_sin`, `weekday_cos`
- short rolling station history: `bikes_avg_30m`, `empty_rate_30m`, `samples_30m`
- station metadata: `station_id`, `region_id`, `lat`, `lon`, `capacity`

## Baseline Evaluation

Temporal split:

- train rows: `171963`
- test rows: `57618`
- train window: `2026-05-24 20:17:28+02:00 -> 2026-05-25 06:09:06+02:00`
- test window: `2026-05-25 06:10:08+02:00 -> 2026-05-25 09:29:04+02:00`

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | Avg precision | Pred empty rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `persistence_empty_now` | 0.9581 | 0.9589 | 0.9408 | 0.9498 | 0.9557 | 0.9271 | 0.4135 |
| `station_prior_empty_rate` | 0.8675 | 0.8764 | 0.7981 | 0.8354 | 0.9107 | 0.8792 | 0.3838 |
| `low_inventory_le_1` | 0.8116 | 0.6937 | 0.9900 | 0.8158 | 0.9711 | 0.9405 | 0.6015 |
| `majority_train_class` | 0.5785 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.4215 | 0.0000 |

## First Models

Models trained:

- `logistic_regression`: one-hot station/region, scaled numeric features, balanced classes.
- `hist_gradient_boosting`: one-hot station/region, numeric features, nonlinear tree boosting.

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | Avg precision | Pred empty rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `hist_gradient_boosting` | 0.9548 | 0.9547 | 0.9372 | 0.9459 | 0.9814 | 0.9728 | 0.4137 |
| `logistic_regression` | 0.9400 | 0.9283 | 0.9295 | 0.9289 | 0.9783 | 0.9685 | 0.4220 |

![F1 comparison](figures/model_f1_comparison.png)

## Recommendation Evaluation

Offline simulation over `194` test timestamps and `5` fixed Brno origins.

Success means the selected station has at least one bike at the 30-minute target.

| Strategy | Success rate | Avg dist m | P90 dist m | No candidate | Avg extra m | Avg empty probability |
|---|---:|---:|---:|---:|---:|---:|
| `bikes>=2` | 0.9979 | 161 | 259 | 0 | 29 | 0.0029 |
| `dist+risk` | 0.9113 | 137 | 222 | 0 | 5 | 0.0113 |
| `model<0.4` | 0.8876 | 135 | 222 | 0 | 3 | 0.0298 |
| `nearest` | 0.8649 | 132 | 222 | 0 | 0 | 0.1220 |

## Findings

- The problem is currently dominated by inertia. `empty_now` is still a very strong predictor of `empty_future`.
- `station_prior_empty_rate` is strong, which means station identity matters.
- `low_inventory_le_1` has perfect or near-perfect recall but many false positives; useful if the product goal is "never walk to a station likely to be empty".
- The trained models are useful as scoring/ranking models because ROC AUC and average precision are high; current F1 versus persistence should not be overinterpreted until we have broader daytime coverage.
- For the current five-origin simulation, requiring at least two bikes is the strongest recommendation rule. It improves reliability at a small walking-distance cost.

## Next Work

1. Keep collecting for several full days.
2. Re-run this report after a morning commute and after one full weekday.
3. Add threshold tuning for product goals: nearest reliable bike cares more about recall than raw accuracy.
4. Add weather and event/calendar features later.
5. Expand route-level evaluation beyond fixed center origins and include realistic walk-to-station travel time.
