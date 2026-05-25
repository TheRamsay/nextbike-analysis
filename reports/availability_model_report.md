# Nextbike Brno Availability Modeling Report

Generated at: `2026-05-25T02:09:59+02:00`

## Executive Summary

We now have an end-to-end local pipeline: poll GBFS, normalize to DuckDB, build a station-level ML dataset, evaluate simple baselines, and train first sklearn tabular models.

The current data is still mostly evening/night behavior, so the strongest baseline is simple persistence: assume a station's 30-minute future emptiness equals its current emptiness. That baseline is very hard to beat because stations barely change during this window.

Best baseline by F1: `persistence_empty_now` with F1 `0.9867`.

Best trained model by F1: `hist_gradient_boosting` with F1 `0.9871`.

Interpretation: the best model is slightly ahead of persistence on this split, but the margin is small. We need morning commute, daytime, and weekend data before judging model class quality.

## Data Snapshot

| Metric | Value |
|---|---|
| `collection_runs` | `353` |
| `first_collected_at` | `2026-05-24 18:32:12+02:00` |
| `latest_collected_at` | `2026-05-25 02:09:06+02:00` |
| `station_rows` | `104841` |
| `free_bike_rows` | `184180` |
| `distinct_stations` | `297` |
| `bikes_available_min_avg_max` | `553/566.28/572` |
| `db_size_mb` | `13.51` |

## Dataset

Target: `empty_future`, meaning whether a station is empty at the first snapshot between 30 and 40 minutes after `collected_at`.

Build settings:

- horizon minutes: `30`
- max target delay minutes: `40`
- lag tolerance minutes: `10`

| Metric | Value |
|---|---|
| `rows` | `94446` |
| `stations` | `297` |
| `first_collected_at` | `2026-05-24 20:17:28+02:00` |
| `latest_collected_at` | `2026-05-25 01:38:47+02:00` |
| `empty_future_rate` | `0.3955` |
| `avg_abs_change` | `0.0832` |
| `changed_rate` | `0.0688` |
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

- train rows: `70686`
- test rows: `23760`
- train window: `2026-05-24 20:17:28+02:00 -> 2026-05-25 00:15:20+02:00`
- test window: `2026-05-25 00:16:23+02:00 -> 2026-05-25 01:38:47+02:00`

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | Avg precision | Pred empty rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `persistence_empty_now` | 0.9896 | 0.9821 | 0.9913 | 0.9867 | 0.9900 | 0.9769 | 0.3901 |
| `station_prior_empty_rate` | 0.9559 | 0.9236 | 0.9656 | 0.9442 | 0.9841 | 0.9626 | 0.4040 |
| `low_inventory_le_1` | 0.8033 | 0.6627 | 1.0000 | 0.7972 | 0.9929 | 0.9793 | 0.5832 |
| `majority_train_class` | 0.6135 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.3865 | 0.0000 |

## First Models

Models trained:

- `logistic_regression`: one-hot station/region, scaled numeric features, balanced classes.
- `hist_gradient_boosting`: one-hot station/region, numeric features, nonlinear tree boosting.

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | Avg precision | Pred empty rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `hist_gradient_boosting` | 0.9900 | 0.9834 | 0.9910 | 0.9871 | 0.9944 | 0.9852 | 0.3895 |
| `logistic_regression` | 0.9851 | 0.9715 | 0.9904 | 0.9809 | 0.9915 | 0.9745 | 0.3940 |

![F1 comparison](figures/model_f1_comparison.png)

## Findings

- The problem is currently dominated by inertia. `empty_now` almost fully predicts `empty_future` in the overnight sample.
- `station_prior_empty_rate` is strong, which means station identity matters.
- `low_inventory_le_1` has perfect or near-perfect recall but many false positives; useful if the product goal is "never walk to a station likely to be empty".
- The trained models are useful as scoring/ranking models because ROC AUC and average precision are high; current F1 is only slightly better than persistence and should not be overinterpreted.

## Next Work

1. Keep collecting for several full days.
2. Re-run this report after a morning commute and after one full weekday.
3. Add threshold tuning for product goals: nearest reliable bike cares more about recall than raw accuracy.
4. Add weather and event/calendar features later.
5. Add route-level evaluation: "would this command recommend a station that still has a bike when I arrive?"
