from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import matplotlib.pyplot as plt
import nbformat as nbf

from nextbike_analysis.datasets import build_station_availability_dataset
from nextbike_analysis.modeling import evaluate_baselines, train_models


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "nextbike.duckdb"
TABLE_NAME = "training_station_availability"
REPORT_PATH = ROOT / "reports" / "availability_model_report.md"
NOTEBOOK_PATH = ROOT / "notebooks" / "01_availability_baseline_model.ipynb"
FIGURE_PATH = ROOT / "reports" / "figures" / "model_f1_comparison.png"


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(ZoneInfo("Europe/Prague")).replace(microsecond=0)
    build_result = build_station_availability_dataset(
        db_path=DB_PATH,
        table_name=TABLE_NAME,
        horizon_minutes=30,
        max_target_delay_minutes=40,
        lag_tolerance_minutes=10,
    )
    baseline = evaluate_baselines(
        db_path=DB_PATH,
        table_name=TABLE_NAME,
        test_fraction=0.25,
        low_bike_threshold=1,
    )
    model_result, _ = train_models(
        db_path=DB_PATH,
        table_name=TABLE_NAME,
        test_fraction=0.25,
        model_dir=None,
    )
    db_summary = load_db_summary()
    dataset_summary = load_dataset_summary()
    write_metric_figure(baseline, model_result)
    REPORT_PATH.write_text(
        build_markdown_report(
            generated_at,
            db_summary,
            dataset_summary,
            build_result,
            baseline,
            model_result,
        ),
        encoding="utf-8",
    )
    write_notebook(generated_at, db_summary, dataset_summary, build_result, baseline, model_result)
    print(f"wrote {REPORT_PATH}")
    print(f"wrote {NOTEBOOK_PATH}")
    print(f"wrote {FIGURE_PATH}")


def load_db_summary() -> dict[str, object]:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        row = con.sql(
            """
            select
                count(*) as collection_runs,
                min(collected_at) as first_collected_at,
                max(collected_at) as latest_collected_at,
                sum(station_count) as station_rows,
                sum(free_bike_count) as free_bike_rows,
                min(bikes_available) as min_bikes_available,
                round(avg(bikes_available), 2) as avg_bikes_available,
                max(bikes_available) as max_bikes_available
            from collection_runs
            """
        ).fetchone()
        distinct_stations = con.sql(
            "select count(distinct station_id) from station_status_snapshots"
        ).fetchone()[0]
    return {
        "collection_runs": row[0],
        "first_collected_at": row[1],
        "latest_collected_at": row[2],
        "station_rows": row[3],
        "free_bike_rows": row[4],
        "distinct_stations": distinct_stations,
        "bikes_available_min_avg_max": f"{row[5]}/{row[6]}/{row[7]}",
        "db_size_mb": round(DB_PATH.stat().st_size / 1024 / 1024, 2),
    }


def load_dataset_summary() -> dict[str, object]:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        row = con.sql(
            f"""
            select
                count(*) as rows,
                count(distinct station_id) as stations,
                min(collected_at) as first_collected_at,
                max(collected_at) as latest_collected_at,
                round(avg(case when empty_future then 1.0 else 0.0 end), 4)
                    as empty_future_rate,
                round(avg(abs(bikes_future - bikes_now)), 4) as avg_abs_change,
                round(avg(case when bikes_future != bikes_now then 1.0 else 0.0 end), 4)
                    as changed_rate,
                count_if(bikes_lag_5m is null) as null_lag_5m,
                count_if(bikes_lag_15m is null) as null_lag_15m
            from {TABLE_NAME}
            """
        ).fetchone()
    return {
        "rows": row[0],
        "stations": row[1],
        "first_collected_at": row[2],
        "latest_collected_at": row[3],
        "empty_future_rate": row[4],
        "avg_abs_change": row[5],
        "changed_rate": row[6],
        "null_lag_5m": row[7],
        "null_lag_15m": row[8],
    }


def write_metric_figure(baseline, model_result) -> None:
    rows = [*baseline.metrics, *model_result.metrics]
    names = [row.name for row in rows]
    f1_scores = [row.f1 for row in rows]
    colors = ["#4c78a8" if row in baseline.metrics else "#f58518" for row in rows]

    plt.figure(figsize=(10, 5))
    plt.barh(names, f1_scores, color=colors)
    plt.xlabel("F1 score for empty_future")
    plt.xlim(0, 1)
    plt.title("Nextbike Brno 30-minute emptiness prediction")
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=160)
    plt.close()


def build_markdown_report(
    generated_at: datetime,
    db_summary: dict[str, object],
    dataset_summary: dict[str, object],
    build_result,
    baseline,
    model_result,
) -> str:
    best_baseline = max(baseline.metrics, key=lambda row: row.f1)
    best_model = max(model_result.metrics, key=lambda row: row.f1)
    model_delta = best_model.f1 - best_baseline.f1
    if model_delta > 0.002:
        interpretation = (
            "The best trained model is slightly ahead of persistence on this split, "
            "but the margin is still small."
        )
    elif model_delta < -0.002:
        interpretation = (
            "The best trained model does not beat persistence on this split. "
            "That makes persistence the operational baseline to beat."
        )
    else:
        interpretation = (
            "The best trained model and persistence are effectively tied on this split."
        )
    return f"""# Nextbike Brno Availability Modeling Report

Generated at: `{generated_at.isoformat()}`

## Executive Summary

We now have an end-to-end local pipeline: poll GBFS, normalize to DuckDB, build a station-level ML dataset, evaluate simple baselines, and train first sklearn tabular models.

The current data now includes evening, overnight, and early morning behavior. The strongest baseline is still simple persistence: assume a station's 30-minute future emptiness equals its current emptiness. That baseline is hard to beat because many stations remain stable over a 30-minute horizon.

Best baseline by F1: `{best_baseline.name}` with F1 `{best_baseline.f1:.4f}`.

Best trained model by F1: `{best_model.name}` with F1 `{best_model.f1:.4f}`.

Interpretation: {interpretation} We still need daytime, afternoon, and weekend data before judging model class quality.

## Data Snapshot

{dict_table(db_summary)}

## Dataset

Target: `empty_future`, meaning whether a station is empty at the first snapshot between 30 and 40 minutes after `collected_at`.

Build settings:

- horizon minutes: `{build_result.horizon_minutes}`
- max target delay minutes: `{build_result.max_target_delay_minutes}`
- lag tolerance minutes: `{build_result.lag_tolerance_minutes}`

{dict_table(dataset_summary)}

Feature engineering started in `build-dataset`:

- current station state: `bikes_now`, `docks_now`, `empty_now`
- lag features: `bikes_lag_5m`, `bikes_lag_15m`
- deltas: `bikes_delta_5m`, `bikes_delta_15m`
- cyclic time features: `hour_sin`, `hour_cos`, `weekday_sin`, `weekday_cos`
- short rolling station history: `bikes_avg_30m`, `empty_rate_30m`, `samples_30m`
- station metadata: `station_id`, `region_id`, `lat`, `lon`, `capacity`

## Baseline Evaluation

Temporal split:

- train rows: `{baseline.train_rows}`
- test rows: `{baseline.test_rows}`
- train window: `{baseline.train_window}`
- test window: `{baseline.test_window}`

{metrics_table(baseline.metrics)}

## First Models

Models trained:

- `logistic_regression`: one-hot station/region, scaled numeric features, balanced classes.
- `hist_gradient_boosting`: one-hot station/region, numeric features, nonlinear tree boosting.

{metrics_table(model_result.metrics)}

![F1 comparison](figures/model_f1_comparison.png)

## Findings

- The problem is currently dominated by inertia. `empty_now` is still a very strong predictor of `empty_future`.
- `station_prior_empty_rate` is strong, which means station identity matters.
- `low_inventory_le_1` has perfect or near-perfect recall but many false positives; useful if the product goal is "never walk to a station likely to be empty".
- The trained models are useful as scoring/ranking models because ROC AUC and average precision are high; current F1 versus persistence should not be overinterpreted until we have broader daytime coverage.

## Next Work

1. Keep collecting for several full days.
2. Re-run this report after a morning commute and after one full weekday.
3. Add threshold tuning for product goals: nearest reliable bike cares more about recall than raw accuracy.
4. Add weather and event/calendar features later.
5. Add route-level evaluation: "would this command recommend a station that still has a bike when I arrive?"
"""


def dict_table(values: dict[str, object]) -> str:
    lines = ["| Metric | Value |", "|---|---|"]
    for key, value in values.items():
        lines.append(f"| `{key}` | `{value}` |")
    return "\n".join(lines)


def metrics_table(rows) -> str:
    lines = [
        "| Model | Accuracy | Precision | Recall | F1 | ROC AUC | Avg precision | Pred empty rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: item.f1, reverse=True):
        roc_auc = f"{row.roc_auc:.4f}" if row.roc_auc is not None else "n/a"
        avg_precision = (
            f"{row.average_precision:.4f}" if row.average_precision is not None else "n/a"
        )
        lines.append(
            "| "
            f"`{row.name}` | {row.accuracy:.4f} | {row.precision:.4f} | "
            f"{row.recall:.4f} | {row.f1:.4f} | {roc_auc} | {avg_precision} | "
            f"{row.positive_rate:.4f} |"
        )
    return "\n".join(lines)


def write_notebook(
    generated_at: datetime,
    db_summary: dict[str, object],
    dataset_summary: dict[str, object],
    build_result,
    baseline,
    model_result,
) -> None:
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        nbf.v4.new_markdown_cell(
            f"""# Nextbike Brno Availability Baseline

Generated at `{generated_at.isoformat()}`.

This notebook reproduces the first 30-minute station emptiness baseline/model analysis.
"""
        ),
        nbf.v4.new_markdown_cell("## Current Summary\n\n" + dict_table(db_summary)),
        nbf.v4.new_markdown_cell("## Dataset Summary\n\n" + dict_table(dataset_summary)),
        nbf.v4.new_markdown_cell(
            f"""Build settings:

- horizon minutes: `{build_result.horizon_minutes}`
- max target delay minutes: `{build_result.max_target_delay_minutes}`
- lag tolerance minutes: `{build_result.lag_tolerance_minutes}`
"""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import duckdb
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / 'data' / 'nextbike.duckdb').exists():
    ROOT = ROOT.parent
DB_PATH = ROOT / 'data' / 'nextbike.duckdb'
TABLE_NAME = 'training_station_availability'

con = duckdb.connect(str(DB_PATH), read_only=True)
df = con.sql(f"select * from {TABLE_NAME} order by collected_at, station_id").df()
df.head()"""
        ),
        nbf.v4.new_code_cell(
            """df[['bikes_now', 'bikes_future', 'empty_now', 'empty_future']].describe(include='all')"""
        ),
        nbf.v4.new_markdown_cell("## Baseline Metrics\n\n" + metrics_table(baseline.metrics)),
        nbf.v4.new_code_cell(
            """from nextbike_analysis.modeling import evaluate_baselines

baseline = evaluate_baselines(
    db_path=DB_PATH,
    table_name=TABLE_NAME,
    test_fraction=0.25,
    low_bike_threshold=1,
)
[(row.name, row.f1, row.roc_auc) for row in baseline.metrics]"""
        ),
        nbf.v4.new_markdown_cell("## Model Metrics\n\n" + metrics_table(model_result.metrics)),
        nbf.v4.new_code_cell(
            """from nextbike_analysis.modeling import train_models

model_result, _ = train_models(
    db_path=DB_PATH,
    table_name=TABLE_NAME,
    test_fraction=0.25,
    model_dir=None,
)
[(row.name, row.f1, row.roc_auc) for row in model_result.metrics]"""
        ),
        nbf.v4.new_markdown_cell(
            """## Notes

The current sample is mostly overnight, so persistence is a very strong baseline.
Do not judge advanced model classes until we have morning/daytime demand in the dataset.
"""
        ),
    ]
    nbf.write(notebook, NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
