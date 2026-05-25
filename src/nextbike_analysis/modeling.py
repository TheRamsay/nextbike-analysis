from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from nextbike_analysis.datasets import validate_table_name


TARGET_COLUMN = "empty_future"
CATEGORICAL_FEATURES = ["station_id", "region_id"]
NUMERIC_FEATURES = [
    "hour",
    "weekday",
    "is_weekend",
    "bikes_now",
    "docks_now",
    "has_bike_now",
    "empty_now",
    "is_renting",
    "is_returning",
    "bikes_lag_5m",
    "minutes_since_lag_5m",
    "bikes_lag_15m",
    "minutes_since_lag_15m",
    "bikes_delta_5m",
    "bikes_delta_15m",
    "bikes_capacity_ratio",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "bikes_avg_30m",
    "empty_rate_30m",
    "samples_30m",
    "capacity",
    "lat",
    "lon",
]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


@dataclass(frozen=True)
class MetricRow:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    average_precision: float | None
    positive_rate: float


@dataclass(frozen=True)
class EvaluationResult:
    table_name: str
    target: str
    split_collected_at: object
    train_rows: int
    test_rows: int
    train_window: str
    test_window: str
    metrics: list[MetricRow]


@dataclass(frozen=True)
class StationRiskPrediction:
    station_id: str
    empty_probability: float
    risk_label: str


def load_dataset(db_path: Path, table_name: str) -> pd.DataFrame:
    table_name = validate_table_name(table_name)
    with duckdb.connect(str(db_path), read_only=True) as con:
        df = con.sql(f"select * from {table_name} order by collected_at, station_id").df()
    missing_columns = [column for column in [TARGET_COLUMN, *MODEL_FEATURES] if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(
            f"dataset table {table_name!r} is missing columns: {missing}. "
            "Rebuild it with `uv run nextbike build-dataset`."
        )
    return df


def load_model(model_path: Path) -> Any:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model does not exist: {model_path}. Run `uv run nextbike train-model` first."
        )
    return joblib.load(model_path)


def predict_latest_station_risk(
    *,
    db_path: Path,
    model_path: Path,
    station_ids: list[str],
) -> dict[str, StationRiskPrediction]:
    if not station_ids:
        return {}
    model = load_model(model_path)
    features = load_latest_station_features(db_path, station_ids)
    if features.empty:
        return {}
    probabilities = model.predict_proba(features[MODEL_FEATURES])[:, 1]
    predictions: dict[str, StationRiskPrediction] = {}
    for station_id, probability in zip(features["station_id"], probabilities, strict=True):
        probability = float(probability)
        predictions[str(station_id)] = StationRiskPrediction(
            station_id=str(station_id),
            empty_probability=probability,
            risk_label=risk_label(probability),
        )
    return predictions


def risk_label(empty_probability: float) -> str:
    if empty_probability >= 0.7:
        return "high"
    if empty_probability >= 0.4:
        return "medium"
    return "low"


def load_latest_station_features(db_path: Path, station_ids: list[str]) -> pd.DataFrame:
    placeholders = ", ".join("?" for _ in station_ids)
    with duckdb.connect(str(db_path), read_only=True) as con:
        return con.execute(
            f"""
            with latest_run as (
                select collected_at
                from collection_runs
                order by collected_at desc
                limit 1
            ),
            station_info as (
                select
                    station_id,
                    name,
                    short_name,
                    region_id,
                    lat,
                    lon,
                    capacity
                from (
                    select
                        station_id,
                        name,
                        short_name,
                        region_id,
                        lat,
                        lon,
                        capacity,
                        row_number() over (
                            partition by station_id
                            order by observed_at desc
                        ) as rn
                    from station_information
                )
                where rn = 1
            ),
            base as (
                select
                    s.station_id,
                    s.collected_at,
                    coalesce(s.num_bikes_available, 0) as bikes_now,
                    coalesce(s.num_docks_available, 0) as docks_now,
                    coalesce(s.num_bikes_available, 0) > 0 as has_bike_now,
                    coalesce(s.num_bikes_available, 0) = 0 as empty_now,
                    s.is_renting,
                    s.is_returning,
                    i.region_id,
                    i.lat,
                    i.lon,
                    i.capacity,
                    cast(strftime(s.collected_at, '%H') as integer) as hour,
                    cast(strftime(s.collected_at, '%w') as integer) as weekday,
                    cast(strftime(s.collected_at, '%w') as integer) in (0, 6) as is_weekend
                from station_status_snapshots s
                join latest_run lr using (collected_at)
                left join station_info i using (station_id)
                where s.station_id in ({placeholders})
            )
            select
                b.station_id,
                b.region_id,
                b.hour,
                b.weekday,
                b.is_weekend,
                b.bikes_now,
                b.docks_now,
                b.has_bike_now,
                b.empty_now,
                b.is_renting,
                b.is_returning,
                lag_5.bikes_lag_5m,
                lag_5.minutes_since_lag_5m,
                lag_15.bikes_lag_15m,
                lag_15.minutes_since_lag_15m,
                case
                    when lag_5.bikes_lag_5m is null then null
                    else b.bikes_now - lag_5.bikes_lag_5m
                end as bikes_delta_5m,
                case
                    when lag_15.bikes_lag_15m is null then null
                    else b.bikes_now - lag_15.bikes_lag_15m
                end as bikes_delta_15m,
                case
                    when b.capacity is null or b.capacity <= 0 then null
                    else round(b.bikes_now::double / b.capacity, 4)
                end as bikes_capacity_ratio,
                sin(2 * pi() * b.hour / 24.0) as hour_sin,
                cos(2 * pi() * b.hour / 24.0) as hour_cos,
                sin(2 * pi() * b.weekday / 7.0) as weekday_sin,
                cos(2 * pi() * b.weekday / 7.0) as weekday_cos,
                rolling_30.bikes_avg_30m,
                rolling_30.empty_rate_30m,
                rolling_30.samples_30m,
                b.capacity,
                b.lat,
                b.lon
            from base b
            left join lateral (
                select
                    coalesce(l.num_bikes_available, 0) as bikes_lag_5m,
                    date_diff('minute', l.collected_at, b.collected_at) as minutes_since_lag_5m
                from station_status_snapshots l
                where l.station_id = b.station_id
                    and l.collected_at <= b.collected_at - (5 * interval '1 minute')
                    and l.collected_at >= b.collected_at - (15 * interval '1 minute')
                order by l.collected_at desc
                limit 1
            ) lag_5 on true
            left join lateral (
                select
                    coalesce(l.num_bikes_available, 0) as bikes_lag_15m,
                    date_diff('minute', l.collected_at, b.collected_at) as minutes_since_lag_15m
                from station_status_snapshots l
                where l.station_id = b.station_id
                    and l.collected_at <= b.collected_at - (15 * interval '1 minute')
                    and l.collected_at >= b.collected_at - (25 * interval '1 minute')
                order by l.collected_at desc
                limit 1
            ) lag_15 on true
            left join lateral (
                select
                    round(avg(coalesce(r.num_bikes_available, 0)), 4) as bikes_avg_30m,
                    round(
                        avg(case when coalesce(r.num_bikes_available, 0) = 0 then 1.0 else 0.0 end),
                        4
                    ) as empty_rate_30m,
                    count(*) as samples_30m
                from station_status_snapshots r
                where r.station_id = b.station_id
                    and r.collected_at < b.collected_at
                    and r.collected_at >= b.collected_at - (30 * interval '1 minute')
            ) rolling_30 on true
            order by b.station_id
            """,
            station_ids,
        ).df()


def temporal_split(df: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if df.empty:
        raise ValueError("dataset is empty")

    timestamps = pd.Series(df["collected_at"].drop_duplicates().sort_values().to_list())
    if len(timestamps) < 2:
        raise ValueError("dataset needs at least two timestamps for temporal split")
    split_index = max(1, min(len(timestamps) - 1, int(len(timestamps) * (1 - test_fraction))))
    split_collected_at = timestamps.iloc[split_index]
    train_df = df[df["collected_at"] < split_collected_at].copy()
    test_df = df[df["collected_at"] >= split_collected_at].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("temporal split produced empty train or test set")
    return train_df, test_df, split_collected_at


def evaluate_baselines(
    *,
    db_path: Path,
    table_name: str,
    test_fraction: float,
    low_bike_threshold: int,
) -> EvaluationResult:
    df = load_dataset(db_path, table_name)
    train_df, test_df, split_collected_at = temporal_split(df, test_fraction)
    y_train = train_df[TARGET_COLUMN].astype(bool).to_numpy()
    y_test = test_df[TARGET_COLUMN].astype(bool).to_numpy()
    majority_empty = bool(np.mean(y_train) >= 0.5)

    station_rates = train_df.groupby("station_id")[TARGET_COLUMN].mean()
    fallback_rate = float(np.mean(y_train))
    station_probs = test_df["station_id"].map(station_rates).fillna(fallback_rate).to_numpy()

    baseline_predictions: list[tuple[str, np.ndarray, np.ndarray | None]] = [
        (
            "majority_train_class",
            np.full(len(test_df), majority_empty, dtype=bool),
            np.full(len(test_df), fallback_rate, dtype=float),
        ),
        (
            "persistence_empty_now",
            test_df["empty_now"].astype(bool).to_numpy(),
            test_df["empty_now"].astype(float).to_numpy(),
        ),
        (
            f"low_inventory_le_{low_bike_threshold}",
            (test_df["bikes_now"].fillna(0).to_numpy() <= low_bike_threshold),
            np.clip((low_bike_threshold + 1 - test_df["bikes_now"].fillna(0).to_numpy()) / (low_bike_threshold + 1), 0, 1),
        ),
        (
            "station_prior_empty_rate",
            station_probs >= 0.5,
            station_probs,
        ),
    ]

    metrics = [
        compute_metrics(name, y_test, prediction.astype(bool), probability)
        for name, prediction, probability in baseline_predictions
    ]
    return make_result(table_name, split_collected_at, train_df, test_df, metrics)


def train_models(
    *,
    db_path: Path,
    table_name: str,
    test_fraction: float,
    model_dir: Path | None,
) -> tuple[EvaluationResult, dict[str, Path]]:
    df = load_dataset(db_path, table_name)
    train_df, test_df, split_collected_at = temporal_split(df, test_fraction)
    y_train = train_df[TARGET_COLUMN].astype(bool)
    y_test = test_df[TARGET_COLUMN].astype(bool).to_numpy()

    models = {
        "logistic_regression": make_logistic_pipeline(),
        "hist_gradient_boosting": make_hist_gradient_pipeline(),
    }

    metrics: list[MetricRow] = []
    saved_models: dict[str, Path] = {}
    for name, model in models.items():
        model.fit(train_df[MODEL_FEATURES], y_train)
        prediction = model.predict(test_df[MODEL_FEATURES]).astype(bool)
        probability = prediction.astype(float)
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(test_df[MODEL_FEATURES])[:, 1]
        metrics.append(compute_metrics(name, y_test, prediction, probability))
        if model_dir is not None:
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"{name}.joblib"
            joblib.dump(model, model_path)
            saved_models[name] = model_path

    return make_result(table_name, split_collected_at, train_df, test_df, metrics), saved_models


def make_logistic_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs"),
            ),
        ]
    )


def make_hist_gradient_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        sparse_threshold=0.0,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06, random_state=42),
            ),
        ]
    )


def compute_metrics(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probability: np.ndarray | None,
) -> MetricRow:
    roc_auc = None
    average_precision = None
    if y_probability is not None and len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_probability))
        average_precision = float(average_precision_score(y_true, y_probability))
    return MetricRow(
        name=name,
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=roc_auc,
        average_precision=average_precision,
        positive_rate=float(np.mean(y_pred)),
    )


def make_result(
    table_name: str,
    split_collected_at: object,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metrics: list[MetricRow],
) -> EvaluationResult:
    return EvaluationResult(
        table_name=table_name,
        target=TARGET_COLUMN,
        split_collected_at=split_collected_at,
        train_rows=len(train_df),
        test_rows=len(test_df),
        train_window=f"{train_df['collected_at'].min()} -> {train_df['collected_at'].max()}",
        test_window=f"{test_df['collected_at'].min()} -> {test_df['collected_at'].max()}",
        metrics=metrics,
    )
