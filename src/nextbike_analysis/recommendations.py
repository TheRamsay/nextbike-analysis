from __future__ import annotations

from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import numpy as np

from nextbike_analysis.modeling import MODEL_FEATURES, load_dataset, load_model, temporal_split


@dataclass(frozen=True)
class Origin:
    name: str
    lat: float
    lon: float


@dataclass(frozen=True)
class RecommendationMetric:
    strategy: str
    label: str
    attempts: int
    no_candidate: int
    success_rate: float | None
    avg_distance_m: float | None
    p90_distance_m: float | None
    avg_empty_probability: float | None
    avg_extra_distance_m: float | None


@dataclass(frozen=True)
class RecommendationEvaluation:
    origins: list[Origin]
    timestamps: int
    attempts_per_strategy: int
    metrics: list[RecommendationMetric]


DEFAULT_ORIGINS = [
    Origin("Moravske namesti", 49.199088, 16.607361),
    Origin("Ceska", 49.197050, 16.606837),
    Origin("Hlavni nadrazi", 49.190331, 16.612873),
    Origin("Luzanky", 49.206228, 16.606837),
    Origin("Mendlovo namesti", 49.190181, 16.594856),
]

STRATEGY_LABELS = {
    "nearest_with_bike": "nearest",
    "nearest_bikes_ge_2": "bikes>=2",
    "nearest_model_low_risk": "model<0.4",
    "distance_plus_risk_penalty": "dist+risk",
}


def evaluate_recommendation_strategies(
    *,
    db_path: Path,
    model_path: Path,
    table_name: str,
    test_fraction: float,
    max_distance_m: float,
) -> RecommendationEvaluation:
    df = load_dataset(db_path, table_name)
    _, test_df, _ = temporal_split(df, test_fraction)
    model = load_model(model_path)
    test_df = test_df.copy()
    test_df["empty_probability"] = model.predict_proba(test_df[MODEL_FEATURES])[:, 1]

    timestamps = list(test_df["collected_at"].drop_duplicates().sort_values())
    rows: list[dict[str, object]] = []
    for collected_at, snapshot in test_df.groupby("collected_at", sort=True):
        snapshot = snapshot.copy()
        for origin in DEFAULT_ORIGINS:
            candidates = snapshot[
                (snapshot["has_bike_now"].astype(bool))
                & snapshot["lat"].notna()
                & snapshot["lon"].notna()
            ].copy()
            if candidates.empty:
                rows.extend(no_candidate_rows(collected_at, origin))
                continue
            candidates["distance_m"] = [
                haversine_m(origin.lat, origin.lon, float(lat), float(lon))
                for lat, lon in zip(candidates["lat"], candidates["lon"], strict=True)
            ]
            candidates = candidates[candidates["distance_m"] <= max_distance_m]
            if candidates.empty:
                rows.extend(no_candidate_rows(collected_at, origin))
                continue

            nearest_distance = float(candidates["distance_m"].min())
            rows.append(evaluate_strategy("nearest_with_bike", collected_at, origin, candidates, nearest_distance))
            rows.append(
                evaluate_strategy(
                    "nearest_bikes_ge_2",
                    collected_at,
                    origin,
                    candidates[candidates["bikes_now"] >= 2],
                    nearest_distance,
                )
            )
            rows.append(
                evaluate_strategy(
                    "nearest_model_low_risk",
                    collected_at,
                    origin,
                    candidates[candidates["empty_probability"] < 0.4],
                    nearest_distance,
                )
            )
            scored = candidates.copy()
            scored["score"] = scored["distance_m"] + scored["empty_probability"] * 350.0
            rows.append(
                evaluate_strategy(
                    "distance_plus_risk_penalty",
                    collected_at,
                    origin,
                    scored,
                    nearest_distance,
                    order_by="score",
                )
            )

    metrics = summarize_rows(rows)
    return RecommendationEvaluation(
        origins=DEFAULT_ORIGINS,
        timestamps=len(timestamps),
        attempts_per_strategy=len(timestamps) * len(DEFAULT_ORIGINS),
        metrics=metrics,
    )


def evaluate_strategy(
    strategy: str,
    collected_at: object,
    origin: Origin,
    candidates,
    nearest_distance: float,
    order_by: str = "distance_m",
) -> dict[str, object]:
    if candidates.empty:
        return {
            "strategy": strategy,
            "collected_at": collected_at,
            "origin": origin.name,
            "success": None,
            "distance_m": None,
            "empty_probability": None,
            "extra_distance_m": None,
            "no_candidate": True,
        }
    selected = candidates.sort_values([order_by, "distance_m", "station_id"]).iloc[0]
    return {
        "strategy": strategy,
        "collected_at": collected_at,
        "origin": origin.name,
        "success": bool(selected["has_bike_future"]),
        "distance_m": float(selected["distance_m"]),
        "empty_probability": float(selected["empty_probability"]),
        "extra_distance_m": float(selected["distance_m"] - nearest_distance),
        "no_candidate": False,
    }


def no_candidate_rows(collected_at: object, origin: Origin) -> list[dict[str, object]]:
    return [
        {
            "strategy": strategy,
            "collected_at": collected_at,
            "origin": origin.name,
            "success": None,
            "distance_m": None,
            "empty_probability": None,
            "extra_distance_m": None,
            "no_candidate": True,
        }
        for strategy in [
            "nearest_with_bike",
            "nearest_bikes_ge_2",
            "nearest_model_low_risk",
            "distance_plus_risk_penalty",
        ]
    ]


def summarize_rows(rows: list[dict[str, object]]) -> list[RecommendationMetric]:
    metrics: list[RecommendationMetric] = []
    strategies = sorted({str(row["strategy"]) for row in rows})
    for strategy in strategies:
        strategy_rows = [row for row in rows if row["strategy"] == strategy]
        valid_rows = [row for row in strategy_rows if not row["no_candidate"]]
        distances = [float(row["distance_m"]) for row in valid_rows]
        probabilities = [float(row["empty_probability"]) for row in valid_rows]
        extra_distances = [float(row["extra_distance_m"]) for row in valid_rows]
        successes = [bool(row["success"]) for row in valid_rows]
        metrics.append(
            RecommendationMetric(
                strategy=strategy,
                label=STRATEGY_LABELS.get(strategy, strategy),
                attempts=len(strategy_rows),
                no_candidate=len(strategy_rows) - len(valid_rows),
                success_rate=float(np.mean(successes)) if successes else None,
                avg_distance_m=float(np.mean(distances)) if distances else None,
                p90_distance_m=float(np.percentile(distances, 90)) if distances else None,
                avg_empty_probability=float(np.mean(probabilities)) if probabilities else None,
                avg_extra_distance_m=float(np.mean(extra_distances)) if extra_distances else None,
            )
        )
    return metrics


def haversine_m(origin_lat: float, origin_lon: float, station_lat: float, station_lon: float) -> float:
    lat_delta = radians(station_lat - origin_lat)
    lon_delta = radians(station_lon - origin_lon)
    a = (
        sin(lat_delta / 2) ** 2
        + cos(radians(origin_lat)) * cos(radians(station_lat)) * sin(lon_delta / 2) ** 2
    )
    return 2 * 6371000 * asin(sqrt(a))
