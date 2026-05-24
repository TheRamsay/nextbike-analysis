from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb

from nextbike_analysis.db import connect_db
from nextbike_analysis.formatting import directory_size_bytes


@dataclass(frozen=True)
class DataHealth:
    runs: int
    window: str
    expected_samples: int | None
    coverage_pct: float | None
    first_collected_at: object
    latest_collected_at: object
    station_rows: int
    free_bike_rows: int | None
    distinct_stations: int
    station_count_min_avg_max: str
    bikes_available_min_avg_max: str
    duplicate_collected_at: int
    gap_count: int
    interval_min_avg_max_seconds: str
    db_size_mb: float
    raw_size_mb: float
    gaps: list[tuple[object, object, int]]


def get_data_health(
    *,
    db_path: Path,
    data_dir: Path,
    expected_interval_seconds: int,
    gap_threshold_seconds: int,
    max_gaps: int,
    since_hours: float | None,
) -> DataHealth:
    time_filter = ""
    params: list[float | int] = []
    if since_hours is not None:
        time_filter = "where collected_at >= now() - (? * interval '1 hour')"
        params.append(since_hours)

    with connect_db(db_path) as con:
        summary = con.execute(
            f"""
            select
                count(*) as runs,
                min(collected_at) as first_collected_at,
                max(collected_at) as latest_collected_at,
                min(station_count) as min_station_count,
                max(station_count) as max_station_count,
                round(avg(station_count), 2) as avg_station_count,
                min(bikes_available) as min_bikes_available,
                max(bikes_available) as max_bikes_available,
                round(avg(bikes_available), 2) as avg_bikes_available,
                count(*) - count(distinct collected_at) as duplicate_collected_at
            from collection_runs
            {time_filter}
            """,
            params,
        ).fetchone()
        station_rows = con.execute(
            f"""
            select count(*)
            from station_status_snapshots
            {time_filter}
            """,
            params,
        ).fetchone()[0]
        try:
            free_bike_rows = con.execute(
                f"""
                select count(*)
                from free_bike_status_snapshots
                {time_filter}
                """,
                params,
            ).fetchone()[0]
        except duckdb.CatalogException:
            free_bike_rows = None
        distinct_stations = con.execute(
            f"""
            select count(distinct station_id)
            from station_status_snapshots
            {time_filter}
            """,
            params,
        ).fetchone()[0]
        gap_summary = con.execute(
            f"""
            with ordered as (
                select
                    collected_at,
                    lag(collected_at) over (order by collected_at) as previous_collected_at
                from collection_runs
                {time_filter}
            ),
            gaps as (
                select
                    previous_collected_at,
                    collected_at,
                    date_diff('second', previous_collected_at, collected_at) as gap_seconds
                from ordered
                where previous_collected_at is not null
            )
            select
                count_if(gap_seconds > ?) as gap_count,
                max(gap_seconds) as max_gap_seconds,
                round(avg(gap_seconds), 2) as avg_interval_seconds,
                min(gap_seconds) as min_interval_seconds
            from gaps
            """,
            [*params, gap_threshold_seconds],
        ).fetchone()
        gaps = con.execute(
            f"""
            with ordered as (
                select
                    collected_at,
                    lag(collected_at) over (order by collected_at) as previous_collected_at
                from collection_runs
                {time_filter}
            )
            select
                previous_collected_at,
                collected_at,
                date_diff('second', previous_collected_at, collected_at) as gap_seconds
            from ordered
            where previous_collected_at is not null
                and date_diff('second', previous_collected_at, collected_at) > ?
            order by gap_seconds desc, collected_at desc
            limit ?
            """,
            [*params, gap_threshold_seconds, max_gaps],
        ).fetchall()

    runs = summary[0] or 0
    first_collected_at = summary[1]
    latest_collected_at = summary[2]
    expected_samples = None
    coverage_pct = None
    if first_collected_at is not None and latest_collected_at is not None:
        elapsed_seconds = (latest_collected_at - first_collected_at).total_seconds()
        expected_samples = int(elapsed_seconds // expected_interval_seconds) + 1
        coverage_pct = round((runs / expected_samples) * 100, 2) if expected_samples else None

    db_size = db_path.stat().st_size if db_path.exists() else 0
    raw_size = directory_size_bytes(data_dir / "raw")

    return DataHealth(
        runs=runs,
        window=f"last {since_hours}h" if since_hours is not None else "all",
        expected_samples=expected_samples,
        coverage_pct=coverage_pct,
        first_collected_at=first_collected_at,
        latest_collected_at=latest_collected_at,
        station_rows=station_rows,
        free_bike_rows=free_bike_rows,
        distinct_stations=distinct_stations,
        station_count_min_avg_max=f"{summary[3]}/{summary[5]}/{summary[4]}",
        bikes_available_min_avg_max=f"{summary[6]}/{summary[8]}/{summary[7]}",
        duplicate_collected_at=summary[9],
        gap_count=gap_summary[0] or 0,
        interval_min_avg_max_seconds=f"{gap_summary[3]}/{gap_summary[2]}/{gap_summary[1]}",
        db_size_mb=db_size / 1024 / 1024,
        raw_size_mb=raw_size / 1024 / 1024,
        gaps=gaps,
    )


def get_system_trend(db_path: Path, limit: int) -> list[tuple[object, int, int, int, float, int]]:
    with connect_db(db_path) as con:
        return con.execute(
            """
            select
                r.collected_at,
                r.bikes_available,
                r.free_bike_count,
                count_if(s.num_bikes_available = 0) as empty_stations,
                round(avg(s.num_bikes_available), 2) as avg_bikes_per_station,
                max(s.num_bikes_available) as max_bikes_at_station
            from collection_runs r
            join station_status_snapshots s using (collected_at)
            group by r.collected_at, r.bikes_available, r.free_bike_count
            order by r.collected_at desc
            limit ?
            """,
            [limit],
        ).fetchall()

