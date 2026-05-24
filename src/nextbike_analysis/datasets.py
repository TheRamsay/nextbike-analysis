from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import duckdb


VALID_TABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class DatasetBuildResult:
    table_name: str
    horizon_minutes: int
    max_target_delay_minutes: int
    lag_tolerance_minutes: int
    rows: int
    stations: int
    first_collected_at: object
    latest_collected_at: object
    null_lag_5m: int
    null_lag_15m: int
    positive_rate: float | None
    empty_future_rate: float | None
    avg_minutes_to_target: float | None


def validate_table_name(table_name: str) -> str:
    if not VALID_TABLE_NAME.fullmatch(table_name):
        raise ValueError("table name must use only letters, numbers, and underscores")
    return table_name


def build_station_availability_dataset(
    *,
    db_path: Path,
    table_name: str,
    horizon_minutes: int,
    max_target_delay_minutes: int,
    lag_tolerance_minutes: int,
) -> DatasetBuildResult:
    if not db_path.exists():
        raise FileNotFoundError(f"Database does not exist: {db_path}")
    if horizon_minutes <= 0:
        raise ValueError("horizon_minutes must be positive")
    if max_target_delay_minutes < horizon_minutes:
        raise ValueError("max_target_delay_minutes must be >= horizon_minutes")
    if lag_tolerance_minutes < 0:
        raise ValueError("lag_tolerance_minutes cannot be negative")

    table_name = validate_table_name(table_name)
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            f"""
            create or replace table {table_name} as
            with station_info as (
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
                    i.name as station_name,
                    i.short_name,
                    i.region_id,
                    i.lat,
                    i.lon,
                    i.capacity,
                    cast(strftime(s.collected_at, '%H') as integer) as hour,
                    cast(strftime(s.collected_at, '%w') as integer) as weekday,
                    cast(strftime(s.collected_at, '%w') as integer) in (0, 6) as is_weekend
                from station_status_snapshots s
                left join station_info i using (station_id)
            )
            select
                b.station_id,
                b.station_name,
                b.short_name,
                b.region_id,
                b.lat,
                b.lon,
                b.capacity,
                b.collected_at,
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
                future.collected_at as target_collected_at,
                date_diff('minute', b.collected_at, future.collected_at) as minutes_to_target,
                future.bikes_future,
                future.bikes_future > 0 as has_bike_future,
                future.bikes_future = 0 as empty_future
            from base b
            join lateral (
                select
                    f.collected_at,
                    coalesce(f.num_bikes_available, 0) as bikes_future
                from station_status_snapshots f
                where f.station_id = b.station_id
                    and f.collected_at >= b.collected_at + (? * interval '1 minute')
                    and f.collected_at <= b.collected_at + (? * interval '1 minute')
                order by f.collected_at
                limit 1
            ) future on true
            left join lateral (
                select
                    coalesce(l.num_bikes_available, 0) as bikes_lag_5m,
                    date_diff('minute', l.collected_at, b.collected_at) as minutes_since_lag_5m
                from station_status_snapshots l
                where l.station_id = b.station_id
                    and l.collected_at <= b.collected_at - (5 * interval '1 minute')
                    and l.collected_at >= b.collected_at - (? * interval '1 minute')
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
                    and l.collected_at >= b.collected_at - (? * interval '1 minute')
                order by l.collected_at desc
                limit 1
            ) lag_15 on true
            """,
            [
                horizon_minutes,
                max_target_delay_minutes,
                5 + lag_tolerance_minutes,
                15 + lag_tolerance_minutes,
            ],
        )
        summary = con.execute(
            f"""
            select
                count(*) as rows,
                count(distinct station_id) as stations,
                min(collected_at) as first_collected_at,
                max(collected_at) as latest_collected_at,
                count_if(bikes_lag_5m is null) as null_lag_5m,
                count_if(bikes_lag_15m is null) as null_lag_15m,
                round(avg(case when has_bike_future then 1.0 else 0.0 end), 4)
                    as positive_rate,
                round(avg(case when empty_future then 1.0 else 0.0 end), 4)
                    as empty_future_rate,
                round(avg(minutes_to_target), 2) as avg_minutes_to_target
            from {table_name}
            """,
        ).fetchone()

    return DatasetBuildResult(
        table_name=table_name,
        horizon_minutes=horizon_minutes,
        max_target_delay_minutes=max_target_delay_minutes,
        lag_tolerance_minutes=lag_tolerance_minutes,
        rows=int(summary[0] or 0),
        stations=int(summary[1] or 0),
        first_collected_at=summary[2],
        latest_collected_at=summary[3],
        null_lag_5m=int(summary[4] or 0),
        null_lag_15m=int(summary[5] or 0),
        positive_rate=summary[6],
        empty_future_rate=summary[7],
        avg_minutes_to_target=summary[8],
    )
