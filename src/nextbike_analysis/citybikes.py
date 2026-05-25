from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import duckdb
import httpx

CITYBIKES_DATA_URL = "https://data.citybik.es/dumps/by-network"


@dataclass(frozen=True)
class CitybikesFile:
    year: int
    name: str
    size: int
    mtime: str

    @property
    def url(self) -> str:
        return f"{CITYBIKES_DATA_URL}/{self.year}/{self.name}"


@dataclass(frozen=True)
class DownloadedCitybikesFile:
    file: CitybikesFile
    path: Path
    downloaded: bool


def list_available_years(timeout_seconds: float = 30.0) -> list[int]:
    with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
        response = client.get(f"{CITYBIKES_DATA_URL}/")
        response.raise_for_status()
        rows = response.json()
    return sorted(int(row["name"]) for row in rows if row.get("type") == "directory")


def list_network_files(
    network: str,
    years: Iterable[int] | None = None,
    timeout_seconds: float = 30.0,
) -> list[CitybikesFile]:
    selected_years = list(years) if years is not None else list_available_years(timeout_seconds)
    pattern = re.compile(rf"^\d{{6}}-{re.escape(network)}-stats\.parquet$")
    files: list[CitybikesFile] = []
    with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
        for year in selected_years:
            response = client.get(f"{CITYBIKES_DATA_URL}/{year}/")
            response.raise_for_status()
            for row in response.json():
                name = row["name"]
                if row.get("type") == "file" and pattern.match(name):
                    files.append(
                        CitybikesFile(
                            year=year,
                            name=name,
                            size=int(row["size"]),
                            mtime=str(row["mtime"]),
                        )
                    )
    return sorted(files, key=lambda item: item.name)


def download_network_files(
    network: str,
    data_dir: Path,
    years: Iterable[int] | None = None,
    timeout_seconds: float = 60.0,
) -> list[DownloadedCitybikesFile]:
    data_dir.mkdir(parents=True, exist_ok=True)
    files = list_network_files(network, years=years, timeout_seconds=timeout_seconds)
    results: list[DownloadedCitybikesFile] = []
    with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
        for file in files:
            path = data_dir / file.name
            if path.exists() and path.stat().st_size == file.size:
                results.append(DownloadedCitybikesFile(file=file, path=path, downloaded=False))
                continue

            tmp_path = path.with_suffix(path.suffix + ".tmp")
            with client.stream("GET", file.url) as response:
                response.raise_for_status()
                with tmp_path.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        handle.write(chunk)
            tmp_path.rename(path)
            results.append(DownloadedCitybikesFile(file=file, path=path, downloaded=True))
    return results


def network_parquet_glob(data_dir: Path, network: str) -> str:
    return str(data_dir / f"*-{network}-stats.parquet")


def ensure_network_files(data_dir: Path, network: str) -> None:
    if not list(data_dir.glob(f"*-{network}-stats.parquet")):
        raise FileNotFoundError(
            f"No CityBikes parquet files found for {network!r} in {data_dir}. "
            f"Run `nextbike citybikes-download --network {network}` first."
        )


def summarize_network_history(data_dir: Path, network: str) -> tuple[tuple[object, ...], list[tuple[object, ...]]]:
    ensure_network_files(data_dir, network)
    parquet_glob = network_parquet_glob(data_dir, network)
    with duckdb.connect() as con:
        summary = con.execute(
            f"""
            with r as (
                select * from read_parquet('{parquet_glob}', filename = true)
            )
            select
                count(*) as row_count,
                min(timestamp) as first_timestamp,
                max(timestamp) as latest_timestamp,
                count(distinct nuid) as stations,
                count(distinct name) as station_names,
                min(regexp_extract(filename, '([0-9]{{6}})-', 1)) as first_month,
                max(regexp_extract(filename, '([0-9]{{6}})-', 1)) as latest_month
            from r
            """
        ).fetchone()
        monthly = con.execute(
            f"""
            with r as (
                select
                    *,
                    regexp_extract(filename, '([0-9]{{6}})-', 1) as month_key
                from read_parquet('{parquet_glob}', filename = true)
            )
            select
                month_key,
                count(*) as row_count,
                count(distinct nuid) as stations,
                round(avg(bikes), 2) as avg_bikes_on_change,
                count_if(bikes = 0) as empty_change_rows
            from r
            group by month_key
            order by month_key
            """
        ).fetchall()
    return summary, monthly


def area_history(
    data_dir: Path,
    network: str,
    lat: float,
    lon: float,
    radius_m: int,
    sample_minutes: int,
) -> tuple[list[tuple[object, ...]], tuple[object, ...], list[tuple[object, ...]], list[tuple[object, ...]], list[tuple[object, ...]]]:
    ensure_network_files(data_dir, network)
    parquet_glob = network_parquet_glob(data_dir, network)
    query = f"""
    with r as (
        select * from read_parquet('{parquet_glob}', filename = true)
    ),
    station_latest as (
        select
            nuid,
            arg_max(name, timestamp) as name,
            arg_max(latitude, timestamp) as latitude,
            arg_max(longitude, timestamp) as longitude
        from r
        group by nuid
    ),
    local_stations as (
        select
            *,
            cast(round(
                6371000 * 2 * asin(sqrt(
                    power(sin(radians((latitude - {lat}) / 2)), 2)
                    + cos(radians({lat})) * cos(radians(latitude))
                    * power(sin(radians((longitude - {lon}) / 2)), 2)
                ))
            ) as integer) as distance_m
        from station_latest
        where latitude is not null and longitude is not null
    ),
    selected_stations as (
        select * from local_stations where distance_m <= {radius_m}
    ),
    bounds as (
        select min(timestamp) as min_ts, max(timestamp) as max_ts from r
    ),
    grid as (
        select sample_ts
        from bounds, generate_series(min_ts, max_ts, interval '{sample_minutes} minutes') as t(sample_ts)
    ),
    sampled as (
        select
            g.sample_ts,
            s.nuid,
            s.name,
            s.distance_m,
            state.bikes
        from grid g
        cross join selected_stations s
        left join lateral (
            select bikes
            from r
            where r.nuid = s.nuid and r.timestamp <= g.sample_ts
            order by r.timestamp desc
            limit 1
        ) state on true
    ),
    area_state as (
        select
            sample_ts,
            strftime(timezone('Europe/Prague', sample_ts at time zone 'UTC'), '%H:00') as local_hour,
            strftime(timezone('Europe/Prague', sample_ts at time zone 'UTC'), '%Y-%m') as local_month,
            sum(coalesce(bikes, 0)) as total_bikes,
            count_if(coalesce(bikes, 0) > 0) as stations_with_bikes,
            count_if(coalesce(bikes, 0) = 0) as empty_stations
        from sampled
        group by sample_ts
    )
    """
    with duckdb.connect() as con:
        stations = con.execute(
            query
            + """
            select nuid, name, distance_m
            from selected_stations
            order by distance_m
            """
        ).fetchall()
        summary = con.execute(
            query
            + """
            select
                count(*) as samples,
                min(sample_ts) as first_sample,
                max(sample_ts) as latest_sample,
                round(avg(total_bikes), 2) as avg_bikes,
                round(avg(stations_with_bikes), 2) as avg_stations_with_bikes,
                round(avg(case when stations_with_bikes = 0 then 1.0 else 0.0 end), 3)
                    as all_empty_rate
            from area_state
            """
        ).fetchone()
        hourly = con.execute(
            query
            + """
            select
                local_hour,
                count(*) as samples,
                round(avg(total_bikes), 2) as avg_bikes,
                round(avg(stations_with_bikes), 2) as avg_stations_with_bikes,
                round(avg(case when stations_with_bikes = 0 then 1.0 else 0.0 end), 3)
                    as all_empty_rate
            from area_state
            where local_hour between '04:00' and '12:00'
            group by local_hour
            order by local_hour
            """
        ).fetchall()
        monthly_morning = con.execute(
            query
            + """
            select
                local_month,
                count(*) as samples,
                round(avg(total_bikes), 2) as avg_bikes,
                round(avg(stations_with_bikes), 2) as avg_stations_with_bikes,
                round(avg(case when stations_with_bikes = 0 then 1.0 else 0.0 end), 3)
                    as all_empty_rate
            from area_state
            where local_hour between '06:00' and '10:00'
            group by local_month
            order by local_month
            """
        ).fetchall()
        station_rates = con.execute(
            query
            + """
            select
                name,
                any_value(distance_m) as distance_m,
                count(*) as samples,
                round(avg(coalesce(bikes, 0)), 2) as avg_bikes,
                round(avg(case when coalesce(bikes, 0) = 0 then 1.0 else 0.0 end), 3)
                    as empty_rate,
                round(avg(case when coalesce(bikes, 0) <= 1 then 1.0 else 0.0 end), 3)
                    as low_rate
            from sampled
            group by nuid, name
            order by distance_m
            """
        ).fetchall()
    return stations, summary, hourly, monthly_morning, station_rates
