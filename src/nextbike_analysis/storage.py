from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb


def utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def snapshot_id(ts: datetime) -> str:
    return ts.strftime("%Y%m%dT%H%M%SZ")


class SnapshotStore:
    def __init__(self, data_dir: Path, db_path: Path) -> None:
        self.data_dir = data_dir
        self.raw_dir = data_dir / "raw"
        self.db_path = db_path

    def write_snapshot(self, feeds: dict[str, dict[str, Any]], collected_at: datetime) -> Path:
        target_dir = self.raw_dir / collected_at.strftime("%Y/%m/%d")
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{snapshot_id(collected_at)}.json"
        payload = {
            "collected_at": collected_at.isoformat(),
            "feeds": feeds,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        return path

    def init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(str(self.db_path)) as con:
            con.execute(
                """
                create table if not exists station_status_snapshots (
                    collected_at timestamptz not null,
                    feed_last_updated bigint,
                    station_id varchar not null,
                    num_bikes_available integer,
                    num_docks_available integer,
                    is_installed boolean,
                    is_renting boolean,
                    is_returning boolean,
                    last_reported bigint,
                    vehicle_types_available json
                )
                """
            )
            con.execute(
                """
                create table if not exists free_bike_status_snapshots (
                    collected_at timestamptz not null,
                    feed_last_updated bigint,
                    bike_id varchar not null,
                    is_reserved boolean,
                    is_disabled boolean,
                    lat double,
                    lon double,
                    vehicle_type_id varchar,
                    station_id varchar,
                    pricing_plan_id varchar,
                    rental_uris json
                )
                """
            )
            con.execute(
                """
                create table if not exists station_information (
                    observed_at timestamptz not null,
                    feed_last_updated bigint,
                    station_id varchar not null,
                    name varchar,
                    short_name varchar,
                    lat double,
                    lon double,
                    region_id varchar,
                    is_virtual_station boolean,
                    capacity integer
                )
                """
            )
            con.execute(
                """
                create table if not exists collection_runs (
                    collected_at timestamptz primary key,
                    raw_path varchar not null,
                    station_count integer,
                    bikes_available integer,
                    free_bike_count integer
                )
                """
            )

    def append_normalized(
        self,
        feeds: dict[str, dict[str, Any]],
        collected_at: datetime,
        raw_path: Path,
    ) -> dict[str, int]:
        self.init_db()
        station_status = feeds.get("station_status", {})
        station_information = feeds.get("station_information", {})
        free_bike_status = feeds.get("free_bike_status", {})

        station_status_rows = station_status.get("data", {}).get("stations", [])
        station_info_rows = station_information.get("data", {}).get("stations", [])
        free_bike_rows = free_bike_status.get("data", {}).get("bikes", [])

        with duckdb.connect(str(self.db_path)) as con:
            con.executemany(
                """
                insert into station_status_snapshots values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        collected_at,
                        station_status.get("last_updated"),
                        row.get("station_id"),
                        row.get("num_bikes_available"),
                        row.get("num_docks_available"),
                        row.get("is_installed"),
                        row.get("is_renting"),
                        row.get("is_returning"),
                        row.get("last_reported"),
                        json.dumps(row.get("vehicle_types_available", []), ensure_ascii=False),
                    )
                    for row in station_status_rows
                ],
            )
            con.executemany(
                """
                insert into station_information values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        collected_at,
                        station_information.get("last_updated"),
                        row.get("station_id"),
                        row.get("name"),
                        row.get("short_name"),
                        row.get("lat"),
                        row.get("lon"),
                        row.get("region_id"),
                        row.get("is_virtual_station"),
                        row.get("capacity"),
                    )
                    for row in station_info_rows
                ],
            )
            if free_bike_rows:
                con.executemany(
                    """
                    insert into free_bike_status_snapshots values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            collected_at,
                            free_bike_status.get("last_updated"),
                            row.get("bike_id"),
                            row.get("is_reserved"),
                            row.get("is_disabled"),
                            row.get("lat"),
                            row.get("lon"),
                            row.get("vehicle_type_id"),
                            row.get("station_id"),
                            row.get("pricing_plan_id"),
                            json.dumps(row.get("rental_uris", {}), ensure_ascii=False),
                        )
                        for row in free_bike_rows
                    ],
                )
            station_count = len(station_status_rows)
            bikes_available = sum(row.get("num_bikes_available") or 0 for row in station_status_rows)
            free_bike_count = len(free_bike_rows)
            con.execute(
                """
                insert into collection_runs values (?, ?, ?, ?, ?)
                """,
                (collected_at, str(raw_path), station_count, bikes_available, free_bike_count),
            )

        return {
            "station_count": station_count,
            "bikes_available": bikes_available,
            "free_bike_count": free_bike_count,
        }
