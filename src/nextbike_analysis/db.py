from __future__ import annotations

from pathlib import Path

import duckdb
import typer


LATEST_STATION_INFO_SQL = """
select station_id, name, short_name, region_id, lat, lon
from (
    select
        station_id,
        name,
        short_name,
        region_id,
        lat,
        lon,
        row_number() over (partition by station_id order by observed_at desc) as rn
    from station_information
)
where rn = 1
"""


def connect_db(db_path: Path) -> duckdb.DuckDBPyConnection:
    if not db_path.exists():
        raise typer.BadParameter(f"Database does not exist: {db_path}")
    return duckdb.connect(str(db_path), read_only=True)

