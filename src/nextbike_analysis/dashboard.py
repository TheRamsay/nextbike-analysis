from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Group
from rich.text import Text

from nextbike_analysis.db import LATEST_STATION_INFO_SQL, connect_db


@dataclass(frozen=True)
class DashboardPoint:
    station_id: str
    name: str
    region_id: str | None
    lat: float
    lon: float
    bikes: int
    docks: int | None


@dataclass(frozen=True)
class DashboardData:
    collected_at: object
    station_count: int
    bikes_available: int
    empty_stations: int
    points: list[DashboardPoint]


def load_dashboard_data(db_path: Path, include_empty: bool) -> DashboardData:
    bike_filter = "" if include_empty else "and s.num_bikes_available > 0"
    with connect_db(db_path) as con:
        summary = con.execute(
            """
            with latest_run as (
                select collected_at
                from collection_runs
                order by collected_at desc
                limit 1
            )
            select
                lr.collected_at,
                count(*) as station_count,
                coalesce(sum(s.num_bikes_available), 0) as bikes_available,
                count_if(coalesce(s.num_bikes_available, 0) = 0) as empty_stations
            from latest_run lr
            join station_status_snapshots s using (collected_at)
            group by lr.collected_at
            """,
        ).fetchone()
        rows = con.execute(
            f"""
            with latest_run as (
                select collected_at
                from collection_runs
                order by collected_at desc
                limit 1
            ),
            station_info as ({LATEST_STATION_INFO_SQL})
            select
                s.station_id,
                coalesce(i.name, s.station_id) as name,
                i.region_id,
                i.lat,
                i.lon,
                coalesce(s.num_bikes_available, 0) as bikes,
                s.num_docks_available as docks
            from station_status_snapshots s
            join latest_run lr using (collected_at)
            left join station_info i using (station_id)
            where i.lat is not null
                and i.lon is not null
                {bike_filter}
            order by name
            """,
        ).fetchall()

    if summary is None:
        return DashboardData(
            collected_at=None,
            station_count=0,
            bikes_available=0,
            empty_stations=0,
            points=[],
        )

    points = [
        DashboardPoint(
            station_id=str(row[0]),
            name=str(row[1]),
            region_id=row[2],
            lat=float(row[3]),
            lon=float(row[4]),
            bikes=int(row[5]),
            docks=row[6],
        )
        for row in rows
    ]
    return DashboardData(
        collected_at=summary[0],
        station_count=int(summary[1]),
        bikes_available=int(summary[2]),
        empty_stations=int(summary[3]),
        points=points,
    )


def render_dashboard(data: DashboardData, width: int, height: int) -> Group:
    map_text = render_ascii_map(data.points, width, height)
    title = Text.assemble(
        ("Nextbike Brno dashboard", "bold"),
        ("  latest: ", "dim"),
        str(data.collected_at),
    )
    stats = Text(
        f"stations: {data.station_count} | shown: {len(data.points)} | "
        f"empty: {data.empty_stations} | bikes available: {data.bikes_available}"
    )
    legend = Text.assemble(
        (".", "blue"),
        " with bikes  ",
        (".", "dim"),
        " empty  ",
        ("*", "bright_blue"),
        " multiple  ",
        ("Ctrl+C to quit", "dim"),
    )
    return Group(title, stats, legend, map_text)


def render_ascii_map(points: list[DashboardPoint], width: int, height: int) -> Text:
    inner_width = max(width - 2, 10)
    inner_height = max(height - 2, 5)
    canvas = [[" " for _ in range(inner_width)] for _ in range(inner_height)]
    styles = [["" for _ in range(inner_width)] for _ in range(inner_height)]

    if points:
        min_lat = min(point.lat for point in points)
        max_lat = max(point.lat for point in points)
        min_lon = min(point.lon for point in points)
        max_lon = max(point.lon for point in points)
        lat_padding = max((max_lat - min_lat) * 0.08, 0.001)
        lon_padding = max((max_lon - min_lon) * 0.08, 0.001)
        min_lat -= lat_padding
        max_lat += lat_padding
        min_lon -= lon_padding
        max_lon += lon_padding

        for point in points:
            x_ratio = (point.lon - min_lon) / (max_lon - min_lon)
            y_ratio = (max_lat - point.lat) / (max_lat - min_lat)
            x = min(inner_width - 1, max(0, round(x_ratio * (inner_width - 1))))
            y = min(inner_height - 1, max(0, round(y_ratio * (inner_height - 1))))
            if canvas[y][x] != " ":
                canvas[y][x] = "*"
                styles[y][x] = "bright_blue"
            elif point.bikes > 0:
                canvas[y][x] = "."
                styles[y][x] = "blue"
            else:
                canvas[y][x] = "."
                styles[y][x] = "dim"

    text = Text()
    text.append("+" + "-" * inner_width + "+\n", style="dim")
    for row, style_row in zip(canvas, styles, strict=True):
        text.append("|", style="dim")
        for char, style in zip(row, style_row, strict=True):
            text.append(char, style=style)
        text.append("|\n", style="dim")
    text.append("+" + "-" * inner_width + "+", style="dim")
    return text
