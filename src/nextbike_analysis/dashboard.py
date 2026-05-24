from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Group
from rich.text import Text

from nextbike_analysis.boundary import BoundaryData, BoundaryPolygon, OSM_ATTRIBUTION
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


def render_dashboard(
    data: DashboardData,
    width: int,
    height: int,
    background: str,
    boundary: BoundaryData | None,
) -> Group:
    map_text = render_ascii_map(data.points, width, height, background, boundary)
    title = Text.assemble(
        ("Nextbike Brno dashboard", "bold"),
        ("  latest: ", "dim"),
        str(data.collected_at),
    )
    stats = Text(
        f"stations: {data.station_count} | shown: {len(data.points)} | "
        f"empty: {data.empty_stations} | bikes available: {data.bikes_available}"
    )
    legend_parts: list[str | tuple[str, str]] = [
        (".", "blue"),
        " with bikes  ",
        (".", "dim"),
        " empty  ",
        ("*", "bright_blue"),
        " multiple  ",
    ]
    if background == "osm" and boundary is not None:
        legend_parts.extend([(":#", "dim"), f" {boundary.name} boundary  "])
    elif background == "footprint":
        legend_parts.extend([(":,", "dim"), " footprint  "])
    legend_parts.append(
        ("Ctrl+C to quit", "dim"),
    )
    legend = Text.assemble(*legend_parts)
    if background == "osm" and boundary is not None:
        attribution = Text(f"{OSM_ATTRIBUTION} | OSM relation {boundary.osm_id}", style="dim")
        return Group(title, stats, legend, map_text, attribution)
    return Group(title, stats, legend, map_text)


def render_ascii_map(
    points: list[DashboardPoint],
    width: int,
    height: int,
    background: str,
    boundary: BoundaryData | None,
) -> Text:
    inner_width = max(width - 2, 10)
    inner_height = max(height - 2, 5)
    canvas = [[" " for _ in range(inner_width)] for _ in range(inner_height)]
    styles = [["" for _ in range(inner_width)] for _ in range(inner_height)]

    if points or boundary is not None:
        min_lat, max_lat, min_lon, max_lon = padded_bounds(points, boundary)
        if background == "osm" and boundary is not None:
            draw_osm_boundary(canvas, styles, boundary, min_lat, max_lat, min_lon, max_lon)
        elif background == "footprint" and points:
            draw_station_footprint(canvas, styles, points, min_lat, max_lat, min_lon, max_lon)

        for point in points:
            x, y = project_point(
                point.lat,
                point.lon,
                inner_width,
                inner_height,
                min_lat,
                max_lat,
                min_lon,
                max_lon,
            )
            if canvas[y][x] in {".", "*"}:
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


def padded_bounds(
    points: list[DashboardPoint],
    boundary: BoundaryData | None,
) -> tuple[float, float, float, float]:
    coords = [(point.lon, point.lat) for point in points]
    if boundary is not None:
        for polygon in boundary.polygons:
            coords.extend(polygon.outer)
            for hole in polygon.holes:
                coords.extend(hole)
    min_lon = min(lon for lon, _ in coords)
    max_lon = max(lon for lon, _ in coords)
    min_lat = min(lat for _, lat in coords)
    max_lat = max(lat for _, lat in coords)
    lat_padding = max((max_lat - min_lat) * 0.08, 0.001)
    lon_padding = max((max_lon - min_lon) * 0.08, 0.001)
    return (
        min_lat - lat_padding,
        max_lat + lat_padding,
        min_lon - lon_padding,
        max_lon + lon_padding,
    )


def project_point(
    lat: float,
    lon: float,
    width: int,
    height: int,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
) -> tuple[int, int]:
    x_ratio = (lon - min_lon) / (max_lon - min_lon)
    y_ratio = (max_lat - lat) / (max_lat - min_lat)
    x = min(width - 1, max(0, round(x_ratio * (width - 1))))
    y = min(height - 1, max(0, round(y_ratio * (height - 1))))
    return x, y


def draw_osm_boundary(
    canvas: list[list[str]],
    styles: list[list[str]],
    boundary: BoundaryData,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
) -> None:
    height = len(canvas)
    width = len(canvas[0])

    for y in range(height):
        lat = max_lat - (y / max(height - 1, 1)) * (max_lat - min_lat)
        for x in range(width):
            lon = min_lon + (x / max(width - 1, 1)) * (max_lon - min_lon)
            if any(point_in_boundary_polygon((lon, lat), polygon) for polygon in boundary.polygons):
                canvas[y][x] = ":"
                styles[y][x] = "dim"

    for polygon in boundary.polygons:
        draw_boundary_ring(canvas, styles, polygon.outer, min_lat, max_lat, min_lon, max_lon)


def draw_boundary_ring(
    canvas: list[list[str]],
    styles: list[list[str]],
    ring: list[tuple[float, float]],
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
) -> None:
    height = len(canvas)
    width = len(canvas[0])
    previous: tuple[int, int] | None = None
    for lon, lat in ring:
        x, y = project_point(lat, lon, width, height, min_lat, max_lat, min_lon, max_lon)
        if previous is not None:
            draw_line(canvas, styles, previous, (x, y), "#", "dim")
        previous = (x, y)


def draw_line(
    canvas: list[list[str]],
    styles: list[list[str]],
    start: tuple[int, int],
    end: tuple[int, int],
    char: str,
    style: str,
) -> None:
    start_x, start_y = start
    end_x, end_y = end
    steps = max(abs(end_x - start_x), abs(end_y - start_y), 1)
    for step in range(steps + 1):
        x = round(start_x + (end_x - start_x) * (step / steps))
        y = round(start_y + (end_y - start_y) * (step / steps))
        canvas[y][x] = char
        styles[y][x] = style


def point_in_boundary_polygon(
    point: tuple[float, float],
    polygon: BoundaryPolygon,
) -> bool:
    if not point_in_ring(point, polygon.outer):
        return False
    return not any(point_in_ring(point, hole) for hole in polygon.holes)


def point_in_ring(point: tuple[float, float], ring: list[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    previous_x, previous_y = ring[-1]
    for current_x, current_y in ring:
        if (current_y > y) != (previous_y > y):
            intersection_x = (previous_x - current_x) * (y - current_y) / (
                previous_y - current_y
            ) + current_x
            if x < intersection_x:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def draw_station_footprint(
    canvas: list[list[str]],
    styles: list[list[str]],
    points: list[DashboardPoint],
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
) -> None:
    height = len(canvas)
    width = len(canvas[0])
    if not points:
        return

    projected_points = [
        project_point_float(point.lat, point.lon, width, height, min_lat, max_lat, min_lon, max_lon)
        for point in points
    ]
    radius = max(2.4, min(width, height) * 0.14)
    soft_edge = radius + 1.7

    for y in range(height):
        for x in range(width):
            distance = min(
                ((x - point_x) ** 2 + (y - point_y) ** 2) ** 0.5
                for point_x, point_y in projected_points
            )
            if distance <= radius:
                canvas[y][x] = ":"
                styles[y][x] = "dim"
            elif distance <= soft_edge:
                canvas[y][x] = ","
                styles[y][x] = "dim"


def project_point_float(
    lat: float,
    lon: float,
    width: int,
    height: int,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
) -> tuple[float, float]:
    x_ratio = (lon - min_lon) / (max_lon - min_lon)
    y_ratio = (max_lat - lat) / (max_lat - min_lat)
    return x_ratio * (width - 1), y_ratio * (height - 1)
