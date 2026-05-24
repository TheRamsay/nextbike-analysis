from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import duckdb
import httpx
import typer
from rich.console import Console
from rich.table import Table

from nextbike_analysis.config import Settings
from nextbike_analysis.gbfs import DEFAULT_FEEDS, GbfsClient
from nextbike_analysis.storage import SnapshotStore, utc_now

app = typer.Typer(no_args_is_help=True)
console = Console()


def make_settings(
    gbfs_url: str | None,
    data_dir: Path | None,
    db_path: Path | None,
) -> Settings:
    defaults = Settings()
    return Settings(
        gbfs_url=gbfs_url or defaults.gbfs_url,
        data_dir=data_dir or defaults.data_dir,
        db_path=db_path or defaults.db_path,
        request_timeout_seconds=defaults.request_timeout_seconds,
    )


def connect_db(db_path: Path) -> duckdb.DuckDBPyConnection:
    if not db_path.exists():
        raise typer.BadParameter(f"Database does not exist: {db_path}")
    return duckdb.connect(str(db_path), read_only=True)


def pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_pid(pid_path: Path) -> int | None:
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def tail_lines(path: Path, line_count: int = 5) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-line_count:]


def poller_paths(data_dir: Path) -> tuple[Path, Path]:
    return data_dir / "poller.pid", data_dir / "poller.log"


def bike_risk(num_bikes_available: int) -> str:
    if num_bikes_available <= 0:
        return "empty"
    if num_bikes_available == 1:
        return "high"
    if num_bikes_available <= 3:
        return "medium"
    return "low"


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


def get_ip_location(timeout_seconds: float) -> tuple[float, float, str]:
    providers = (
        (
            "ipapi.co",
            "https://ipapi.co/json/",
            lambda data: (
                data.get("latitude"),
                data.get("longitude"),
                data.get("city"),
                data.get("country_name"),
            ),
        ),
        (
            "ipwho.is",
            "https://ipwho.is/",
            lambda data: (
                data.get("latitude"),
                data.get("longitude"),
                data.get("city"),
                data.get("country"),
            ),
        ),
    )
    errors: list[str] = []
    with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
        for provider_name, url, parser in providers:
            try:
                response = client.get(url)
                response.raise_for_status()
                data = response.json()
                lat, lon, city, country = parser(data)
                if lat is None or lon is None:
                    errors.append(f"{provider_name}: missing latitude/longitude")
                    continue
                label = f"{provider_name} approximate location ({city}, {country})"
                return float(lat), float(lon), label
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{provider_name}: {exc}")

    raise typer.BadParameter("IP geolocation failed: " + "; ".join(errors))


def get_address_location(address: str, timeout_seconds: float) -> tuple[float, float, str]:
    headers = {"User-Agent": "nextbike-analysis/0.1 (local CLI geocoder)"}
    params = {
        "q": address,
        "format": "jsonv2",
        "limit": 1,
        "addressdetails": 1,
    }
    with httpx.Client(timeout=timeout_seconds, follow_redirects=True, headers=headers) as client:
        response = client.get("https://nominatim.openstreetmap.org/search", params=params)
        response.raise_for_status()
        results = response.json()

    if not results:
        raise typer.BadParameter(f"Address not found: {address}")

    result = results[0]
    display_name = result.get("display_name", address)
    return float(result["lat"]), float(result["lon"]), f"Nominatim address match ({display_name})"


@app.command()
def info(
    gbfs_url: Annotated[str | None, typer.Option(help="GBFS discovery URL.")] = None,
    language: Annotated[str, typer.Option(help="GBFS language key.")] = "en",
) -> None:
    """Show available feeds for the configured GBFS system."""
    settings = make_settings(gbfs_url, None, None)
    client = GbfsClient(settings.gbfs_url, settings.request_timeout_seconds)
    feeds = client.discover_feeds(language)

    table = Table(title=f"GBFS feeds ({language})")
    table.add_column("Feed")
    table.add_column("URL")
    for feed in feeds:
        table.add_row(feed.name, feed.url)
    console.print(table)


def collect_once(settings: Settings, language: str, include_free_bikes: bool) -> dict[str, int | Path]:
    client = GbfsClient(settings.gbfs_url, settings.request_timeout_seconds)
    feed_names = list(DEFAULT_FEEDS)
    if not include_free_bikes:
        feed_names.remove("free_bike_status")

    collected_at = utc_now()
    feeds = client.fetch_feeds(language=language, names=feed_names)
    store = SnapshotStore(settings.data_dir, settings.db_path)
    raw_path = store.write_snapshot(feeds, collected_at)
    metrics = store.append_normalized(feeds, collected_at, raw_path)
    return {**metrics, "raw_path": raw_path}


@app.command()
def collect(
    gbfs_url: Annotated[str | None, typer.Option(help="GBFS discovery URL.")] = None,
    data_dir: Annotated[Path | None, typer.Option(help="Directory for raw data.")] = None,
    db_path: Annotated[Path | None, typer.Option(help="DuckDB file path.")] = None,
    language: Annotated[str, typer.Option(help="GBFS language key.")] = "en",
    include_free_bikes: Annotated[
        bool,
        typer.Option(help="Store free_bike_status raw snapshots and aggregate counts."),
    ] = True,
) -> None:
    """Collect one GBFS snapshot and append normalized station status rows."""
    settings = make_settings(gbfs_url, data_dir, db_path)
    metrics = collect_once(settings, language, include_free_bikes)
    console.print(
        "[green]collected[/green] "
        f"stations={metrics['station_count']} "
        f"bikes_available={metrics['bikes_available']} "
        f"free_bikes={metrics['free_bike_count']} "
        f"raw={metrics['raw_path']}"
    )


@app.command()
def poll(
    gbfs_url: Annotated[str | None, typer.Option(help="GBFS discovery URL.")] = None,
    data_dir: Annotated[Path | None, typer.Option(help="Directory for raw data.")] = None,
    db_path: Annotated[Path | None, typer.Option(help="DuckDB file path.")] = None,
    language: Annotated[str, typer.Option(help="GBFS language key.")] = "en",
    interval_seconds: Annotated[int, typer.Option(help="Delay between snapshots.")] = 60,
    max_samples: Annotated[int | None, typer.Option(help="Stop after this many snapshots.")] = None,
    include_free_bikes: Annotated[
        bool,
        typer.Option(help="Store free_bike_status raw snapshots and aggregate counts."),
    ] = True,
) -> None:
    """Collect GBFS snapshots repeatedly."""
    if interval_seconds <= 0:
        raise typer.BadParameter("interval_seconds must be positive")

    settings = make_settings(gbfs_url, data_dir, db_path)
    sample = 0
    while max_samples is None or sample < max_samples:
        sample += 1
        try:
            metrics = collect_once(settings, language, include_free_bikes)
            console.print(
                "[green]collected[/green] "
                f"sample={sample} "
                f"stations={metrics['station_count']} "
                f"bikes_available={metrics['bikes_available']} "
                f"free_bikes={metrics['free_bike_count']}"
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]collection failed[/red] sample={sample} error={exc}")

        if max_samples is not None and sample >= max_samples:
            break
        time.sleep(interval_seconds)


@app.command("poller-start")
def poller_start(
    gbfs_url: Annotated[str | None, typer.Option(help="GBFS discovery URL.")] = None,
    data_dir: Annotated[Path | None, typer.Option(help="Directory for raw data.")] = None,
    db_path: Annotated[Path | None, typer.Option(help="DuckDB file path.")] = None,
    language: Annotated[str, typer.Option(help="GBFS language key.")] = "en",
    interval_seconds: Annotated[int, typer.Option(help="Delay between snapshots.")] = 60,
    include_free_bikes: Annotated[
        bool,
        typer.Option(help="Store free_bike_status raw snapshots and aggregate counts."),
    ] = False,
    force: Annotated[bool, typer.Option(help="Replace a stale PID file if present.")] = False,
) -> None:
    """Start the GBFS poller as a background process."""
    if interval_seconds <= 0:
        raise typer.BadParameter("interval_seconds must be positive")

    settings = make_settings(gbfs_url, data_dir, db_path)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    pid_path, log_path = poller_paths(settings.data_dir)

    existing_pid = read_pid(pid_path)
    if existing_pid is not None and pid_is_running(existing_pid):
        console.print(f"[yellow]poller already running[/yellow] pid={existing_pid}")
        raise typer.Exit(code=0)
    if existing_pid is not None and not force:
        console.print(f"[red]stale PID file exists[/red] pid={existing_pid} path={pid_path}")
        console.print("Run with --force to replace it.")
        raise typer.Exit(code=1)

    command = [
        sys.argv[0],
        "poll",
        "--gbfs-url",
        settings.gbfs_url,
        "--data-dir",
        str(settings.data_dir),
        "--db-path",
        str(settings.db_path),
        "--language",
        language,
        "--interval-seconds",
        str(interval_seconds),
        "--include-free-bikes" if include_free_bikes else "--no-include-free-bikes",
    ]
    with log_path.open("ab") as log_file:
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=Path.cwd(),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    console.print(
        "[green]poller started[/green] "
        f"pid={process.pid} interval_seconds={interval_seconds} log={log_path}"
    )


@app.command("poller-stop")
def poller_stop(
    data_dir: Annotated[Path | None, typer.Option(help="Directory containing poller.pid.")] = None,
    timeout_seconds: Annotated[float, typer.Option(help="Seconds to wait after SIGTERM.")] = 10.0,
    force: Annotated[bool, typer.Option(help="Send SIGKILL if SIGTERM does not stop it.")] = False,
) -> None:
    """Stop the background GBFS poller."""
    if timeout_seconds <= 0:
        raise typer.BadParameter("timeout_seconds must be positive")

    settings = make_settings(None, data_dir, None)
    pid_path, _ = poller_paths(settings.data_dir)
    pid = read_pid(pid_path)
    if pid is None:
        console.print("[yellow]poller PID file not found or invalid[/yellow]")
        return
    if not pid_is_running(pid):
        pid_path.unlink(missing_ok=True)
        console.print(f"[yellow]removed stale poller PID[/yellow] pid={pid}")
        return

    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not pid_is_running(pid):
            pid_path.unlink(missing_ok=True)
            console.print(f"[green]poller stopped[/green] pid={pid}")
            return
        time.sleep(0.2)

    if force:
        os.kill(pid, signal.SIGKILL)
        pid_path.unlink(missing_ok=True)
        console.print(f"[green]poller killed[/green] pid={pid}")
        return

    console.print(f"[red]poller did not stop within {timeout_seconds}s[/red] pid={pid}")
    console.print("Run with --force to send SIGKILL.")
    raise typer.Exit(code=1)


@app.command("poller-status")
def poller_status(
    data_dir: Annotated[Path | None, typer.Option(help="Directory containing poller files.")] = None,
    db_path: Annotated[Path | None, typer.Option(help="DuckDB file path.")] = None,
    log_lines: Annotated[int, typer.Option(help="Number of poller log lines to show.")] = 5,
) -> None:
    """Show background poller process, log, and latest collection status."""
    if log_lines < 0:
        raise typer.BadParameter("log_lines cannot be negative")

    settings = make_settings(None, data_dir, db_path)
    pid_path, log_path = poller_paths(settings.data_dir)
    pid = read_pid(pid_path)
    running = pid is not None and pid_is_running(pid)

    latest_row = None
    seconds_since_latest = None
    db_error = None
    if settings.db_path.exists():
        try:
            with duckdb.connect(str(settings.db_path), read_only=True) as con:
                latest_row = con.sql(
                    """
                    select collected_at, station_count, bikes_available, free_bike_count
                    from collection_runs
                    order by collected_at desc
                    limit 1
                    """
                ).fetchone()
        except duckdb.IOException as exc:
            db_error = str(exc).splitlines()[0]
        if latest_row is not None:
            collected_at = latest_row[0]
            if collected_at.tzinfo is None:
                collected_at = collected_at.replace(tzinfo=UTC)
            seconds_since_latest = int((datetime.now(collected_at.tzinfo) - collected_at).total_seconds())

    table = Table(title="Poller status")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("pid", str(pid) if pid is not None else "none")
    table.add_row("running", "yes" if running else "no")
    table.add_row("pid path", str(pid_path))
    table.add_row("log path", str(log_path))
    table.add_row("db path", str(settings.db_path))
    if db_error is not None:
        table.add_row("db status", f"busy: {db_error}")
    if latest_row is not None:
        table.add_row("latest collected", str(latest_row[0]))
        table.add_row("seconds since latest", str(seconds_since_latest))
        table.add_row("latest station count", str(latest_row[1]))
        table.add_row("latest bikes available", str(latest_row[2]))
        table.add_row("latest free bike rows", str(latest_row[3]))
    else:
        table.add_row("latest collected", "none")
    console.print(table)

    lines = tail_lines(log_path, log_lines)
    if lines:
        console.print("[dim]Recent poller log:[/dim]")
        for line in lines:
            console.print(line)


@app.command("db-stats")
def db_stats(
    db_path: Annotated[Path | None, typer.Option(help="DuckDB file path.")] = None,
) -> None:
    """Show high-level stats for the local collection database."""
    settings = make_settings(None, None, db_path)
    with connect_db(settings.db_path) as con:
        row = con.sql(
            """
            select
                count(*) as runs,
                min(collected_at) as first_collected_at,
                max(collected_at) as latest_collected_at,
                sum(station_count) as station_status_rows,
                max(station_count) as max_station_count,
                max(bikes_available) as max_bikes_available
            from collection_runs
            """
        ).fetchone()
        distinct_stations = con.sql(
            "select count(distinct station_id) from station_status_snapshots"
        ).fetchone()[0]
        latest = con.sql(
            """
            select station_count, bikes_available, free_bike_count, raw_path
            from collection_runs
            order by collected_at desc
            limit 1
            """
        ).fetchone()

    table = Table(title=f"Database stats: {settings.db_path}")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("collection runs", str(row[0]))
    table.add_row("first collected", str(row[1]))
    table.add_row("latest collected", str(row[2]))
    table.add_row("station rows", str(row[3]))
    table.add_row("distinct stations", str(distinct_stations))
    table.add_row("max stations/snapshot", str(row[4]))
    table.add_row("max bikes available", str(row[5]))
    if latest is not None:
        table.add_row("latest station count", str(latest[0]))
        table.add_row("latest bikes available", str(latest[1]))
        table.add_row("latest free bike rows", str(latest[2]))
        table.add_row("latest raw path", str(latest[3]))
    console.print(table)


@app.command()
def latest(
    db_path: Annotated[Path | None, typer.Option(help="DuckDB file path.")] = None,
) -> None:
    """Show summary metrics for the newest snapshot."""
    settings = make_settings(None, None, db_path)
    with connect_db(settings.db_path) as con:
        row = con.sql(
            """
            with latest_run as (
                select collected_at
                from collection_runs
                order by collected_at desc
                limit 1
            )
            select
                r.collected_at,
                r.station_count,
                r.bikes_available,
                r.free_bike_count,
                count_if(s.num_bikes_available = 0) as empty_stations,
                count_if(s.num_bikes_available > 0) as stations_with_bikes,
                round(avg(s.num_bikes_available), 2) as avg_bikes_per_station,
                max(s.num_bikes_available) as max_bikes_at_station,
                r.raw_path
            from collection_runs r
            join latest_run lr using (collected_at)
            join station_status_snapshots s using (collected_at)
            group by
                r.collected_at,
                r.station_count,
                r.bikes_available,
                r.free_bike_count,
                r.raw_path
            """
        ).fetchone()

    if row is None:
        console.print("[yellow]No collection runs found.[/yellow]")
        return

    table = Table(title="Latest snapshot")
    table.add_column("Metric")
    table.add_column("Value")
    labels = (
        "collected_at",
        "station_count",
        "bikes_available",
        "free_bike_count",
        "empty_stations",
        "stations_with_bikes",
        "avg_bikes_per_station",
        "max_bikes_at_station",
        "raw_path",
    )
    for label, value in zip(labels, row, strict=True):
        table.add_row(label, str(value))
    console.print(table)


@app.command("top-stations")
def top_stations(
    db_path: Annotated[Path | None, typer.Option(help="DuckDB file path.")] = None,
    limit: Annotated[int, typer.Option(help="Maximum station rows to show.")] = 15,
    by: Annotated[str, typer.Option(help="Ranking mode: latest or avg.")] = "latest",
) -> None:
    """Show stations with the most bikes, either in the latest snapshot or on average."""
    if limit <= 0:
        raise typer.BadParameter("limit must be positive")
    if by not in {"latest", "avg"}:
        raise typer.BadParameter("by must be either 'latest' or 'avg'")

    settings = make_settings(None, None, db_path)
    with connect_db(settings.db_path) as con:
        if by == "latest":
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
                    s.num_bikes_available,
                    s.num_docks_available,
                    to_timestamp(s.last_reported) as last_reported
                from station_status_snapshots s
                join latest_run lr using (collected_at)
                left join station_info i using (station_id)
                order by s.num_bikes_available desc, name
                limit ?
                """,
                [limit],
            ).fetchall()
        else:
            rows = con.execute(
                f"""
                with station_info as ({LATEST_STATION_INFO_SQL})
                select
                    s.station_id,
                    coalesce(i.name, s.station_id) as name,
                    i.region_id,
                    round(avg(s.num_bikes_available), 2) as avg_bikes_available,
                    max(s.num_bikes_available) as max_bikes_available,
                    count(*) as samples
                from station_status_snapshots s
                left join station_info i using (station_id)
                group by s.station_id, i.name, i.region_id
                order by avg_bikes_available desc, name
                limit ?
                """,
                [limit],
            ).fetchall()

    table = Table(title=f"Top stations by {by}")
    table.add_column("Station ID")
    table.add_column("Name")
    table.add_column("Region")
    if by == "latest":
        table.add_column("Bikes", justify="right")
        table.add_column("Docks", justify="right")
        table.add_column("Last reported")
    else:
        table.add_column("Avg bikes", justify="right")
        table.add_column("Max bikes", justify="right")
        table.add_column("Samples", justify="right")
    for row in rows:
        table.add_row(*(str(value) for value in row))
    console.print(table)


@app.command("empty-stations")
def empty_stations(
    db_path: Annotated[Path | None, typer.Option(help="DuckDB file path.")] = None,
    limit: Annotated[int, typer.Option(help="Maximum station rows to show.")] = 30,
    by: Annotated[str, typer.Option(help="Ranking mode: latest or empty-rate.")] = "latest",
) -> None:
    """Show empty stations now, or stations with the highest historical empty rate."""
    if limit <= 0:
        raise typer.BadParameter("limit must be positive")
    if by not in {"latest", "empty-rate"}:
        raise typer.BadParameter("by must be either 'latest' or 'empty-rate'")

    settings = make_settings(None, None, db_path)
    with connect_db(settings.db_path) as con:
        if by == "latest":
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
                    to_timestamp(s.last_reported) as last_reported
                from station_status_snapshots s
                join latest_run lr using (collected_at)
                left join station_info i using (station_id)
                where s.num_bikes_available = 0
                order by name
                limit ?
                """,
                [limit],
            ).fetchall()
        else:
            rows = con.execute(
                f"""
                with station_info as ({LATEST_STATION_INFO_SQL})
                select
                    s.station_id,
                    coalesce(i.name, s.station_id) as name,
                    i.region_id,
                    round(avg(case when s.num_bikes_available = 0 then 1.0 else 0.0 end), 3)
                        as empty_rate,
                    count(*) as samples,
                    round(avg(s.num_bikes_available), 2) as avg_bikes_available
                from station_status_snapshots s
                left join station_info i using (station_id)
                group by s.station_id, i.name, i.region_id
                order by empty_rate desc, samples desc, name
                limit ?
                """,
                [limit],
            ).fetchall()

    table = Table(title=f"Empty stations by {by}")
    table.add_column("Station ID")
    table.add_column("Name")
    table.add_column("Region")
    if by == "latest":
        table.add_column("Last reported")
    else:
        table.add_column("Empty rate", justify="right")
        table.add_column("Samples", justify="right")
        table.add_column("Avg bikes", justify="right")
    for row in rows:
        table.add_row(*(str(value) for value in row))
    console.print(table)


@app.command()
def nearest(
    lat: Annotated[float | None, typer.Option(help="Latitude of the search origin.")] = None,
    lon: Annotated[float | None, typer.Option(help="Longitude of the search origin.")] = None,
    address: Annotated[str | None, typer.Option(help="Address to geocode as the search origin.")] = None,
    db_path: Annotated[Path | None, typer.Option(help="DuckDB file path.")] = None,
    limit: Annotated[int, typer.Option(help="Maximum station rows to show.")] = 10,
    max_distance_m: Annotated[
        float | None,
        typer.Option(help="Only show stations within this distance in meters."),
    ] = None,
    include_empty: Annotated[
        bool,
        typer.Option(help="Include stations with zero available bikes."),
    ] = False,
    whereami: Annotated[
        bool,
        typer.Option(help="Use approximate IP-based geolocation for the search origin."),
    ] = False,
    refresh: Annotated[
        bool,
        typer.Option(help="Collect a fresh station-level snapshot before searching."),
    ] = False,
) -> None:
    """Show the nearest stations from the latest snapshot."""
    location_label = "manual"
    settings = make_settings(None, None, db_path)
    origin_modes = sum(
        [
            lat is not None or lon is not None,
            address is not None,
            whereami,
        ]
    )
    if origin_modes != 1:
        raise typer.BadParameter("Use exactly one origin: both --lat/--lon, --address, or --whereami")

    if whereami:
        lat, lon, location_label = get_ip_location(settings.request_timeout_seconds)
    elif address is not None:
        lat, lon, location_label = get_address_location(address, settings.request_timeout_seconds)

    if lat is None or lon is None:
        raise typer.BadParameter("Provide both --lat and --lon")
    if not -90 <= lat <= 90:
        raise typer.BadParameter("lat must be between -90 and 90")
    if not -180 <= lon <= 180:
        raise typer.BadParameter("lon must be between -180 and 180")
    if limit <= 0:
        raise typer.BadParameter("limit must be positive")
    if max_distance_m is not None and max_distance_m <= 0:
        raise typer.BadParameter("max_distance_m must be positive")

    if refresh:
        metrics = collect_once(settings, language="en", include_free_bikes=False)
        console.print(
            "[green]refreshed[/green] "
            f"stations={metrics['station_count']} "
            f"bikes_available={metrics['bikes_available']}"
        )

    bike_filter = "" if include_empty else "and s.num_bikes_available > 0"
    distance_filter = "" if max_distance_m is None else "where distance_m <= ?"
    params: list[float | int] = [lat, lat, lon]
    if max_distance_m is not None:
        params.append(max_distance_m)
    params.append(limit)

    with connect_db(settings.db_path) as con:
        rows = con.execute(
            f"""
            with latest_run as (
                select collected_at
                from collection_runs
                order by collected_at desc
                limit 1
            ),
            station_info as ({LATEST_STATION_INFO_SQL}),
            candidates as (
                select
                    s.station_id,
                    coalesce(i.name, s.station_id) as name,
                    i.region_id,
                    s.num_bikes_available,
                    i.lat,
                    i.lon,
                    2 * 6371000 * asin(sqrt(
                        pow(sin(radians(i.lat - ?) / 2), 2)
                        + cos(radians(?)) * cos(radians(i.lat))
                        * pow(sin(radians(i.lon - ?) / 2), 2)
                    )) as distance_m
                from station_status_snapshots s
                join latest_run lr using (collected_at)
                left join station_info i using (station_id)
                where i.lat is not null
                    and i.lon is not null
                    {bike_filter}
            )
            select
                station_id,
                name,
                region_id,
                num_bikes_available,
                round(distance_m, 0)::integer as distance_m,
                lat,
                lon
            from candidates
            {distance_filter}
            order by distance_m, name
            limit ?
            """,
            params,
        ).fetchall()

    table = Table(title=f"Nearest stations from {lat:.6f}, {lon:.6f}")
    table.add_column("Station ID")
    table.add_column("Name")
    table.add_column("Region")
    table.add_column("Bikes", justify="right")
    table.add_column("Risk")
    table.add_column("Distance m", justify="right")
    table.add_column("Lat", justify="right")
    table.add_column("Lon", justify="right")
    for row in rows:
        station_id, name, region_id, bikes, distance_m, station_lat, station_lon = row
        table.add_row(
            str(station_id),
            str(name),
            str(region_id),
            str(bikes),
            bike_risk(int(bikes)),
            str(distance_m),
            str(station_lat),
            str(station_lon),
        )
    console.print(f"[dim]Location source: {location_label}[/dim]")
    console.print(table)
