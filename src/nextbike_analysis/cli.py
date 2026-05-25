from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import termios
import time
import tty
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Iterable, Sequence

import duckdb
import typer
from rich.live import Live
from rich.console import Console
from rich.table import Table

from nextbike_analysis.boundary import load_brno_boundary
from nextbike_analysis.config import Settings
from nextbike_analysis.dashboard import load_dashboard_data, render_dashboard
from nextbike_analysis.datasets import build_station_availability_dataset
from nextbike_analysis.db import LATEST_STATION_INFO_SQL, connect_db
from nextbike_analysis.formatting import bike_risk, tail_lines
from nextbike_analysis.geo import get_address_location, get_ip_location
from nextbike_analysis.gbfs import DEFAULT_FEEDS, GbfsClient
from nextbike_analysis.modeling import (
    EvaluationResult,
    evaluate_baselines,
    predict_latest_station_risk,
    train_models,
)
from nextbike_analysis.poller import pid_is_running, poller_paths, read_pid
from nextbike_analysis.recommendations import evaluate_recommendation_strategies
from nextbike_analysis.reports import get_data_health, get_system_trend
from nextbike_analysis.storage import SnapshotStore, utc_now

app = typer.Typer(no_args_is_help=True)
console = Console()


def render_evaluation_result(result: EvaluationResult, title: str) -> None:
    summary = Table(title=title)
    summary.add_column("Metric")
    summary.add_column("Value")
    summary.add_row("table", result.table_name)
    summary.add_row("target", result.target)
    summary.add_row("split collected_at", str(result.split_collected_at))
    summary.add_row("train rows", str(result.train_rows))
    summary.add_row("test rows", str(result.test_rows))
    summary.add_row("train window", result.train_window)
    summary.add_row("test window", result.test_window)
    console.print(summary)

    metrics = Table(title="Metrics")
    metrics.add_column("Model")
    metrics.add_column("Accuracy", justify="right")
    metrics.add_column("Precision", justify="right")
    metrics.add_column("Recall", justify="right")
    metrics.add_column("F1", justify="right")
    metrics.add_column("ROC AUC", justify="right")
    metrics.add_column("Avg precision", justify="right")
    metrics.add_column("Pred empty rate", justify="right")
    for row in sorted(result.metrics, key=lambda item: item.f1, reverse=True):
        metrics.add_row(
            row.name,
            f"{row.accuracy:.4f}",
            f"{row.precision:.4f}",
            f"{row.recall:.4f}",
            f"{row.f1:.4f}",
            f"{row.roc_auc:.4f}" if row.roc_auc is not None else "n/a",
            f"{row.average_precision:.4f}" if row.average_precision is not None else "n/a",
            f"{row.positive_rate:.4f}",
        )
    console.print(metrics)


@contextmanager
def cbreak_stdin() -> object:
    if not sys.stdin.isatty():
        yield
        return

    fd = sys.stdin.fileno()
    previous_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous_settings)


def read_keypress() -> str | None:
    if not sys.stdin.isatty():
        return None
    readable, _, _ = select.select([sys.stdin], [], [], 0)
    if not readable:
        return None
    return sys.stdin.read(1)


def station_recommendation(bikes: int, empty_probability: float) -> str:
    if bikes <= 0:
        return "skip"
    if empty_probability >= 0.7:
        return "risky"
    if empty_probability >= 0.4:
        return "watch"
    return "safe"


def resolve_origin(
    settings: Settings,
    lat: float | None,
    lon: float | None,
    address: str | None,
    whereami: bool,
) -> tuple[float, float, str]:
    origin_modes = sum(
        [
            lat is not None or lon is not None,
            address is not None,
            whereami,
        ]
    )
    if origin_modes != 1:
        raise typer.BadParameter("Use exactly one origin: both --lat/--lon, --address, or --whereami")

    location_label = "manual"
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

    return lat, lon, location_label


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    row_list = [[str(value) for value in row] for row in rows]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in row_list:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


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
    free_bike_rows = None
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
                if latest_row is not None:
                    try:
                        free_bike_rows = con.sql(
                            """
                            select count(*)
                            from free_bike_status_snapshots
                            where collected_at = (
                                select collected_at
                                from collection_runs
                                order by collected_at desc
                                limit 1
                            )
                            """
                        ).fetchone()[0]
                    except duckdb.CatalogException:
                        free_bike_rows = None
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
        if free_bike_rows is not None:
            table.add_row("latest free bike DB rows", str(free_bike_rows))
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


@app.command("build-dataset")
def build_dataset(
    db_path: Annotated[Path | None, typer.Option(help="DuckDB file path.")] = None,
    table_name: Annotated[
        str,
        typer.Option(help="Output table name for the materialized training dataset."),
    ] = "training_station_availability",
    horizon_minutes: Annotated[
        int,
        typer.Option(help="Prediction target horizon in minutes."),
    ] = 30,
    max_target_delay_minutes: Annotated[
        int | None,
        typer.Option(
            help="Maximum accepted target delay. Defaults to horizon + 10 minutes.",
        ),
    ] = None,
    lag_tolerance_minutes: Annotated[
        int,
        typer.Option(help="Extra minutes tolerated when looking up 5m/15m lag features."),
    ] = 10,
) -> None:
    """Build a station-level training table for future bike availability."""
    if horizon_minutes <= 0:
        raise typer.BadParameter("horizon_minutes must be positive")
    if max_target_delay_minutes is None:
        max_target_delay_minutes = horizon_minutes + 10
    if max_target_delay_minutes < horizon_minutes:
        raise typer.BadParameter("max_target_delay_minutes must be >= horizon_minutes")
    if lag_tolerance_minutes < 0:
        raise typer.BadParameter("lag_tolerance_minutes cannot be negative")

    settings = make_settings(None, None, db_path)
    try:
        result = build_station_availability_dataset(
            db_path=settings.db_path,
            table_name=table_name,
            horizon_minutes=horizon_minutes,
            max_target_delay_minutes=max_target_delay_minutes,
            lag_tolerance_minutes=lag_tolerance_minutes,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    table = Table(title=f"Dataset built: {result.table_name}")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("horizon minutes", str(result.horizon_minutes))
    table.add_row("max target delay minutes", str(result.max_target_delay_minutes))
    table.add_row("lag tolerance minutes", str(result.lag_tolerance_minutes))
    table.add_row("rows", str(result.rows))
    table.add_row("stations", str(result.stations))
    table.add_row("first collected", str(result.first_collected_at))
    table.add_row("latest collected", str(result.latest_collected_at))
    table.add_row("null lag 5m rows", str(result.null_lag_5m))
    table.add_row("null lag 15m rows", str(result.null_lag_15m))
    table.add_row("future has-bike rate", str(result.positive_rate))
    table.add_row("future empty rate", str(result.empty_future_rate))
    table.add_row("avg minutes to target", str(result.avg_minutes_to_target))
    console.print(table)


@app.command("evaluate-baseline")
def evaluate_baseline(
    db_path: Annotated[Path | None, typer.Option(help="DuckDB file path.")] = None,
    table_name: Annotated[
        str,
        typer.Option(help="Training dataset table name."),
    ] = "training_station_availability",
    test_fraction: Annotated[
        float,
        typer.Option(help="Newest fraction of timestamps used as test set."),
    ] = 0.25,
    low_bike_threshold: Annotated[
        int,
        typer.Option(help="Threshold for low-inventory baseline."),
    ] = 1,
) -> None:
    """Evaluate simple baselines for predicting whether a station will be empty."""
    if not 0 < test_fraction < 1:
        raise typer.BadParameter("test_fraction must be between 0 and 1")
    if low_bike_threshold < 0:
        raise typer.BadParameter("low_bike_threshold cannot be negative")

    settings = make_settings(None, None, db_path)
    try:
        result = evaluate_baselines(
            db_path=settings.db_path,
            table_name=table_name,
            test_fraction=test_fraction,
            low_bike_threshold=low_bike_threshold,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    render_evaluation_result(result, "Baseline evaluation")


@app.command("train-model")
def train_model(
    db_path: Annotated[Path | None, typer.Option(help="DuckDB file path.")] = None,
    data_dir: Annotated[Path | None, typer.Option(help="Directory for model artifacts.")] = None,
    table_name: Annotated[
        str,
        typer.Option(help="Training dataset table name."),
    ] = "training_station_availability",
    test_fraction: Annotated[
        float,
        typer.Option(help="Newest fraction of timestamps used as test set."),
    ] = 0.25,
    save_models: Annotated[
        bool,
        typer.Option(help="Persist trained sklearn models under data/models/."),
    ] = True,
) -> None:
    """Train first sklearn tabular models for future station emptiness."""
    if not 0 < test_fraction < 1:
        raise typer.BadParameter("test_fraction must be between 0 and 1")

    settings = make_settings(None, data_dir, db_path)
    model_dir = settings.data_dir / "models" if save_models else None
    try:
        result, saved_models = train_models(
            db_path=settings.db_path,
            table_name=table_name,
            test_fraction=test_fraction,
            model_dir=model_dir,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    render_evaluation_result(result, "Model evaluation")
    if saved_models:
        table = Table(title="Saved models")
        table.add_column("Model")
        table.add_column("Path")
        for name, path in saved_models.items():
            table.add_row(name, str(path))
        console.print(table)


@app.command("evaluate-recommendations")
def evaluate_recommendations(
    db_path: Annotated[Path | None, typer.Option(help="DuckDB file path.")] = None,
    data_dir: Annotated[Path | None, typer.Option(help="Directory containing model artifacts.")] = None,
    table_name: Annotated[
        str,
        typer.Option(help="Training dataset table name."),
    ] = "training_station_availability",
    model_path: Annotated[
        Path | None,
        typer.Option(help="Path to trained model. Defaults to data/models/hist_gradient_boosting.joblib."),
    ] = None,
    test_fraction: Annotated[
        float,
        typer.Option(help="Newest fraction of timestamps used as test set."),
    ] = 0.25,
    max_distance_m: Annotated[
        float,
        typer.Option(help="Maximum walking distance considered for a recommendation."),
    ] = 1000.0,
) -> None:
    """Evaluate historical nearest-station recommendation strategies."""
    if not 0 < test_fraction < 1:
        raise typer.BadParameter("test_fraction must be between 0 and 1")
    if max_distance_m <= 0:
        raise typer.BadParameter("max_distance_m must be positive")

    settings = make_settings(None, data_dir, db_path)
    model_path = model_path or (settings.data_dir / "models" / "hist_gradient_boosting.joblib")
    try:
        result = evaluate_recommendation_strategies(
            db_path=settings.db_path,
            model_path=model_path,
            table_name=table_name,
            test_fraction=test_fraction,
            max_distance_m=max_distance_m,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    summary = Table(title="Recommendation evaluation")
    summary.add_column("Metric")
    summary.add_column("Value")
    summary.add_row("origins", ", ".join(origin.name for origin in result.origins))
    summary.add_row("timestamps", str(result.timestamps))
    summary.add_row("attempts per strategy", str(result.attempts_per_strategy))
    summary.add_row("max distance m", str(max_distance_m))
    summary.add_row("model", str(model_path))
    console.print(summary)

    table = Table(title="Strategies")
    table.add_column("Strategy")
    table.add_column("Attempts", justify="right")
    table.add_column("No candidate", justify="right")
    table.add_column("Success rate", justify="right")
    table.add_column("Avg dist m", justify="right")
    table.add_column("P90 dist m", justify="right")
    table.add_column("Avg empty prob", justify="right")
    table.add_column("Avg extra m", justify="right")
    for row in sorted(result.metrics, key=lambda item: item.success_rate or 0, reverse=True):
        table.add_row(
            row.label,
            str(row.attempts),
            str(row.no_candidate),
            f"{row.success_rate:.4f}" if row.success_rate is not None else "n/a",
            f"{row.avg_distance_m:.0f}" if row.avg_distance_m is not None else "n/a",
            f"{row.p90_distance_m:.0f}" if row.p90_distance_m is not None else "n/a",
            f"{row.avg_empty_probability:.4f}" if row.avg_empty_probability is not None else "n/a",
            f"{row.avg_extra_distance_m:.0f}" if row.avg_extra_distance_m is not None else "n/a",
        )
    console.print(table)


@app.command("data-health")
def data_health(
    db_path: Annotated[Path | None, typer.Option(help="DuckDB file path.")] = None,
    data_dir: Annotated[Path | None, typer.Option(help="Directory for raw data.")] = None,
    expected_interval_seconds: Annotated[
        int,
        typer.Option(help="Expected polling interval in seconds."),
    ] = 60,
    gap_threshold_seconds: Annotated[
        int,
        typer.Option(help="Minimum interval treated as a collection gap."),
    ] = 120,
    max_gaps: Annotated[int, typer.Option(help="Maximum gaps to show.")] = 10,
    since_hours: Annotated[
        float | None,
        typer.Option(help="Only evaluate snapshots collected within this many recent hours."),
    ] = None,
) -> None:
    """Show collection continuity and local storage health."""
    if expected_interval_seconds <= 0:
        raise typer.BadParameter("expected_interval_seconds must be positive")
    if gap_threshold_seconds <= 0:
        raise typer.BadParameter("gap_threshold_seconds must be positive")
    if max_gaps < 0:
        raise typer.BadParameter("max_gaps cannot be negative")
    if since_hours is not None and since_hours <= 0:
        raise typer.BadParameter("since_hours must be positive")

    settings = make_settings(None, data_dir, db_path)
    health = get_data_health(
        db_path=settings.db_path,
        data_dir=settings.data_dir,
        expected_interval_seconds=expected_interval_seconds,
        gap_threshold_seconds=gap_threshold_seconds,
        max_gaps=max_gaps,
        since_hours=since_hours,
    )

    table = Table(title="Data health")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("collection runs", str(health.runs))
    table.add_row("window", health.window)
    table.add_row("expected samples", str(health.expected_samples))
    table.add_row("coverage pct", str(health.coverage_pct))
    table.add_row("first collected", str(health.first_collected_at))
    table.add_row("latest collected", str(health.latest_collected_at))
    table.add_row("station rows", str(health.station_rows))
    table.add_row(
        "free bike rows",
        str(health.free_bike_rows) if health.free_bike_rows is not None else "not initialized",
    )
    table.add_row("distinct stations", str(health.distinct_stations))
    table.add_row("station count min/avg/max", health.station_count_min_avg_max)
    table.add_row("bikes available min/avg/max", health.bikes_available_min_avg_max)
    table.add_row("duplicate collected_at", str(health.duplicate_collected_at))
    table.add_row("gap count", str(health.gap_count))
    table.add_row("interval min/avg/max seconds", health.interval_min_avg_max_seconds)
    table.add_row("db size MB", f"{health.db_size_mb:.2f}")
    table.add_row("raw size MB", f"{health.raw_size_mb:.2f}")
    console.print(table)

    if health.gaps:
        gap_table = Table(title=f"Largest gaps > {gap_threshold_seconds}s")
        gap_table.add_column("Previous")
        gap_table.add_column("Next")
        gap_table.add_column("Gap seconds", justify="right")
        for row in health.gaps:
            gap_table.add_row(str(row[0]), str(row[1]), str(row[2]))
        console.print(gap_table)


@app.command("system-trend")
def system_trend(
    db_path: Annotated[Path | None, typer.Option(help="DuckDB file path.")] = None,
    limit: Annotated[int, typer.Option(help="Number of recent snapshots to show.")] = 20,
) -> None:
    """Show recent system-wide availability trend."""
    if limit <= 0:
        raise typer.BadParameter("limit must be positive")

    settings = make_settings(None, None, db_path)
    rows = get_system_trend(settings.db_path, limit)

    table = Table(title=f"System trend ({len(rows)} snapshots)")
    table.add_column("Collected at")
    table.add_column("Bikes", justify="right")
    table.add_column("Free bike rows", justify="right")
    table.add_column("Empty stations", justify="right")
    table.add_column("Avg bikes/station", justify="right")
    table.add_column("Max bikes/station", justify="right")
    for row in reversed(rows):
        table.add_row(*(str(value) for value in row))
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
def station(
    station_ref: Annotated[str, typer.Argument(help="Station ID or case-insensitive name fragment.")],
    db_path: Annotated[Path | None, typer.Option(help="DuckDB file path.")] = None,
    samples: Annotated[int, typer.Option(help="Number of recent snapshots to show.")] = 20,
) -> None:
    """Show station metadata, latest status, and recent availability trend."""
    if samples <= 0:
        raise typer.BadParameter("samples must be positive")

    settings = make_settings(None, None, db_path)
    with connect_db(settings.db_path) as con:
        matches = con.execute(
            f"""
            with station_info as ({LATEST_STATION_INFO_SQL})
            select station_id, name, short_name, region_id, lat, lon
            from station_info
            where station_id = ?
                or contains(lower(name), lower(?))
                or contains(lower(coalesce(short_name, '')), lower(?))
            order by
                case when station_id = ? then 0 else 1 end,
                name
            limit 25
            """,
            [station_ref, station_ref, station_ref, station_ref],
        ).fetchall()

        if not matches:
            console.print(f"[yellow]No station found for {station_ref!r}.[/yellow]")
            raise typer.Exit(code=1)

        if len(matches) > 1 and not any(row[0] == station_ref for row in matches):
            table = Table(title=f"Multiple stations match {station_ref!r}")
            table.add_column("Station ID")
            table.add_column("Name")
            table.add_column("Short name")
            table.add_column("Region")
            table.add_column("Lat", justify="right")
            table.add_column("Lon", justify="right")
            for row in matches:
                table.add_row(*(str(value) for value in row))
            console.print(table)
            raise typer.Exit(code=1)

        selected = next((row for row in matches if row[0] == station_ref), matches[0])
        station_id, name, short_name, region_id, lat, lon = selected

        latest_row = con.execute(
            """
            select
                collected_at,
                num_bikes_available,
                num_docks_available,
                is_renting,
                is_returning,
                to_timestamp(last_reported) as last_reported
            from station_status_snapshots
            where station_id = ?
            order by collected_at desc
            limit 1
            """,
            [station_id],
        ).fetchone()
        aggregate_row = con.execute(
            """
            select
                count(*) as samples,
                round(avg(num_bikes_available), 2) as avg_bikes,
                min(num_bikes_available) as min_bikes,
                max(num_bikes_available) as max_bikes,
                round(avg(case when num_bikes_available = 0 then 1.0 else 0.0 end), 3)
                    as empty_rate
            from station_status_snapshots
            where station_id = ?
            """,
            [station_id],
        ).fetchone()
        trend_rows = con.execute(
            """
            select
                collected_at,
                num_bikes_available,
                num_docks_available,
                is_renting,
                is_returning
            from station_status_snapshots
            where station_id = ?
            order by collected_at desc
            limit ?
            """,
            [station_id, samples],
        ).fetchall()

    summary = Table(title=f"Station {station_id}")
    summary.add_column("Metric")
    summary.add_column("Value")
    summary.add_row("name", str(name))
    summary.add_row("short_name", str(short_name))
    summary.add_row("region", str(region_id))
    summary.add_row("lat/lon", f"{lat}, {lon}")
    if latest_row is not None:
        summary.add_row("latest collected", str(latest_row[0]))
        summary.add_row("latest bikes", str(latest_row[1]))
        summary.add_row("latest risk", bike_risk(int(latest_row[1])))
        summary.add_row("latest docks", str(latest_row[2]))
        summary.add_row("renting/returning", f"{latest_row[3]}/{latest_row[4]}")
        summary.add_row("last reported", str(latest_row[5]))
    if aggregate_row is not None:
        summary.add_row("samples", str(aggregate_row[0]))
        summary.add_row("avg bikes", str(aggregate_row[1]))
        summary.add_row("min/max bikes", f"{aggregate_row[2]}/{aggregate_row[3]}")
        summary.add_row("empty rate", str(aggregate_row[4]))
    console.print(summary)

    trend = Table(title=f"Recent trend ({len(trend_rows)} samples)")
    trend.add_column("Collected at")
    trend.add_column("Bikes", justify="right")
    trend.add_column("Risk")
    trend.add_column("Docks", justify="right")
    trend.add_column("Renting")
    trend.add_column("Returning")
    for row in reversed(trend_rows):
        collected_at, bikes, docks, is_renting, is_returning = row
        trend.add_row(
            str(collected_at),
            str(bikes),
            bike_risk(int(bikes)),
            str(docks),
            str(is_renting),
            str(is_returning),
        )
    console.print(trend)


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
    predict_risk: Annotated[
        bool,
        typer.Option(help="Score nearest stations with the trained 30-minute emptiness model."),
    ] = False,
    reliable: Annotated[
        bool,
        typer.Option(help="Prefer stations with at least --reliable-min-bikes bikes."),
    ] = False,
    reliable_min_bikes: Annotated[
        int,
        typer.Option(help="Minimum bikes for reliable nearest recommendations."),
    ] = 2,
    model_path: Annotated[
        Path | None,
        typer.Option(help="Path to a trained sklearn model. Defaults to data/models/hist_gradient_boosting.joblib."),
    ] = None,
) -> None:
    """Show the nearest stations from the latest snapshot."""
    settings = make_settings(None, None, db_path)
    lat, lon, location_label = resolve_origin(settings, lat, lon, address, whereami)
    if limit <= 0:
        raise typer.BadParameter("limit must be positive")
    if max_distance_m is not None and max_distance_m <= 0:
        raise typer.BadParameter("max_distance_m must be positive")
    if reliable_min_bikes <= 0:
        raise typer.BadParameter("reliable_min_bikes must be positive")

    if refresh:
        metrics = collect_once(settings, language="en", include_free_bikes=False)
        console.print(
            "[green]refreshed[/green] "
            f"stations={metrics['station_count']} "
            f"bikes_available={metrics['bikes_available']}"
        )

    bike_filter = "" if include_empty else "and s.num_bikes_available > 0"
    distance_filter = "" if max_distance_m is None else "where distance_m <= ?"
    candidate_limit = max(limit, 50) if reliable else limit
    params: list[float | int] = [lat, lat, lon]
    if max_distance_m is not None:
        params.append(max_distance_m)
    params.append(candidate_limit)

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

    nearest_distance = int(rows[0][4]) if rows else None
    reliable_fallback = False
    if reliable:
        reliable_rows = [row for row in rows if int(row[3]) >= reliable_min_bikes]
        if reliable_rows:
            rows = reliable_rows[:limit]
        else:
            rows = rows[:limit]
            reliable_fallback = True

    predictions = {}
    if predict_risk:
        model_path = model_path or (settings.data_dir / "models" / "hist_gradient_boosting.joblib")
        try:
            predictions = predict_latest_station_risk(
                db_path=settings.db_path,
                model_path=model_path,
                station_ids=[str(row[0]) for row in rows],
            )
        except (FileNotFoundError, ValueError) as exc:
            raise typer.BadParameter(str(exc)) from exc

    table = Table(title=f"Nearest stations from {lat:.6f}, {lon:.6f}")
    if predict_risk:
        table.add_column("Name", ratio=3)
        table.add_column("Bikes", justify="right", no_wrap=True)
        table.add_column("Now", no_wrap=True)
        table.add_column("Empty 30m", no_wrap=True)
        table.add_column("Rec", no_wrap=True)
        if reliable:
            table.add_column("Extra m", justify="right", no_wrap=True)
        table.add_column("Distance m", justify="right", no_wrap=True)
    elif reliable:
        table.add_column("Name", ratio=3)
        table.add_column("Bikes", justify="right", no_wrap=True)
        table.add_column("Risk", no_wrap=True)
        table.add_column("Extra m", justify="right", no_wrap=True)
        table.add_column("Distance m", justify="right", no_wrap=True)
    else:
        table.add_column("Name", ratio=3)
        table.add_column("Station ID", no_wrap=True)
        table.add_column("Bikes", justify="right", no_wrap=True)
        table.add_column("Risk", no_wrap=True)
        table.add_column("Distance m", justify="right", no_wrap=True)
        table.add_column("Lat", justify="right", no_wrap=True)
        table.add_column("Lon", justify="right", no_wrap=True)
    for row in rows:
        station_id, name, region_id, bikes, distance_m, station_lat, station_lon = row
        current_risk = bike_risk(int(bikes))
        extra_distance = int(distance_m) - nearest_distance if nearest_distance is not None else 0
        if predict_risk:
            table_row = [
                str(name),
                str(bikes),
                current_risk,
            ]
            prediction = predictions.get(str(station_id))
            if prediction is None:
                table_row.extend(["n/a", "?"])
            else:
                table_row.extend(
                    [
                        f"{prediction.empty_probability * 100:.1f}% {prediction.risk_label}",
                        station_recommendation(int(bikes), prediction.empty_probability),
                    ]
                )
            if reliable:
                table_row.append(f"+{extra_distance}")
            table_row.append(str(distance_m))
        elif reliable:
            table_row = [
                str(name),
                str(bikes),
                current_risk,
                f"+{extra_distance}",
                str(distance_m),
            ]
        else:
            table_row = [
                str(name),
                str(station_id),
                str(bikes),
                current_risk,
            ]
            table_row.append(str(distance_m))
            table_row.extend([str(station_lat), str(station_lon)])
        table.add_row(
            *table_row,
        )
    console.print(f"[dim]Location source: {location_label}[/dim]")
    if reliable:
        if reliable_fallback:
            console.print(
                f"[yellow]No station with bikes >= {reliable_min_bikes} found in candidates; "
                "showing nearest available stations.[/yellow]"
            )
        else:
            console.print(
                f"[dim]Reliable mode: bikes >= {reliable_min_bikes}; "
                "Extra m is distance beyond nearest station with a bike.[/dim]"
            )
    if predict_risk:
        console.print(f"[dim]Risk model: {model_path}[/dim]")
    console.print(table)


@app.command("area-trend")
def area_trend(
    lat: Annotated[float | None, typer.Option(help="Latitude of the area origin.")] = None,
    lon: Annotated[float | None, typer.Option(help="Longitude of the area origin.")] = None,
    address: Annotated[str | None, typer.Option(help="Address to geocode as the area origin.")] = None,
    whereami: Annotated[
        bool,
        typer.Option(help="Use approximate IP-based geolocation for the area origin."),
    ] = False,
    db_path: Annotated[Path | None, typer.Option(help="DuckDB file path.")] = None,
    radius_m: Annotated[int, typer.Option(help="Walkable area radius in meters.")] = 600,
    limit: Annotated[int, typer.Option(help="Maximum nearby station rows to show.")] = 12,
    timezone: Annotated[str, typer.Option(help="Timezone for hourly grouping.")] = "Europe/Prague",
    report_path: Annotated[Path | None, typer.Option(help="Optional Markdown report output path.")] = None,
) -> None:
    """Analyze when bikes disappear around an address or point."""
    settings = make_settings(None, None, db_path)
    lat, lon, location_label = resolve_origin(settings, lat, lon, address, whereami)
    if radius_m <= 0:
        raise typer.BadParameter("radius_m must be positive")
    if limit <= 0:
        raise typer.BadParameter("limit must be positive")

    distance_sql = """
        cast(round(
            6371000 * 2 * asin(sqrt(
                power(sin(radians((i.lat - ?) / 2)), 2)
                + cos(radians(?)) * cos(radians(i.lat))
                * power(sin(radians((i.lon - ?) / 2)), 2)
            ))
        ) as integer)
    """
    station_cte = f"""
        with station_info as ({LATEST_STATION_INFO_SQL}),
        stations as (
            select
                i.station_id,
                coalesce(i.name, i.station_id) as name,
                i.region_id,
                i.lat,
                i.lon,
                {distance_sql} as distance_m
            from station_info i
        )
    """

    with connect_db(settings.db_path) as con:
        coverage_row = con.execute(
            f"""
            {station_cte},
            area_rows as (
                select s.collected_at, s.station_id, s.num_bikes_available
                from station_status_snapshots s
                join stations st using (station_id)
                where st.distance_m <= ?
            )
            select
                min(collected_at) as first_collected,
                max(collected_at) as latest_collected,
                count(distinct collected_at) as snapshots,
                count(*) as station_rows,
                count(distinct station_id) as stations
            from area_rows
            """,
            [lat, lat, lon, radius_m],
        ).fetchone()

        latest_area_row = con.execute(
            f"""
            {station_cte},
            latest_snapshot as (
                select max(collected_at) as collected_at
                from station_status_snapshots
            )
            select
                sum(s.num_bikes_available) as total_bikes,
                count_if(s.num_bikes_available > 0) as stations_with_bikes,
                count_if(s.num_bikes_available = 0) as empty_stations
            from station_status_snapshots s
            join latest_snapshot l using (collected_at)
            join stations st using (station_id)
            where st.distance_m <= ?
            """,
            [lat, lat, lon, radius_m],
        ).fetchone()

        station_rows = con.execute(
            f"""
            {station_cte},
            latest_snapshot as (
                select max(collected_at) as collected_at
                from station_status_snapshots
            ),
            station_history as (
                select
                    s.station_id,
                    count(*) as samples,
                    round(avg(s.num_bikes_available), 2) as avg_bikes,
                    round(avg(case when s.num_bikes_available = 0 then 1.0 else 0.0 end), 3)
                        as empty_rate,
                    round(avg(case when s.num_bikes_available <= 1 then 1.0 else 0.0 end), 3)
                        as low_rate
                from station_status_snapshots s
                join stations st using (station_id)
                where st.distance_m <= ?
                group by s.station_id
            )
            select
                st.station_id,
                st.name,
                coalesce(ls.num_bikes_available, 0) as latest_bikes,
                st.distance_m,
                coalesce(h.avg_bikes, 0) as avg_bikes,
                coalesce(h.empty_rate, 0) as empty_rate,
                coalesce(h.low_rate, 0) as low_rate,
                coalesce(h.samples, 0) as samples
            from stations st
            left join latest_snapshot l on true
            left join station_status_snapshots ls
                on ls.station_id = st.station_id
                and ls.collected_at = l.collected_at
            left join station_history h
                on h.station_id = st.station_id
            where st.distance_m <= ?
            order by st.distance_m, st.name
            limit ?
            """,
            [lat, lat, lon, radius_m, radius_m, limit],
        ).fetchall()

        hourly_rows = con.execute(
            f"""
            {station_cte},
            area_state as (
                select
                    s.collected_at,
                    strftime(timezone(?, s.collected_at), '%H:00') as local_hour,
                    sum(s.num_bikes_available) as total_bikes,
                    count_if(s.num_bikes_available > 0) as stations_with_bikes,
                    count_if(s.num_bikes_available = 0) as empty_stations
                from station_status_snapshots s
                join stations st using (station_id)
                where st.distance_m <= ?
                group by s.collected_at, local_hour
            )
            select
                local_hour,
                count(*) as samples,
                round(avg(total_bikes), 2) as avg_total_bikes,
                round(avg(stations_with_bikes), 2) as avg_stations_with_bikes,
                round(avg(empty_stations), 2) as avg_empty_stations,
                round(avg(case when stations_with_bikes = 0 then 1.0 else 0.0 end), 3)
                    as all_empty_rate
            from area_state
            group by local_hour
            order by local_hour
            """,
            [lat, lat, lon, timezone, radius_m],
        ).fetchall()

        depletion_rows = con.execute(
            f"""
            {station_cte},
            transitions as (
                select
                    s.station_id,
                    s.collected_at,
                    s.num_bikes_available,
                    lag(s.num_bikes_available) over (
                        partition by s.station_id
                        order by s.collected_at
                    ) as previous_bikes
                from station_status_snapshots s
                join stations st using (station_id)
                where st.distance_m <= ?
            )
            select
                strftime(timezone(?, collected_at), '%H:00') as local_hour,
                count(*) as depletion_events
            from transitions
            where previous_bikes > 0 and num_bikes_available = 0
            group by local_hour
            order by depletion_events desc, local_hour
            """,
            [lat, lat, lon, radius_m, timezone],
        ).fetchall()

    summary = Table(title=f"Area trend around {lat:.6f}, {lon:.6f}")
    summary.add_column("Metric")
    summary.add_column("Value")
    summary.add_row("location source", location_label)
    summary.add_row("radius", f"{radius_m} m")
    summary.add_row("first collected", str(coverage_row[0]))
    summary.add_row("latest collected", str(coverage_row[1]))
    summary.add_row("area snapshots", str(coverage_row[2] or 0))
    summary.add_row("station rows", str(coverage_row[3] or 0))
    summary.add_row("stations in radius", str(coverage_row[4] or 0))
    summary.add_row("latest bikes in radius", str(latest_area_row[0] or 0))
    summary.add_row("latest stations with bikes", str(latest_area_row[1] or 0))
    summary.add_row("latest empty stations", str(latest_area_row[2] or 0))
    console.print(summary)

    if not station_rows:
        console.print("[yellow]No stations found inside this radius.[/yellow]")
        raise typer.Exit(code=1)

    station_table = Table(title="Nearby stations")
    station_table.add_column("Name", ratio=3)
    station_table.add_column("Station ID", no_wrap=True)
    station_table.add_column("Now", justify="right", no_wrap=True)
    station_table.add_column("Distance m", justify="right", no_wrap=True)
    station_table.add_column("Avg bikes", justify="right", no_wrap=True)
    station_table.add_column("Empty", justify="right", no_wrap=True)
    station_table.add_column("<=1 bike", justify="right", no_wrap=True)
    for station_id, name, latest_bikes, distance_m, avg_bikes, empty_rate, low_rate, _samples in station_rows:
        station_table.add_row(
            str(name),
            str(station_id),
            str(latest_bikes),
            str(distance_m),
            str(avg_bikes),
            f"{float(empty_rate) * 100:.1f}%",
            f"{float(low_rate) * 100:.1f}%",
        )
    console.print(station_table)

    hourly_table = Table(title="Hourly area availability")
    hourly_table.add_column("Hour")
    hourly_table.add_column("Samples", justify="right")
    hourly_table.add_column("Avg bikes", justify="right")
    hourly_table.add_column("Avg stations with bikes", justify="right")
    hourly_table.add_column("Avg empty stations", justify="right")
    hourly_table.add_column("All empty", justify="right")
    for local_hour, samples, avg_total, avg_with_bikes, avg_empty, all_empty_rate in hourly_rows:
        hourly_table.add_row(
            str(local_hour),
            str(samples),
            str(avg_total),
            str(avg_with_bikes),
            str(avg_empty),
            f"{float(all_empty_rate) * 100:.1f}%",
        )
    console.print(hourly_table)

    depletion_table = Table(title="Observed station depletion events")
    depletion_table.add_column("Hour")
    depletion_table.add_column("Events", justify="right")
    for local_hour, events in depletion_rows:
        depletion_table.add_row(str(local_hour), str(events))
    console.print(depletion_table)

    morning_rows = [row for row in hourly_rows if "05:00" <= str(row[0]) <= "12:00"]
    first_low_row = next((row for row in morning_rows if float(row[2]) <= 1.0), None)
    first_all_empty_row = next((row for row in morning_rows if float(row[5]) > 0.0), None)
    full_empty_row = next((row for row in morning_rows if float(row[5]) >= 1.0), None)
    interpretation_rows = [
        (
            "first morning hour averaging <= 1 bike",
            first_low_row[0] if first_low_row is not None else "not observed",
        ),
        (
            "first morning hour with all-empty samples",
            first_all_empty_row[0] if first_all_empty_row is not None else "not observed",
        ),
        (
            "first morning hour fully empty in current data",
            full_empty_row[0] if full_empty_row is not None else "not observed",
        ),
    ]
    interpretation_table = Table(title="Preliminary read")
    interpretation_table.add_column("Signal")
    interpretation_table.add_column("Hour", justify="right")
    for signal, hour in interpretation_rows:
        interpretation_table.add_row(signal, str(hour))
    console.print(interpretation_table)

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = [
            "# Local Nextbike Availability Trend",
            "",
            f"- Location: `{lat:.6f}, {lon:.6f}`",
            f"- Source: {location_label}",
            f"- Radius: `{radius_m} m`",
            f"- Data window: `{coverage_row[0]}` to `{coverage_row[1]}`",
            f"- Snapshots: `{coverage_row[2] or 0}`",
            f"- Stations in radius: `{coverage_row[4] or 0}`",
            f"- Latest bikes in radius: `{latest_area_row[0] or 0}`",
            "",
            "## Preliminary Read",
            "",
            markdown_table(["Signal", "Hour"], interpretation_rows),
            "",
            "## Nearby Stations",
            "",
            markdown_table(
                ["Name", "Station ID", "Now", "Distance m", "Avg bikes", "Empty", "<=1 bike"],
                [
                    [
                        name,
                        station_id,
                        latest_bikes,
                        distance_m,
                        avg_bikes,
                        f"{float(empty_rate) * 100:.1f}%",
                        f"{float(low_rate) * 100:.1f}%",
                    ]
                    for (
                        station_id,
                        name,
                        latest_bikes,
                        distance_m,
                        avg_bikes,
                        empty_rate,
                        low_rate,
                        _samples,
                    ) in station_rows
                ],
            ),
            "",
            "## Hourly Area Availability",
            "",
            markdown_table(
                [
                    "Hour",
                    "Samples",
                    "Avg bikes",
                    "Avg stations with bikes",
                    "Avg empty stations",
                    "All empty",
                ],
                [
                    [
                        local_hour,
                        samples,
                        avg_total,
                        avg_with_bikes,
                        avg_empty,
                        f"{float(all_empty_rate) * 100:.1f}%",
                    ]
                    for (
                        local_hour,
                        samples,
                        avg_total,
                        avg_with_bikes,
                        avg_empty,
                        all_empty_rate,
                    ) in hourly_rows
                ],
            ),
            "",
            "## Observed Station Depletion Events",
            "",
            markdown_table(["Hour", "Events"], depletion_rows),
            "",
        ]
        report_path.write_text("\n".join(report), encoding="utf-8")
        console.print(f"[green]wrote report[/green] {report_path}")


@app.command()
def dashboard(
    db_path: Annotated[Path | None, typer.Option(help="DuckDB file path.")] = None,
    data_dir: Annotated[Path | None, typer.Option(help="Directory for cached map data.")] = None,
    width: Annotated[int | None, typer.Option(help="Map width in terminal columns.")] = None,
    height: Annotated[int | None, typer.Option(help="Map height in terminal rows.")] = None,
    refresh_seconds: Annotated[float, typer.Option(help="Refresh interval for live mode.")] = 10.0,
    include_empty: Annotated[bool, typer.Option(help="Show empty stations too.")] = True,
    background: Annotated[
        str,
        typer.Option(help="Map background: osm, footprint, or none."),
    ] = "osm",
    once: Annotated[bool, typer.Option(help="Render once and exit.")] = False,
) -> None:
    """Show a live ASCII map dashboard for the latest Brno station snapshot."""
    if width is not None and width < 20:
        raise typer.BadParameter("width must be at least 20")
    if height is not None and height < 8:
        raise typer.BadParameter("height must be at least 8")
    if refresh_seconds <= 0:
        raise typer.BadParameter("refresh_seconds must be positive")
    if background not in {"osm", "footprint", "none"}:
        raise typer.BadParameter("background must be one of: osm, footprint, none")

    settings = make_settings(None, data_dir, db_path)
    boundary = None
    effective_background = background
    if background == "osm":
        try:
            boundary = load_brno_boundary(settings.data_dir, settings.request_timeout_seconds)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]OSM boundary unavailable, using station footprint:[/yellow] {exc}")
            effective_background = "footprint"

    def render_current() -> object:
        console_width, console_height = console.size
        map_width = width or max(20, min(120, console_width - 2))
        map_height = height or max(8, min(36, console_height - 5))
        data = load_dashboard_data(settings.db_path, include_empty)
        return render_dashboard(data, map_width, map_height, effective_background, boundary)

    if once:
        console.print(render_current())
        return

    try:
        with cbreak_stdin():
            with Live(render_current(), console=console, screen=True, refresh_per_second=4) as live:
                next_refresh = time.monotonic() + refresh_seconds
                while True:
                    key = read_keypress()
                    if key in {"q", "Q"}:
                        break
                    if time.monotonic() >= next_refresh:
                        live.update(render_current())
                        next_refresh = time.monotonic() + refresh_seconds
                    time.sleep(0.1)
    except KeyboardInterrupt:
        console.print("[dim]dashboard stopped[/dim]")
