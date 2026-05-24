from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

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

