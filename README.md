# Nextbike Analysis

Local data collection and analysis tooling for the public Nextbike Brno GBFS feed.

## Setup

```bash
uv sync
```

## Commands

Show feed metadata:

```bash
uv run nextbike info
```

Collect one snapshot:

```bash
uv run nextbike collect
```

Poll continuously every minute:

```bash
uv run nextbike poll --interval-seconds 60
```

Prefer station-level collection without per-bike rows for long-running local polling:

```bash
uv run nextbike poll --interval-seconds 60 --no-include-free-bikes
```

Run it in the background:

```bash
uv run nextbike poller-start
uv run nextbike poller-status
uv run nextbike poller-stop
```

Inspect the local database:

```bash
uv run nextbike db-stats
uv run nextbike data-health
uv run nextbike latest
uv run nextbike system-trend --limit 20
uv run nextbike top-stations --limit 10
uv run nextbike top-stations --by avg --limit 10
uv run nextbike empty-stations --limit 20
uv run nextbike empty-stations --by empty-rate --limit 20
uv run nextbike station 27619716
uv run nextbike station Veselá
uv run nextbike nearest --lat 49.1951 --lon 16.6068
uv run nextbike nearest --address "Moravské náměstí, Brno"
uv run nextbike nearest --address "Moravské náměstí, Brno" --refresh
uv run nextbike nearest --whereami
uv run nextbike dashboard
uv run nextbike dashboard --once --width 100 --height 30
uv run nextbike dashboard --background footprint
uv run nextbike dashboard --background none
```

By default data is written to:

- raw JSON snapshots: `data/raw/`
- normalized local database: `data/nextbike.duckdb`

The feed URL is configurable:

```bash
uv run nextbike info --gbfs-url https://gbfs.nextbike.net/maps/gbfs/v2/nextbike_te/gbfs.json
```

The dashboard caches the Brno city boundary from OpenStreetMap/Nominatim in `data/cache/`
and shows the required OpenStreetMap attribution in the terminal.
