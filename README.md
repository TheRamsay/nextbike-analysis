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
nohup .venv/bin/nextbike poll --interval-seconds 60 --no-include-free-bikes >> data/poller.log 2>&1 &
echo $! > data/poller.pid
```

By default data is written to:

- raw JSON snapshots: `data/raw/`
- normalized local database: `data/nextbike.duckdb`

The feed URL is configurable:

```bash
uv run nextbike info --gbfs-url https://gbfs.nextbike.net/maps/gbfs/v2/nextbike_te/gbfs.json
```
