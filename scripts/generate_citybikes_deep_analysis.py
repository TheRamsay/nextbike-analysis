from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

from nextbike_analysis.citybikes import ensure_network_files, network_parquet_glob
from nextbike_analysis.config import Settings
from nextbike_analysis.geo import get_address_location


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deep CityBikes analysis artifacts.")
    parser.add_argument("--network", default="nextbike-brno")
    parser.add_argument("--data-dir", type=Path, default=Path("data/citybikes"))
    parser.add_argument("--address")
    parser.add_argument("--lat", type=float)
    parser.add_argument("--lon", type=float)
    parser.add_argument("--radius-m", type=int, default=600)
    parser.add_argument("--city-sample-minutes", type=int, default=60)
    parser.add_argument("--local-sample-minutes", type=int, default=15)
    parser.add_argument("--max-move-window-minutes", type=int, default=720)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/local_citybikes"))
    return parser.parse_args()


def resolve_location(args: argparse.Namespace) -> tuple[float, float, str]:
    if args.address:
        settings = Settings()
        return get_address_location(args.address, settings.request_timeout_seconds)
    if args.lat is None or args.lon is None:
        raise SystemExit("Provide --address or both --lat/--lon")
    return args.lat, args.lon, "manual coordinates"


def quote_path(path: str) -> str:
    return path.replace("'", "''")


def load_raw_summary(con: duckdb.DuckDBPyConnection, parquet_glob: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = con.execute(
        f"""
        with r as (
            select * from read_parquet('{quote_path(parquet_glob)}', filename = true)
        )
        select
            count(*) as row_count,
            min(timestamp) as first_timestamp,
            max(timestamp) as latest_timestamp,
            count(distinct nuid) as station_ids,
            count(distinct name) as station_names,
            min(regexp_extract(filename, '([0-9]{{6}})-', 1)) as first_month,
            max(regexp_extract(filename, '([0-9]{{6}})-', 1)) as latest_month
        from r
        """
    ).df()
    monthly = con.execute(
        f"""
        with r as (
            select *, regexp_extract(filename, '([0-9]{{6}})-', 1) as month_key
            from read_parquet('{quote_path(parquet_glob)}', filename = true)
        )
        select
            month_key,
            count(*) as change_rows,
            count(distinct nuid) as station_ids,
            round(avg(bikes), 2) as avg_bikes_on_change,
            count_if(bikes = 0) as empty_change_rows
        from r
        group by month_key
        order by month_key
        """
    ).df()
    return summary, monthly


def sampled_city_query(parquet_glob: str, sample_minutes: int) -> str:
    return f"""
    with r as (
        select * from read_parquet('{quote_path(parquet_glob)}', filename = true)
    ),
    station_latest as (
        select
            nuid,
            arg_max(name, timestamp) as station_name,
            arg_max(latitude, timestamp) as latitude,
            arg_max(longitude, timestamp) as longitude
        from r
        group by nuid
    ),
    events as (
        select
            nuid,
            timestamp as start_ts,
            lead(timestamp) over (partition by nuid order by timestamp) as end_ts,
            bikes
        from r
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
            strftime(timezone('Europe/Prague', g.sample_ts at time zone 'UTC'), '%Y-%m') as local_month,
            strftime(timezone('Europe/Prague', g.sample_ts at time zone 'UTC'), '%H:00') as local_hour,
            e.nuid,
            s.station_name,
            s.latitude,
            s.longitude,
            e.bikes
        from grid g
        join events e
            on g.sample_ts >= e.start_ts
            and (e.end_ts is null or g.sample_ts < e.end_ts)
        join station_latest s using (nuid)
    )
    """


def load_city_sampled(
    con: duckdb.DuckDBPyConnection,
    parquet_glob: str,
    sample_minutes: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    query = sampled_city_query(parquet_glob, sample_minutes)
    monthly = con.execute(
        query
        + """
        , city_state as (
            select
                sample_ts,
                local_month,
                sum(bikes) as total_bikes,
                count(*) as observed_stations,
                count_if(bikes > 0) as stations_with_bikes,
                count_if(bikes = 0) as empty_stations
            from sampled
            group by sample_ts, local_month
        )
        select
            local_month,
            count(*) as samples,
            round(avg(total_bikes), 2) as avg_total_bikes,
            round(avg(total_bikes / observed_stations), 3) as avg_bikes_per_station,
            round(avg(stations_with_bikes), 2) as avg_stations_with_bikes,
            round(avg(empty_stations::double / observed_stations), 3) as empty_station_rate
        from city_state
        group by local_month
        order by local_month
        """
    ).df()
    hourly = con.execute(
        query
        + """
        , city_state as (
            select
                sample_ts,
                local_hour,
                sum(bikes) as total_bikes,
                count(*) as observed_stations,
                count_if(bikes = 0) as empty_stations
            from sampled
            group by sample_ts, local_hour
        )
        select
            local_hour,
            count(*) as samples,
            round(avg(total_bikes / observed_stations), 3) as avg_bikes_per_station,
            round(avg(empty_stations::double / observed_stations), 3) as empty_station_rate
        from city_state
        group by local_hour
        order by local_hour
        """
    ).df()
    month_hour = con.execute(
        query
        + """
        , city_state as (
            select
                sample_ts,
                local_month,
                local_hour,
                count(*) as observed_stations,
                count_if(bikes = 0) as empty_stations
            from sampled
            group by sample_ts, local_month, local_hour
        )
        select
            local_month,
            local_hour,
            round(avg(empty_stations::double / observed_stations), 3) as empty_station_rate
        from city_state
        group by local_month, local_hour
        order by local_month, local_hour
        """
    ).df()
    station_rates = con.execute(
        query
        + """
        select
            nuid,
            any_value(station_name) as station_name,
            any_value(latitude) as latitude,
            any_value(longitude) as longitude,
            count(*) as samples,
            round(avg(bikes), 3) as avg_bikes,
            round(avg(case when bikes = 0 then 1.0 else 0.0 end), 3) as empty_rate,
            round(avg(case when bikes <= 1 then 1.0 else 0.0 end), 3) as low_rate
        from sampled
        group by nuid
        """
    ).df()
    return monthly, hourly, month_hour, station_rates


def local_sampled_query(
    parquet_glob: str,
    lat: float,
    lon: float,
    radius_m: int,
    sample_minutes: int,
) -> str:
    return f"""
    with r as (
        select * from read_parquet('{quote_path(parquet_glob)}', filename = true)
    ),
    station_latest as (
        select
            nuid,
            arg_max(name, timestamp) as station_name,
            arg_max(latitude, timestamp) as latitude,
            arg_max(longitude, timestamp) as longitude
        from r
        group by nuid
    ),
    station_dist as (
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
        select * from station_dist where distance_m <= {radius_m}
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
            strftime(timezone('Europe/Prague', g.sample_ts at time zone 'UTC'), '%Y-%m') as local_month,
            strftime(timezone('Europe/Prague', g.sample_ts at time zone 'UTC'), '%H:00') as local_hour,
            s.nuid,
            s.station_name,
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
    )
    """


def load_local_sampled(
    con: duckdb.DuckDBPyConnection,
    parquet_glob: str,
    lat: float,
    lon: float,
    radius_m: int,
    sample_minutes: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    query = local_sampled_query(parquet_glob, lat, lon, radius_m, sample_minutes)
    stations = con.execute(
        query
        + """
        select distinct nuid, station_name, distance_m
        from selected_stations
        order by distance_m
        """
    ).df()
    hourly = con.execute(
        query
        + """
        , area_state as (
            select
                sample_ts,
                local_hour,
                sum(coalesce(bikes, 0)) as total_bikes,
                count_if(coalesce(bikes, 0) > 0) as stations_with_bikes
            from sampled
            group by sample_ts, local_hour
        )
        select
            local_hour,
            count(*) as samples,
            round(avg(total_bikes), 2) as avg_bikes,
            round(avg(stations_with_bikes), 2) as avg_stations_with_bikes,
            round(avg(case when stations_with_bikes = 0 then 1.0 else 0.0 end), 3)
                as all_empty_rate
        from area_state
        group by local_hour
        order by local_hour
        """
    ).df()
    monthly_morning = con.execute(
        query
        + """
        , area_state as (
            select
                sample_ts,
                local_month,
                local_hour,
                sum(coalesce(bikes, 0)) as total_bikes,
                count_if(coalesce(bikes, 0) > 0) as stations_with_bikes
            from sampled
            group by sample_ts, local_month, local_hour
        )
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
    ).df()
    station_rates = con.execute(
        query
        + """
        select
            nuid,
            station_name,
            any_value(distance_m) as distance_m,
            count(*) as samples,
            round(avg(coalesce(bikes, 0)), 2) as avg_bikes,
            round(avg(case when coalesce(bikes, 0) = 0 then 1.0 else 0.0 end), 3)
                as empty_rate,
            round(avg(case when coalesce(bikes, 0) <= 1 then 1.0 else 0.0 end), 3)
                as low_rate
        from sampled
        group by nuid, station_name
        order by distance_m
        """
    ).df()
    station_hour = con.execute(
        query
        + """
        select
            station_name,
            local_hour,
            round(avg(case when coalesce(bikes, 0) = 0 then 1.0 else 0.0 end), 3)
                as empty_rate
        from sampled
        group by station_name, local_hour
        order by station_name, local_hour
        """
    ).df()
    return stations, hourly, monthly_morning, station_rates, station_hour


def load_flows(
    con: duckdb.DuckDBPyConnection,
    parquet_glob: str,
    lat: float,
    lon: float,
    radius_m: int,
    max_window_minutes: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    query = f"""
    with r as (
        select * from read_parquet('{quote_path(parquet_glob)}', filename = true)
    ),
    station_latest as (
        select
            nuid,
            arg_max(name, timestamp) as station_name,
            arg_max(latitude, timestamp) as latitude,
            arg_max(longitude, timestamp) as longitude
        from r
        group by nuid
    ),
    station_dist as (
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
    obs as (
        select
            r.timestamp,
            r.nuid,
            sd.station_name,
            sd.latitude,
            sd.longitude,
            sd.distance_m,
            unnest(json_extract(r.extra, '$.bike_uids')::varchar[]) as bike_uid
        from r
        join station_dist sd using (nuid)
        where
            r.extra is not null
            and json_array_length(json_extract(r.extra, '$.bike_uids')) > 0
    ),
    lagged as (
        select
            *,
            lag(timestamp) over (partition by bike_uid order by timestamp) as prev_timestamp,
            lag(nuid) over (partition by bike_uid order by timestamp) as prev_nuid,
            lag(station_name) over (partition by bike_uid order by timestamp) as prev_name,
            lag(latitude) over (partition by bike_uid order by timestamp) as prev_latitude,
            lag(longitude) over (partition by bike_uid order by timestamp) as prev_longitude,
            lag(distance_m) over (partition by bike_uid order by timestamp) as prev_distance_m
        from obs
    ),
    changed as (
        select
            *,
            date_diff('minute', prev_timestamp, timestamp) as window_minutes,
            cast(round(
                6371000 * 2 * asin(sqrt(
                    power(sin(radians((latitude - prev_latitude) / 2)), 2)
                    + cos(radians(prev_latitude)) * cos(radians(latitude))
                    * power(sin(radians((longitude - prev_longitude) / 2)), 2)
                ))
            ) as integer) as move_m,
            strftime(timezone('Europe/Prague', prev_timestamp at time zone 'UTC'), '%H:00') as local_hour,
            strftime(timezone('Europe/Prague', prev_timestamp at time zone 'UTC'), '%Y-%m') as local_month
        from lagged
        where
            prev_nuid is not null
            and nuid <> prev_nuid
            and date_diff('minute', prev_timestamp, timestamp) <= {max_window_minutes}
    )
    """
    hourly = con.execute(
        query
        + f"""
        select
            local_hour,
            count_if(prev_distance_m <= {radius_m} and distance_m > {radius_m}) as outbound,
            count_if(prev_distance_m > {radius_m} and distance_m <= {radius_m}) as inbound,
            count_if(prev_distance_m <= {radius_m} and distance_m <= {radius_m}) as internal
        from changed
        group by local_hour
        order by local_hour
        """
    ).df()
    monthly = con.execute(
        query
        + f"""
        select
            local_month,
            count_if(prev_distance_m <= {radius_m} and distance_m > {radius_m}) as outbound,
            count_if(prev_distance_m > {radius_m} and distance_m <= {radius_m}) as inbound,
            count_if(prev_distance_m <= {radius_m} and distance_m <= {radius_m}) as internal
        from changed
        group by local_month
        order by local_month
        """
    ).df()
    top_destinations = con.execute(
        query
        + f"""
        select
            station_name,
            any_value(distance_m) as distance_m,
            count(*) as moves,
            round(avg(window_minutes), 1) as avg_window_minutes,
            round(avg(move_m), 0) as avg_move_m
        from changed
        where prev_distance_m <= {radius_m} and distance_m > {radius_m}
        group by nuid, station_name
        order by moves desc, station_name
        limit 20
        """
    ).df()
    origin_stations = con.execute(
        query
        + f"""
        select
            prev_name as station_name,
            any_value(prev_distance_m) as distance_m,
            count(*) as outbound
        from changed
        where prev_distance_m <= {radius_m} and distance_m > {radius_m}
        group by prev_nuid, prev_name
        order by outbound desc, station_name
        """
    ).df()
    summary = con.execute(
        query
        + f"""
        select
            count(*) as all_city_moves,
            count_if(prev_distance_m <= {radius_m} and distance_m > {radius_m}) as outbound,
            count_if(prev_distance_m > {radius_m} and distance_m <= {radius_m}) as inbound,
            count_if(prev_distance_m <= {radius_m} and distance_m <= {radius_m}) as internal,
            count(distinct bike_uid) as bikes
        from changed
        """
    ).df()
    return summary, hourly, monthly, top_destinations, origin_stations


def save_dataframes(output_dir: Path, frames: dict[str, pd.DataFrame]) -> None:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(tables_dir / f"{name}.csv", index=False)


def frame_to_markdown(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_line_plot(path: Path, frame: pd.DataFrame, x: str, ys: list[str], title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for y in ys:
        ax.plot(frame[x], frame[y], marker="o", linewidth=1.8, label=y.replace("_", " "))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def with_percent_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    plot_frame = frame.copy()
    for column in columns:
        plot_frame[f"{column}_pct"] = plot_frame[column] * 100
    return plot_frame


def save_heatmap(path: Path, matrix: pd.DataFrame, title: str, cbar_label: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    im = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_barh(path: Path, frame: pd.DataFrame, label_col: str, value_col: str, title: str, xlabel: str) -> None:
    plot_frame = frame.sort_values(value_col).tail(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_frame[label_col], plot_frame[value_col], color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_station_map(
    path: Path,
    station_rates: pd.DataFrame,
    local_stations: pd.DataFrame,
    title: str,
) -> None:
    local_ids = set(local_stations["nuid"].astype(str))
    colors = station_rates["empty_rate"].to_numpy()
    sizes = 12 + station_rates["avg_bikes"].clip(lower=0).to_numpy() * 10
    edgecolors = ["#D62728" if str(nuid) in local_ids else "white" for nuid in station_rates["nuid"]]

    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(
        station_rates["longitude"],
        station_rates["latitude"],
        c=colors,
        s=sizes,
        cmap="magma_r",
        alpha=0.82,
        linewidth=0.7,
        edgecolors=edgecolors,
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Empty rate")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_flow_hourly(path: Path, hourly: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(hourly["local_hour"], hourly["outbound"], marker="o", label="outbound")
    ax.plot(hourly["local_hour"], hourly["inbound"], marker="o", label="inbound")
    ax.bar(hourly["local_hour"], hourly["outbound"] - hourly["inbound"], alpha=0.22, label="net out")
    ax.set_title("Inferred local bike moves by departure hour")
    ax.set_ylabel("Moves")
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build_report(
    path: Path,
    args: argparse.Namespace,
    location_label: str,
    summary: pd.DataFrame,
    city_monthly: pd.DataFrame,
    local_hourly: pd.DataFrame,
    local_monthly: pd.DataFrame,
    local_station_rates: pd.DataFrame,
    flow_summary: pd.DataFrame,
    flow_hourly: pd.DataFrame,
    top_destinations: pd.DataFrame,
    figures: dict[str, Path],
) -> None:
    summary_row = summary.iloc[0]
    flow_row = flow_summary.iloc[0]
    worst_months = local_monthly.sort_values("all_empty_rate", ascending=False).head(5)
    morning_flow = flow_hourly[flow_hourly["local_hour"].between("05:00", "10:00")]
    morning_out = int(morning_flow["outbound"].sum())
    morning_in = int(morning_flow["inbound"].sum())

    lines = [
        "# Nextbike Brno Deep Historical Analysis",
        "",
        "## Scope",
        "",
        f"- Network: `{args.network}`",
        f"- Local origin: `{args.lat:.6f}, {args.lon:.6f}`",
        f"- Geocoder/source: {location_label}",
        f"- Local radius: `{args.radius_m} m`",
        f"- Raw CityBikes rows: `{int(summary_row['row_count'])}`",
        f"- Raw window: `{summary_row['first_timestamp']}` to `{summary_row['latest_timestamp']}`",
        f"- Distinct station IDs: `{int(summary_row['station_ids'])}`",
        "",
        "Method note: CityBikes files contain station status changes, not uniform snapshots. "
        "The analysis reconstructs station state by carrying the last known station status forward "
        f"on a `{args.city_sample_minutes}` minute grid for citywide plots and a "
        f"`{args.local_sample_minutes}` minute grid for the local area.",
        "",
        "## Key Findings",
        "",
        f"- The local `{args.radius_m} m` area is structurally sparse: the five closest stations average "
        f"`{local_station_rates['avg_bikes'].sum():.2f}` bikes combined across the reconstructed history.",
        f"- The local area is completely empty `{local_hourly['all_empty_rate'].mean() * 100:.1f}%` of sampled hourly periods on average across the day.",
        f"- Morning all-empty risk is highly seasonal. The worst months in the 06:00-10:00 window are "
        + ", ".join(
            f"{row.local_month} ({row.all_empty_rate * 100:.1f}%)"
            for row in worst_months.itertuples()
        )
        + ".",
        f"- Inferred bike UID movements do not support a simple one-way story over the full period: "
        f"`{int(flow_row['outbound'])}` outbound local moves vs `{int(flow_row['inbound'])}` inbound moves.",
        f"- The morning commute window is more nuanced: 05:00-10:00 has `{morning_out}` outbound vs `{morning_in}` inbound inferred moves. "
        "The 06:00 hour is the clearest outbound-biased hour, but 08:00-09:00 already has substantial inbound activity.",
        "- The stronger explanation is low local inventory plus high station-level empty rates, not only one-way downhill depletion.",
        "",
        "## Figures",
        "",
    ]
    for title, figure_path in figures.items():
        rel = figure_path.relative_to(path.parent)
        lines.append(f"- [{title}]({rel})")

    lines.extend(
        [
            "",
            "## Local Station Availability",
            "",
            frame_to_markdown(
                local_station_rates[
                    ["station_name", "distance_m", "avg_bikes", "empty_rate", "low_rate"]
                ]
            ),
            "",
            "## Worst Local Morning Months",
            "",
            frame_to_markdown(worst_months),
            "",
            "## Top Inferred Outbound Destinations",
            "",
            frame_to_markdown(top_destinations.head(12)),
            "",
            "## Caveats",
            "",
            "- Bike movement routes are inferred from `bike_uids` appearing at one station and later another station. "
            "They are not GPS ride traces.",
            "- Timestamps are treated as UTC and converted to Europe/Prague for local-hour grouping.",
            "- This does not include elevation data. To test the downhill hypothesis directly, the next step is adding a DEM/elevation lookup for origin and destination stations.",
            "- CityBikes terms ask for attribution: Bike-share data by CityBikes contributors and available from https://data.citybik.es.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.lat, args.lon, location_label = resolve_location(args)
    ensure_network_files(args.data_dir, args.network)

    output_dir = args.output_dir
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    parquet_glob = network_parquet_glob(args.data_dir, args.network)
    setup_plot_style()
    with duckdb.connect() as con:
        summary, raw_monthly = load_raw_summary(con, parquet_glob)
        city_monthly, city_hourly, city_month_hour, city_station_rates = load_city_sampled(
            con, parquet_glob, args.city_sample_minutes
        )
        local_stations, local_hourly, local_monthly, local_station_rates, local_station_hour = (
            load_local_sampled(
                con,
                parquet_glob,
                args.lat,
                args.lon,
                args.radius_m,
                args.local_sample_minutes,
            )
        )
        flow_summary, flow_hourly, flow_monthly, top_destinations, origin_stations = load_flows(
            con,
            parquet_glob,
            args.lat,
            args.lon,
            args.radius_m,
            args.max_move_window_minutes,
        )

    frames = {
        "raw_monthly": raw_monthly,
        "city_monthly": city_monthly,
        "city_hourly": city_hourly,
        "city_station_rates": city_station_rates,
        "local_stations": local_stations,
        "local_hourly": local_hourly,
        "local_monthly_morning": local_monthly,
        "local_station_rates": local_station_rates,
        "local_station_hour": local_station_hour,
        "flow_summary": flow_summary,
        "flow_hourly": flow_hourly,
        "flow_monthly": flow_monthly,
        "top_destinations": top_destinations,
        "origin_stations": origin_stations,
    }
    save_dataframes(output_dir, frames)

    figures = {
        "Citywide monthly availability": figures_dir / "city_monthly_availability.png",
        "Citywide empty-rate heatmap": figures_dir / "city_empty_rate_month_hour.png",
        "Station empty-rate map": figures_dir / "station_empty_rate_map.png",
        "Local hourly availability": figures_dir / "local_hourly_availability.png",
        "Local morning seasonality": figures_dir / "local_morning_monthly.png",
        "Local station empty-rate heatmap": figures_dir / "local_station_hour_empty_rate.png",
        "Local inferred flows by hour": figures_dir / "local_flow_hourly.png",
        "Top outbound destinations": figures_dir / "top_outbound_destinations.png",
    }

    save_line_plot(
        figures["Citywide monthly availability"],
        with_percent_columns(city_monthly, ["empty_station_rate"]),
        "local_month",
        ["avg_bikes_per_station", "empty_station_rate_pct"],
        "Citywide monthly reconstructed availability",
        "Bikes per station / empty station %",
    )
    city_matrix = city_month_hour.pivot(
        index="local_month", columns="local_hour", values="empty_station_rate"
    ).fillna(0)
    save_heatmap(
        figures["Citywide empty-rate heatmap"],
        city_matrix,
        "Citywide empty station rate by month and hour",
        "Empty station rate",
    )
    save_station_map(
        figures["Station empty-rate map"],
        city_station_rates,
        local_stations,
        "Brno station empty rate, local stations highlighted",
    )
    save_line_plot(
        figures["Local hourly availability"],
        with_percent_columns(local_hourly, ["all_empty_rate"]),
        "local_hour",
        ["avg_bikes", "avg_stations_with_bikes", "all_empty_rate_pct"],
        f"Local availability within {args.radius_m} m",
        "Bikes / stations / all-empty %",
    )
    save_line_plot(
        figures["Local morning seasonality"],
        with_percent_columns(local_monthly, ["all_empty_rate"]),
        "local_month",
        ["avg_bikes", "avg_stations_with_bikes", "all_empty_rate_pct"],
        "Local morning availability by month, 06:00-10:00",
        "Bikes / stations / all-empty %",
    )
    local_station_matrix = local_station_hour.pivot(
        index="station_name", columns="local_hour", values="empty_rate"
    ).fillna(0)
    save_heatmap(
        figures["Local station empty-rate heatmap"],
        local_station_matrix,
        "Local station empty rate by hour",
        "Empty rate",
    )
    save_flow_hourly(figures["Local inferred flows by hour"], flow_hourly)
    save_barh(
        figures["Top outbound destinations"],
        top_destinations,
        "station_name",
        "moves",
        "Top inferred destinations from local stations",
        "Inferred outbound moves",
    )

    build_report(
        output_dir / "report.md",
        args,
        location_label,
        summary,
        city_monthly,
        local_hourly,
        local_monthly,
        local_station_rates,
        flow_summary,
        flow_hourly,
        top_destinations,
        figures,
    )
    print(f"wrote {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
