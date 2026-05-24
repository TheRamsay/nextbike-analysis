from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

BRNO_BOUNDARY_URL = "https://nominatim.openstreetmap.org/search"
OSM_ATTRIBUTION = "Boundary data © OpenStreetMap contributors, ODbL 1.0"
OSM_COPYRIGHT_URL = "https://www.openstreetmap.org/copyright"


@dataclass(frozen=True)
class BoundaryPolygon:
    outer: list[tuple[float, float]]
    holes: list[list[tuple[float, float]]]


@dataclass(frozen=True)
class BoundaryData:
    name: str
    osm_type: str
    osm_id: int
    licence: str
    polygons: list[BoundaryPolygon]


def load_brno_boundary(data_dir: Path, timeout_seconds: float) -> BoundaryData:
    cache_path = data_dir / "cache" / "brno_boundary_nominatim.json"
    if not cache_path.exists():
        fetch_brno_boundary(cache_path, timeout_seconds)

    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    geojson = payload["geojson"]
    return BoundaryData(
        name=str(payload.get("name") or "Brno"),
        osm_type=str(payload.get("osm_type") or ""),
        osm_id=int(payload.get("osm_id") or 0),
        licence=str(payload.get("licence") or OSM_ATTRIBUTION),
        polygons=parse_geojson_polygons(geojson),
    )


def fetch_brno_boundary(cache_path: Path, timeout_seconds: float) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {
        "User-Agent": "nextbike-analysis/0.1 (https://github.com/TheRamsay/nextbike-analysis)",
    }
    params = {
        "city": "Brno",
        "country": "Czechia",
        "format": "jsonv2",
        "polygon_geojson": "1",
        "limit": "1",
    }
    with httpx.Client(timeout=timeout_seconds, headers=headers) as client:
        response = client.get(BRNO_BOUNDARY_URL, params=params)
        response.raise_for_status()
    rows = response.json()
    if not rows:
        raise RuntimeError("Nominatim did not return a Brno boundary")
    row = rows[0]
    if "geojson" not in row:
        raise RuntimeError("Nominatim response did not include geojson")
    cache_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_geojson_polygons(geojson: dict[str, Any]) -> list[BoundaryPolygon]:
    geometry_type = geojson.get("type")
    coordinates = geojson.get("coordinates")
    if geometry_type == "Polygon":
        return [parse_polygon(coordinates)]
    if geometry_type == "MultiPolygon":
        return [parse_polygon(polygon_coordinates) for polygon_coordinates in coordinates]
    raise RuntimeError(f"Unsupported boundary geometry type: {geometry_type}")


def parse_polygon(coordinates: list[Any]) -> BoundaryPolygon:
    rings = [parse_ring(ring) for ring in coordinates]
    return BoundaryPolygon(outer=rings[0], holes=rings[1:])


def parse_ring(ring: list[Any]) -> list[tuple[float, float]]:
    return [(float(point[0]), float(point[1])) for point in ring]
