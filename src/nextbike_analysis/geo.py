from __future__ import annotations

import httpx
import typer


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

