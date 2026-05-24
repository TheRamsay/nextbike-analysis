from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


DEFAULT_FEEDS = (
    "system_information",
    "vehicle_types",
    "station_information",
    "station_status",
    "free_bike_status",
    "system_regions",
    "system_pricing_plans",
)


@dataclass(frozen=True)
class FeedRef:
    name: str
    url: str


class GbfsClient:
    def __init__(self, gbfs_url: str, timeout_seconds: float = 20.0) -> None:
        self.gbfs_url = gbfs_url
        self.timeout_seconds = timeout_seconds

    def fetch_json(self, url: str) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout_seconds, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.json()

    def discover_feeds(self, language: str = "en") -> list[FeedRef]:
        gbfs = self.fetch_json(self.gbfs_url)
        feeds_by_language = gbfs.get("data", {})
        feeds = feeds_by_language.get(language)
        if feeds is None:
            available = ", ".join(sorted(feeds_by_language))
            raise ValueError(f"Language {language!r} not found in GBFS feed. Available: {available}")

        return [FeedRef(name=item["name"], url=item["url"]) for item in feeds["feeds"]]

    def fetch_feeds(
        self,
        language: str = "en",
        names: tuple[str, ...] | list[str] = DEFAULT_FEEDS,
    ) -> dict[str, dict[str, Any]]:
        wanted = set(names)
        feed_refs = [feed for feed in self.discover_feeds(language) if feed.name in wanted]
        return {feed.name: self.fetch_json(feed.url) for feed in feed_refs}

