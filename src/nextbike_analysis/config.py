from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    gbfs_url: str = "https://gbfs.nextbike.net/maps/gbfs/v2/nextbike_te/gbfs.json"
    data_dir: Path = Path("data")
    db_path: Path = Path("data/nextbike.duckdb")
    request_timeout_seconds: float = Field(default=20.0, gt=0)

