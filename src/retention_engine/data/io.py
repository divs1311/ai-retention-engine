"""Read/write helpers. Parquet is the default; CSV is accepted for bring-your-own data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from retention_engine.config import Settings, get_settings
from retention_engine.data.schema import validate_orders, validate_users
from retention_engine.data.synthetic import generate


def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".csv", ".gz"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def write_table(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    return path


def generate_and_save(settings: Settings | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    settings = settings or get_settings()
    settings.ensure_dirs()
    ds = generate(settings.simulation)
    write_table(ds.users, settings.users_path)
    write_table(ds.orders, settings.orders_path)
    return ds.users, ds.orders


def load_dataset(
    settings: Settings | None = None, *, generate_if_missing: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate ``users`` and ``orders``; simulate them on first run."""
    settings = settings or get_settings()
    if not (settings.users_path.exists() and settings.orders_path.exists()):
        if not generate_if_missing:
            raise FileNotFoundError(
                f"No data in {settings.data_dir}. Run `retention-engine generate` first."
            )
        generate_and_save(settings)
    users = validate_users(read_table(settings.users_path))
    orders = validate_orders(read_table(settings.orders_path), users)
    return users, orders
