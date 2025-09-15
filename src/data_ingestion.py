from typing import Optional, Dict, Any
import os
import pandas as pd
from supabase import create_client, Client

from src.config.config_loader import load_config
from src.utils.logger import get_logger


logger = get_logger(__name__)
BATCH_SIZE = 1000  # Supabase/PostgREST returns up to 1000 rows per page by default


def _default_config_path() -> str:
    """Resolve the config.yaml path relative to this file (src/config/config.yaml)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "config", "config.yaml"))


def get_supabase_client(config: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None) -> Client:
    """
    Create and return a Supabase client using values from config.

    Args:
        config: Optional pre-loaded configuration dict.
        config_path: Optional explicit path to a YAML config file.

    Returns:
        Supabase Client instance.
    """
    if config is None:
        config = load_config(config_path or _default_config_path())

    sup_cfg = config.get("supabase", {})
    url = sup_cfg.get("url")
    key = sup_cfg.get("key")

    if not url or not key:
        raise ValueError("Supabase URL or Key missing in config. Ensure supabase.url and supabase.key are set.")

    logger.info("Initializing Supabase client for URL: %s", url)
    return create_client(url, key)


def ingest_from_supabase(
    table_name: Optional[str] = None,
    *,
    select: str = "*",
    filters: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
    config_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch rows from a Supabase table into a pandas DataFrame using simple config-driven setup.

    Args:
        table_name: Table to read from. If not provided, uses supabase.table_name from config.
        select: Projection for columns, defaults to "*".
        filters: Simple equality filters as a dict, e.g., {"language": "en"}.
        limit: Optional row limit.
        config_path: Optional explicit path to config.yaml.

    Returns:
        pd.DataFrame with the queried rows.
    """
    config = load_config(config_path or _default_config_path())
    sb: Client = get_supabase_client(config=config)

    sup_cfg = config.get("supabase", {})
    table = table_name or sup_cfg.get("table_name")
    if not table:
        raise ValueError("Table name not provided and supabase.table_name missing from config.")

    logger.info("Ingesting data from Supabase table: %s", table)
    # Base query
    base_query = sb.table(table).select(select)
    if filters:
        for col, val in filters.items():
            base_query = base_query.eq(col, val)

    # Paginate to fetch all rows (or up to 'limit' if provided)
    all_records = []
    offset = 0
    remaining = limit if limit is not None else None

    while True:
        if remaining is None:
            batch = BATCH_SIZE
        else:
            if remaining <= 0:
                break
            batch = min(BATCH_SIZE, remaining)

        q = base_query.range(offset, offset + batch - 1)
        res = q.execute()
        records = getattr(res, "data", None)
        if records is None and isinstance(res, dict):
            records = res.get("data")

        if not records:
            break

        all_records.extend(records)
        fetched = len(records)
        offset += fetched
        if remaining is not None:
            remaining -= fetched
        if fetched < batch:
            # Last page
            break

    if not all_records:
        logger.warning("No data returned from Supabase query. Returning empty DataFrame.")
        return pd.DataFrame()

    df = pd.DataFrame.from_records(all_records)
    logger.info("Fetched %d rows from %s", len(df), table)
    return df


__all__ = ["get_supabase_client", "ingest_from_supabase"]