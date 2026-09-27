"""SQLite persistence layer for the risk engine.

Uses the Python standard library ``sqlite3`` driver with pandas for bulk loads
(no ORM is required for this workload). Tables: ``counterparties``,
``exposures``, ``risk_metrics``, ``scenario_definitions`` and
``stress_results``; DDL lives in ``sql/schema.sql`` and the analytical queries
in ``sql/queries.sql``.

Typical use::

    conn = build_database(features_df, risk_df, stress_results, scenarios)
    top = run_named_query(conn, "top_counterparties_by_ead", n=10)
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Mapping

import pandas as pd

from src import config as cfg
from src.stress_testing import THEMES, Scenario

logger = logging.getLogger(__name__)

# Columns loaded into each table (must match sql/schema.sql).
COUNTERPARTY_COLUMNS: list[str] = [
    "counterparty_id", "counterparty_name", "industry", "country", "region", "rating",
    "rating_prior_year", "annual_revenue", "revenue_prior_year", "total_assets",
    "total_debt", "cash", "current_assets", "current_liabilities", "ebitda",
    "interest_expense", "operating_cash_flow", "operating_cash_flow_prior_year",
    "debt_to_ebitda", "interest_coverage", "current_ratio", "debt_to_assets",
    "cash_to_debt", "ocf_to_debt", "ebitda_margin", "revenue_growth", "ocf_growth",
    "credit_health_score", "credit_score_band", "default_flag",
]
EXPOSURE_COLUMNS: list[str] = [
    "counterparty_id", "product_type", "notional", "maturity_years", "current_exposure",
    "current_exposure_prior_quarter", "fx_exposure", "interest_rate_exposure",
    "collateral", "collateral_type",
]
RISK_METRIC_COLUMNS: list[str] = [
    "counterparty_id", "pd", "lgd", "addon_factor", "maturity_factor", "pfe",
    "collateral_haircut", "eligible_collateral", "unsecured_exposure",
    "collateral_coverage", "ead", "expected_loss",
]
STRESS_RESULT_COLUMNS: list[str] = [
    "counterparty_id", "scenario", "stressed_pd", "stressed_lgd",
    "stressed_current_exposure", "stressed_pfe", "stressed_eligible_collateral",
    "stressed_ead", "stressed_expected_loss", "incremental_ead",
    "incremental_expected_loss",
]


# --------------------------------------------------------------------------- #
# Connection and schema
# --------------------------------------------------------------------------- #
def get_connection(db_path: Path | str = cfg.DATABASE_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with foreign keys enforced.

    Use ``":memory:"`` for an in-memory database.
    """
    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_schema(conn: sqlite3.Connection, schema_path: Path = cfg.SCHEMA_SQL_PATH) -> None:
    """(Re)create all tables, indexes and views from ``schema.sql``."""
    if not Path(schema_path).exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    conn.executescript(Path(schema_path).read_text(encoding="utf-8"))
    conn.execute("PRAGMA foreign_keys = ON")  # executescript may reset the session state
    conn.commit()


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _select(df: pd.DataFrame, columns: list[str], table: str) -> pd.DataFrame:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot load '{table}': missing columns {missing}")
    return df[columns]


def scenario_definitions_frame(
    scenarios: Mapping[str, Scenario], theme_keys: set[str] | None = None
) -> pd.DataFrame:
    """Flatten Scenario objects into rows for ``scenario_definitions``."""
    theme_keys = set(THEMES) if theme_keys is None else theme_keys
    rows = []
    for key, sc in scenarios.items():
        overrides = {
            k: dict(v) for k, v in sc.to_dict().items()
            if isinstance(v, Mapping) and v
        }
        rows.append(
            {
                "scenario": key,
                "scenario_type": "THEMATIC" if key in theme_keys else "MACRO",
                "description": sc.description,
                "pd_mult": sc.pd_mult,
                "lgd_mult": sc.lgd_mult,
                "exposure_mult": sc.exposure_mult,
                "fx_exposure_mult": sc.fx_exposure_mult,
                "ir_exposure_mult": sc.ir_exposure_mult,
                "collateral_haircut_add": sc.collateral_haircut_add,
                "overrides_json": json.dumps(overrides, sort_keys=True) if overrides else None,
            }
        )
    return pd.DataFrame(rows)


def load_tables(
    conn: sqlite3.Connection,
    features: pd.DataFrame,
    risk: pd.DataFrame,
    stress_results: pd.DataFrame,
    scenarios: Mapping[str, Scenario],
) -> dict[str, int]:
    """Insert data into all tables (schema must already exist).

    Args:
        features: Output of ``preprocessing.build_feature_table`` (raw + ratios).
        risk: Output of ``risk_engine.run_risk_engine`` (raw + risk metrics).
        stress_results: Long-format output of ``stress_testing.run_scenarios``.
        scenarios: Scenario objects for ``scenario_definitions``.

    Returns:
        Row counts per table.
    """
    tables = {
        "counterparties": _select(features, COUNTERPARTY_COLUMNS, "counterparties"),
        "exposures": _select(risk, EXPOSURE_COLUMNS, "exposures"),
        "risk_metrics": _select(risk, RISK_METRIC_COLUMNS, "risk_metrics"),
        "scenario_definitions": scenario_definitions_frame(scenarios),
        "stress_results": _select(stress_results, STRESS_RESULT_COLUMNS, "stress_results"),
    }
    counts: dict[str, int] = {}
    for name, frame in tables.items():  # dict order respects foreign-key dependencies
        frame.to_sql(name, conn, if_exists="append", index=False, chunksize=5000)
        counts[name] = len(frame)
    conn.commit()
    logger.info("Loaded tables: %s", counts)
    return counts


def build_database(
    features: pd.DataFrame,
    risk: pd.DataFrame,
    stress_results: pd.DataFrame,
    scenarios: Mapping[str, Scenario],
    db_path: Path | str = cfg.DATABASE_PATH,
) -> sqlite3.Connection:
    """Create the schema, load all tables and return an open connection."""
    conn = get_connection(db_path)
    create_schema(conn)
    load_tables(conn, features, risk, stress_results, scenarios)
    return conn


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #
_NAME_RE = re.compile(r"^--\s*name:\s*(\w+)\s*$", re.MULTILINE)


def load_named_queries(path: Path = cfg.QUERIES_SQL_PATH) -> dict[str, str]:
    """Parse ``queries.sql`` into ``{query_name: sql}``."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Query file not found: {path}")
    text = Path(path).read_text(encoding="utf-8")
    matches = list(_NAME_RE.finditer(text))
    queries: dict[str, str] = {}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        queries[match.group(1)] = text[match.end():end].strip()
    return queries


def run_query(conn: sqlite3.Connection, sql: str, params: Mapping[str, object] | None = None) -> pd.DataFrame:
    """Run a SQL statement and return the result as a DataFrame."""
    try:
        return pd.read_sql_query(sql, conn, params=dict(params or {}))
    except (sqlite3.Error, pd.errors.DatabaseError) as exc:
        raise RuntimeError(f"SQL execution failed: {exc}") from exc


# Sensible defaults so every named query can run without arguments.
DEFAULT_QUERY_PARAMS: dict[str, object] = {
    "n": 10,
    "pd_threshold": 0.10,
    "scenario": "SEVERE_RECESSION",
    "max_leverage": 6.0,
    "min_coverage": 2.0,
}


def run_named_query(conn: sqlite3.Connection, name: str, **params: object) -> pd.DataFrame:
    """Run a query from ``sql/queries.sql`` by name, e.g. ``n=10``."""
    queries = load_named_queries()
    if name not in queries:
        raise KeyError(f"Unknown query '{name}'. Available: {sorted(queries)}")
    merged = {**DEFAULT_QUERY_PARAMS, **params}
    return run_query(conn, queries[name], merged)
