"""End-to-end batch pipeline.

    python run_pipeline.py                # 5,000 counterparties, seed 42
    python run_pipeline.py --n 2500 --seed 7

Steps
-----
1. Generate the synthetic counterparty dataset  -> data/raw/
2. Validate + engineer credit features
3. Run the exposure / expected-loss engine
4. Run macro and thematic stress scenarios
5. Save processed tables and build the SQLite database -> data/processed/
6. Print a portfolio summary, scenario table and a few SQL results

All data is synthetic and all methodologies are simplified and educational.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src import config as cfg
from src.data_generator import generate_counterparties, save_dataset
from src.database import build_database, run_named_query
from src.preprocessing import build_feature_table
from src.risk_engine import aggregate_risk, portfolio_summary, run_risk_engine
from src.stress_testing import ALL_SCENARIOS, run_scenarios

logger = logging.getLogger("pipeline")


def run_pipeline(n: int, seed: int, skip_db: bool = False, db_path: Path = cfg.DATABASE_PATH) -> dict[str, object]:
    """Run the full pipeline and return the main artefacts."""
    raw = generate_counterparties(n, seed)
    save_dataset(raw, cfg.RAW_DATA_PATH)

    features = build_feature_table(raw)
    risk = run_risk_engine(features)
    results, summary = run_scenarios(risk, ALL_SCENARIOS)

    cfg.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    risk.to_csv(cfg.PROCESSED_DATA_PATH, index=False)
    results.to_csv(cfg.STRESS_RESULTS_PATH, index=False)

    conn = None
    if not skip_db:
        conn = build_database(features, risk, results, ALL_SCENARIOS, db_path)
    return {"raw": raw, "risk": risk, "stress_results": results, "summary": summary, "conn": conn}


def _print_report(artefacts: dict[str, object]) -> None:
    risk: pd.DataFrame = artefacts["risk"]  # type: ignore[assignment]
    summary: pd.DataFrame = artefacts["summary"]  # type: ignore[assignment]
    pd.options.display.float_format = "{:,.3f}".format
    pd.options.display.width = 200

    print("\n=== PORTFOLIO SUMMARY (EUR mn) ===")
    for key, value in portfolio_summary(risk).items():
        print(f"{key:>28}: {value:,.4f}" if isinstance(value, float) else f"{key:>28}: {value}")

    print("\n=== EXPECTED LOSS BY RATING ===")
    print(aggregate_risk(risk, "rating")[["rating", "n_counterparties", "ead", "expected_loss",
                                          "ead_weighted_pd", "el_to_ead_bps"]].to_string(index=False))

    print("\n=== SCENARIO SUMMARY (EUR mn) ===")
    cols = ["scenario", "base_ead", "stressed_ead", "base_expected_loss", "stressed_expected_loss",
            "incremental_expected_loss", "expected_loss_pct_change"]
    print(summary[cols].to_string(index=False))

    conn = artefacts["conn"]
    if conn is not None:
        print("\n=== SQL: top 5 counterparties by EAD ===")
        print(run_named_query(conn, "top_counterparties_by_ead", n=5).to_string(index=False))
        print("\n=== SQL: industry concentration (HHI) ===")
        print(run_named_query(conn, "industry_concentration_hhi").to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the counterparty risk pipeline.")
    parser.add_argument("--n", type=int, default=cfg.DEFAULT_N_COUNTERPARTIES)
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_SEED)
    parser.add_argument("--skip-db", action="store_true", help="Do not build the SQLite database")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    _print_report(run_pipeline(args.n, args.seed, args.skip_db))


if __name__ == "__main__":
    main()
