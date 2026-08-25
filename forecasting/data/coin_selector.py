"""
forecasting.data.coin_selector -- Top-N Coin Selection from Ranking Source.

Reads the project's coin_mapping.csv (output/coin_mapping.csv) which contains
CMC-ranked cryptocurrencies with a Status column (READY / NOT_LISTED / STABLECOIN_SKIP).

When --top-n N is requested, selects the first N coins with Status == READY,
ordered by their CMC Rank, and verifies each has a corresponding feature dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Default ranking source — the coin_mapping.csv produced by the data-engineering pipeline
DEFAULT_RANKING_FILE = Path("output") / "coin_mapping.csv"


@dataclass
class SelectionResult:
    """Container for the coin selection output."""

    ranking_source: str
    top_n_requested: int
    ranked_ready_coins: list[dict]      # All READY coins from the ranking, in rank order
    selected_symbols: list[str]         # The top-N symbols selected for training
    available_symbols: list[str]        # Symbols that have feature datasets
    missing_datasets: list[str]         # Selected but missing a feature file
    trainable_symbols: list[str]        # Final list: selected AND have feature data
    ranking_df: pd.DataFrame            # Full ranking dataframe for snapshot


def load_ranking(ranking_path: Path | str | None = None) -> pd.DataFrame:
    """
    Load and validate the coin ranking CSV.

    Parameters
    ----------
    ranking_path : Path or str, optional
        Path to the ranking CSV. Defaults to output/coin_mapping.csv.

    Returns
    -------
    pd.DataFrame
        Ranking dataframe with at least Rank, Symbol, Status columns.
    """
    path = Path(ranking_path) if ranking_path else DEFAULT_RANKING_FILE

    if not path.exists():
        raise FileNotFoundError(
            f"Ranking file not found: {path}. "
            f"Run ranking.py or mapping.py to generate it."
        )

    df = pd.read_csv(path)

    required_cols = {"Rank", "Symbol", "Status"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Ranking file {path} missing required columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    logger.info("Loaded ranking from %s: %d entries", path, len(df))
    return df


def select_top_n(
    top_n: int,
    ranking_df: pd.DataFrame,
    available_symbols: list[str],
    ranking_source: str = "",
) -> SelectionResult:
    """
    Select the top N READY-status coins from the ranking.

    Parameters
    ----------
    top_n : int
        Number of coins to select.
    ranking_df : pd.DataFrame
        Full ranking dataframe with Rank, Symbol, Status columns.
    available_symbols : list[str]
        Symbols that have feature datasets on disk.
    ranking_source : str
        Human-readable description of the ranking source for display.

    Returns
    -------
    SelectionResult
        Complete selection result with audit information.
    """
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}")

    # Filter to READY coins and sort by rank
    ready = ranking_df[ranking_df["Status"] == "READY"].copy()
    ready = ready.sort_values("Rank", ascending=True).reset_index(drop=True)

    if len(ready) == 0:
        raise ValueError("No coins with Status == 'READY' found in ranking file")

    # Build ordered list of READY coins
    ranked_ready = []
    for _, row in ready.iterrows():
        ranked_ready.append({
            "rank": int(row["Rank"]),
            "symbol": str(row["Symbol"]),
            "name": str(row.get("Name", "")),
            "market_cap": row.get("MarketCap"),
        })

    # Select top N
    if top_n > len(ranked_ready):
        logger.warning(
            "Requested top-%d but only %d READY coins exist. Using all %d.",
            top_n, len(ranked_ready), len(ranked_ready),
        )
        top_n = len(ranked_ready)

    selected = ranked_ready[:top_n]
    selected_symbols = [c["symbol"] for c in selected]

    # Check feature dataset availability
    available_set = set(available_symbols)
    missing = [s for s in selected_symbols if s not in available_set]
    trainable = [s for s in selected_symbols if s in available_set]

    return SelectionResult(
        ranking_source=ranking_source or str(DEFAULT_RANKING_FILE),
        top_n_requested=top_n,
        ranked_ready_coins=ranked_ready,
        selected_symbols=selected_symbols,
        available_symbols=available_symbols,
        missing_datasets=missing,
        trainable_symbols=trainable,
        ranking_df=ranking_df,
    )


def save_snapshots(exp_dir: Path, result: SelectionResult) -> None:
    """
    Save ranking and selection snapshots to the experiment directory.

    Creates:
        exp_dir/ranking_snapshot.csv  — Full ranking with 'selected' column
        exp_dir/selected_coins.csv   — Only the selected coins
    """
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 1. ranking_snapshot.csv — full ranking with selection flag
    snapshot = result.ranking_df.copy()
    snapshot["selected"] = snapshot["Symbol"].isin(result.trainable_symbols)
    snapshot_path = exp_dir / "ranking_snapshot.csv"
    snapshot.to_csv(snapshot_path, index=False)
    logger.info("Saved ranking snapshot to %s", snapshot_path)

    # 2. selected_coins.csv — ordered list of selected coins
    rows = []
    for i, sym in enumerate(result.trainable_symbols, 1):
        # Find the original rank info
        info = next((c for c in result.ranked_ready_coins if c["symbol"] == sym), {})
        rows.append({
            "selection_rank": i,
            "cmc_rank": info.get("rank", ""),
            "symbol": sym,
            "name": info.get("name", ""),
            "market_cap": info.get("market_cap", ""),
        })

    selected_df = pd.DataFrame(rows)
    selected_path = exp_dir / "selected_coins.csv"
    selected_df.to_csv(selected_path, index=False)
    logger.info("Saved selected coins to %s (%d coins)", selected_path, len(rows))


def print_selection_audit(result: SelectionResult) -> None:
    """Print a formatted audit of the coin selection to stdout."""
    lines = [
        "",
        "=" * 70,
        "  TOP-N COIN SELECTION",
        "=" * 70,
        f"  Ranking source : {result.ranking_source}",
        f"  Requested top-N: {result.top_n_requested}",
        f"  READY in ranking: {len(result.ranked_ready_coins)}",
        f"  Selected        : {len(result.selected_symbols)}",
        f"  Have features   : {len(result.trainable_symbols)}",
        f"  Missing features: {len(result.missing_datasets)}",
        "=" * 70,
    ]

    if result.missing_datasets:
        lines.append("")
        lines.append("  Missing feature datasets:")
        for sym in result.missing_datasets:
            lines.append(f"    - {sym}")
        lines.append("")

    lines.append("")
    lines.append("  SELECTED CRYPTOCURRENCIES")
    lines.append("  " + "-" * 46)
    lines.append(f"  {'Rank':<6} {'CMC':<6} {'Symbol':<10} {'Name'}")
    lines.append("  " + "-" * 46)

    for i, sym in enumerate(result.trainable_symbols, 1):
        info = next((c for c in result.ranked_ready_coins if c["symbol"] == sym), {})
        cmc_rank = info.get("rank", "?")
        name = info.get("name", "")
        lines.append(f"  {i:<6} {cmc_rank:<6} {sym:<10} {name}")

    lines.append("  " + "-" * 46)
    lines.append(f"  Selected for training: {len(result.trainable_symbols)}")
    lines.append("=" * 70)
    lines.append("")

    msg = "\n".join(lines)
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"), flush=True)
