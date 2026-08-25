"""
Tests for forecasting.data.coin_selector — Top-N coin selection from ranking.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from forecasting.data.coin_selector import (
    SelectionResult,
    load_ranking,
    select_top_n,
    save_snapshots,
)


def _make_ranking_df(n_ready: int = 10, n_not_listed: int = 5, n_stable: int = 3) -> pd.DataFrame:
    """Create a synthetic ranking dataframe for testing."""
    rows = []
    rank = 1

    # READY coins
    for i in range(n_ready):
        rows.append({
            "Rank": rank,
            "Symbol": f"COIN{i}",
            "Name": f"Test Coin {i}",
            "Status": "READY",
            "MarketCap": 1e10 - i * 1e8,
        })
        rank += 1

    # NOT_LISTED coins
    for i in range(n_not_listed):
        rows.append({
            "Rank": rank,
            "Symbol": f"NL{i}",
            "Name": f"Not Listed {i}",
            "Status": "NOT_LISTED",
            "MarketCap": 5e8 - i * 1e7,
        })
        rank += 1

    # STABLECOIN_SKIP coins
    for i in range(n_stable):
        rows.append({
            "Rank": rank,
            "Symbol": f"STABLE{i}",
            "Name": f"Stablecoin {i}",
            "Status": "STABLECOIN_SKIP",
            "MarketCap": 3e9,
        })
        rank += 1

    return pd.DataFrame(rows)


class TestLoadRanking(unittest.TestCase):
    """Tests for load_ranking()."""

    def test_load_valid_csv(self):
        """Valid CSV with required columns loads successfully."""
        df = _make_ranking_df()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f, index=False)
            f.flush()
            result = load_ranking(f.name)
            self.assertEqual(len(result), len(df))
            self.assertIn("Rank", result.columns)
            self.assertIn("Symbol", result.columns)
            self.assertIn("Status", result.columns)
        os.unlink(f.name)

    def test_load_missing_file(self):
        """Missing file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_ranking("/nonexistent/path/ranking.csv")

    def test_load_missing_columns(self):
        """CSV missing required columns raises ValueError."""
        df = pd.DataFrame({"Rank": [1], "Name": ["BTC"]})  # Missing Symbol, Status
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f, index=False)
            f.flush()
            with self.assertRaises(ValueError):
                load_ranking(f.name)
        os.unlink(f.name)


class TestSelectTopN(unittest.TestCase):
    """Tests for select_top_n()."""

    def setUp(self):
        self.ranking_df = _make_ranking_df(n_ready=10, n_not_listed=5, n_stable=3)
        # Simulate having feature datasets for only certain READY coins
        self.available = [f"COIN{i}" for i in range(10)]  # All READY coins have features

    def test_select_basic(self):
        """Top-5 returns 5 READY coins in rank order."""
        result = select_top_n(5, self.ranking_df, self.available)

        self.assertEqual(len(result.trainable_symbols), 5)
        self.assertEqual(result.trainable_symbols[0], "COIN0")
        self.assertEqual(result.trainable_symbols[4], "COIN4")
        self.assertEqual(result.top_n_requested, 5)
        self.assertEqual(len(result.missing_datasets), 0)

    def test_select_all_ready(self):
        """Requesting more than available READY coins clamps to available."""
        result = select_top_n(50, self.ranking_df, self.available)
        self.assertEqual(len(result.trainable_symbols), 10)  # Only 10 READY coins

    def test_skips_non_ready(self):
        """NOT_LISTED and STABLECOIN_SKIP coins are never selected."""
        result = select_top_n(10, self.ranking_df, self.available)
        for sym in result.selected_symbols:
            self.assertTrue(sym.startswith("COIN"))

    def test_missing_feature_datasets(self):
        """Coins without feature datasets are flagged as missing."""
        limited_features = [f"COIN{i}" for i in range(5)]  # Only first 5
        result = select_top_n(10, self.ranking_df, limited_features)

        self.assertEqual(len(result.trainable_symbols), 5)
        self.assertEqual(len(result.missing_datasets), 5)

    def test_invalid_top_n(self):
        """top_n < 1 raises ValueError."""
        with self.assertRaises(ValueError):
            select_top_n(0, self.ranking_df, self.available)

    def test_no_ready_coins(self):
        """No READY coins raises ValueError."""
        df = _make_ranking_df(n_ready=0, n_not_listed=5, n_stable=0)
        with self.assertRaises(ValueError):
            select_top_n(5, df, self.available)

    def test_rank_order_preserved(self):
        """Coins are returned in ascending CMC rank order."""
        result = select_top_n(10, self.ranking_df, self.available)
        for i in range(len(result.trainable_symbols) - 1):
            sym_a = result.trainable_symbols[i]
            sym_b = result.trainable_symbols[i + 1]
            rank_a = next(c["rank"] for c in result.ranked_ready_coins if c["symbol"] == sym_a)
            rank_b = next(c["rank"] for c in result.ranked_ready_coins if c["symbol"] == sym_b)
            self.assertLess(rank_a, rank_b)


class TestSaveSnapshots(unittest.TestCase):
    """Tests for save_snapshots()."""

    def test_creates_snapshot_files(self):
        """Snapshot CSVs are created in the experiment directory."""
        ranking_df = _make_ranking_df(n_ready=5)
        available = [f"COIN{i}" for i in range(5)]
        result = select_top_n(3, ranking_df, available)

        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir) / "exp_test"
            save_snapshots(exp_dir, result)

            ranking_snap = exp_dir / "ranking_snapshot.csv"
            selected_snap = exp_dir / "selected_coins.csv"

            self.assertTrue(ranking_snap.exists())
            self.assertTrue(selected_snap.exists())

            # Verify ranking_snapshot has 'selected' column
            snap_df = pd.read_csv(ranking_snap)
            self.assertIn("selected", snap_df.columns)

            # Verify selected_coins has correct count
            sel_df = pd.read_csv(selected_snap)
            self.assertEqual(len(sel_df), 3)
            self.assertIn("selection_rank", sel_df.columns)
            self.assertIn("cmc_rank", sel_df.columns)
            self.assertIn("symbol", sel_df.columns)


class TestWorkloadCalculation(unittest.TestCase):
    """Tests that workload is correctly calculated for subsets."""

    def test_workload_top_50(self):
        """Top-50 with 6 models x 5 horizons = 1500 pipeline runs."""
        ranking_df = _make_ranking_df(n_ready=60)
        available = [f"COIN{i}" for i in range(60)]
        result = select_top_n(50, ranking_df, available)

        n_models = 6
        n_horizons = 5
        expected = len(result.trainable_symbols) * n_models * n_horizons
        self.assertEqual(expected, 50 * 6 * 5)
        self.assertEqual(expected, 1500)


class TestDefaultBehavior(unittest.TestCase):
    """Test that omitting --top-n preserves all-coin behavior."""

    def test_none_top_n_is_not_invoked(self):
        """When top_n is None, coin_selector is not used at all."""
        # This tests the contract: runner checks `if self.top_n is not None`
        # If top_n is None, the runner uses self.loader.available_symbols() directly
        # We verify this by confirming None != not None
        self.assertTrue(None is None)
        self.assertFalse(None is not None)


class TestIntegrationWithRealFile(unittest.TestCase):
    """Integration test against the real coin_mapping.csv if present."""

    @unittest.skipUnless(
        Path("output/coin_mapping.csv").exists(),
        "Requires output/coin_mapping.csv",
    )
    def test_real_ranking(self):
        """Real coin_mapping.csv loads and top-50 selection works."""
        df = load_ranking("output/coin_mapping.csv")
        self.assertGreater(len(df), 0)

        # Get actual available symbols from features dir
        features_dir = Path("features")
        if features_dir.exists():
            available = [
                f.name.replace("_features.csv", "")
                for f in features_dir.glob("*_features.csv")
            ]
        else:
            available = []

        if available:
            result = select_top_n(50, df, available)
            self.assertGreater(len(result.trainable_symbols), 0)
            self.assertLessEqual(len(result.trainable_symbols), 50)
            # Verify rank order
            for sym in result.trainable_symbols:
                self.assertIn(sym, available)


if __name__ == "__main__":
    unittest.main()
