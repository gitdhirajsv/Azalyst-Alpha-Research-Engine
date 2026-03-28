"""
Azalyst v4 — Test Suite
pytest -v tests/test_azalyst.py
"""
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

# Ensure repo root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Database (azalyst_db.py)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAzalystDB:
    @pytest.fixture(autouse=True)
    def setup_db(self, tmp_path):
        from azalyst_db import AzalystDB
        self.db = AzalystDB(str(tmp_path / "test.db"))
        yield
        self.db.close()

    def test_start_run(self):
        rid = self.db.start_run("r1", {"gpu": True})
        assert rid == "r1"
        runs = self.db.list_runs()
        assert len(runs) == 1
        assert runs.iloc[0]["status"] == "running"

    def test_finish_run(self):
        self.db.start_run("r1")
        self.db.finish_run("r1", "completed")
        runs = self.db.list_runs()
        assert runs.iloc[0]["status"] == "completed"
        assert runs.iloc[0]["finished_at"] is not None

    def test_insert_and_get_trades(self):
        self.db.start_run("r1")
        trades = [
            {"week": 1, "symbol": "BTCUSDT", "signal": "BUY",
             "pred_prob": 0.65, "pnl_percent": 1.5, "meta_size": 0.9},
            {"week": 1, "symbol": "ETHUSDT", "signal": "SELL",
             "pred_prob": 0.35, "pnl_percent": -0.3, "meta_size": 0.7},
        ]
        self.db.insert_trades("r1", trades)
        df = self.db.get_trades("r1")
        assert len(df) == 2
        assert set(df["symbol"].tolist()) == {"BTCUSDT", "ETHUSDT"}

    def test_insert_trades_empty(self):
        self.db.start_run("r1")
        self.db.insert_trades("r1", [])  # should not raise

    def test_weekly_metrics(self):
        self.db.start_run("r1")
        metric = {
            "week": 1, "week_start": "2025-01-06", "week_end": "2025-01-13",
            "n_symbols": 200, "n_trades": 60, "week_return_pct": 0.45,
            "annualised_pct": 26.0, "ic": 0.03, "turnover_pct": 30.0,
            "on_track": True, "cum_return_pct": 0.45,
            "max_drawdown_pct": -0.1, "regime": "BULL_TREND",
        }
        self.db.insert_weekly_metric("r1", metric)
        df = self.db.get_weekly_metrics("r1")
        assert len(df) == 1
        assert df.iloc[0]["regime"] == "BULL_TREND"

    def test_shap_values(self):
        self.db.start_run("r1")
        shap = {"ret_1d": 0.08, "rsi_14": 0.05, "vol_ratio": 0.03}
        self.db.insert_shap_values("r1", "base", shap)
        df = self.db.get_shap_values("r1", "base")
        assert len(df) == 3
        assert df.iloc[0]["feature"] == "ret_1d"  # highest SHAP first

    def test_model_artifact(self):
        self.db.start_run("r1")
        self.db.insert_model_artifact(
            "r1", "base_y1", 0, "/tmp/model.json", "/tmp/scaler.pkl",
            auc=0.55, ic=0.03, icir=0.5, n_features=56
        )
        # No get method — just verify no error
        # Could query directly if needed

    def test_feature_ic(self):
        self.db.start_run("r1")
        ics = {"ret_1d": 0.04, "rsi_14": -0.01, "vol_ratio": 0.02}
        self.db.insert_feature_ic("r1", 4, ics, ["ret_1d", "vol_ratio"])
        df = self.db.get_feature_ic("r1")
        assert len(df) == 3
        selected = df[df["selected"] == 1]
        assert len(selected) == 2

    def test_performance_summary(self):
        self.db.start_run("r1")
        perf = {
            "total_weeks": 52, "total_trades": 5000, "retrains": 4,
            "total_return_pct": 12.5, "annualised_pct": 12.5,
            "sharpe": 0.8, "max_drawdown_pct": -8.0,
            "ic_mean": 0.025, "icir": 0.6, "ic_positive_pct": 58.0,
            "var_95": -2.1, "cvar_95": -3.4, "kill_switch_hit": False,
        }
        self.db.insert_performance_summary("r1", perf)
        result = self.db.get_performance_summary("r1")
        assert result is not None
        assert result["sharpe"] == 0.8

    def test_compare_runs(self):
        for rid in ["r1", "r2"]:
            self.db.start_run(rid)
            self.db.insert_performance_summary(rid, {
                "total_weeks": 52, "total_trades": 1000, "retrains": 4,
                "total_return_pct": 10.0 if rid == "r1" else 15.0,
                "annualised_pct": 10.0, "sharpe": 0.5,
                "max_drawdown_pct": -5.0, "ic_mean": 0.02,
                "icir": 0.4, "ic_positive_pct": 55.0,
                "var_95": -2.0, "cvar_95": -3.0, "kill_switch_hit": False,
            })
        df = self.db.compare_runs(["r1", "r2"])
        assert len(df) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Feature Engineering (azalyst_factors_v2.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_ohlcv(n_bars=2000):
    """Synthetic 5-min OHLCV data."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=n_bars, freq="5min", tz="UTC")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.002, n_bars)))
    high = close * (1 + rng.uniform(0, 0.005, n_bars))
    low = close * (1 - rng.uniform(0, 0.005, n_bars))
    open_ = close * (1 + rng.normal(0, 0.001, n_bars))
    volume = rng.exponential(1000, n_bars)
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    }, index=dates)


class TestFeatureEngineering:
    def test_build_features_shape(self):
        from azalyst_factors_v2 import build_features, FEATURE_COLS
        df = _make_ohlcv(2000)
        result = build_features(df, timeframe="5min")
        # all expected columns present
        for col in FEATURE_COLS:
            assert col in result.columns, f"Missing feature: {col}"

    def test_build_features_no_nan_explosion(self):
        from azalyst_factors_v2 import build_features, FEATURE_COLS
        df = _make_ohlcv(2000)
        result = build_features(df, timeframe="5min")
        # After warm-up period, most features should be non-NaN
        tail = result.iloc[500:]
        nan_pct = tail[FEATURE_COLS].isna().mean().mean()
        assert nan_pct < 0.3, f"Too many NaNs: {nan_pct*100:.1f}%"

    def test_feature_cols_count(self):
        from azalyst_factors_v2 import FEATURE_COLS
        assert len(FEATURE_COLS) == 56

    def test_frac_diff(self):
        from azalyst_factors_v2 import frac_diff_ffd
        rng = np.random.default_rng(42)
        prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.001, 2000))))
        result = frac_diff_ffd(prices, d=0.4, threshold=1e-5)
        assert len(result) == len(prices)
        # frac_diff may produce many NaNs for short series;
        # just verify it returns the right length and doesn't crash
        assert isinstance(result, pd.Series)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Risk Module (azalyst_risk.py)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRiskManager:
    @pytest.fixture(autouse=True)
    def setup_risk(self):
        from azalyst_risk import RiskManager
        self.rm = RiskManager()

    def test_var_negative(self):
        returns = pd.Series(np.random.default_rng(42).normal(0.001, 0.02, 100))
        var = self.rm.calculate_var(returns, 0.95)
        assert var < 0  # VaR should be negative (loss)

    def test_cvar_worse_than_var(self):
        returns = pd.Series(np.random.default_rng(42).normal(0.001, 0.02, 100))
        var = self.rm.calculate_var(returns, 0.95)
        cvar = self.rm.calculate_cvar(returns, 0.95)
        assert cvar <= var  # CVaR (expected shortfall) is worse than VaR

    def test_apply_constraints(self):
        weights = np.array([0.5, 0.3, 0.15, 0.05])
        constrained = self.rm.apply_constraints(weights, max_weight=0.2)
        # Verify it returns an array that sums close to 1
        assert abs(constrained.sum() - 1.0) < 0.05
        # Verify the largest weight was reduced
        assert constrained[0] < weights[0]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. V4 Engine Components (azalyst_v4_engine.py)
# ═══════════════════════════════════════════════════════════════════════════════

class TestV4EngineComponents:
    def test_compute_drawdown_empty(self):
        from azalyst_v4_engine import compute_drawdown
        assert compute_drawdown([]) == 0.0

    def test_compute_drawdown_all_positive(self):
        from azalyst_v4_engine import compute_drawdown
        dd = compute_drawdown([0.01, 0.02, 0.01, 0.03])
        assert dd == 0.0  # no drawdown if always going up

    def test_compute_drawdown_loss(self):
        from azalyst_v4_engine import compute_drawdown
        dd = compute_drawdown([0.10, -0.20, 0.05, -0.10])
        assert dd < 0
        assert dd > -1.0

    def test_select_features_by_ic_no_history(self):
        from azalyst_v4_engine import select_features_by_ic
        features = ["a", "b", "c", "d"]
        result = select_features_by_ic({}, features)
        assert result == features  # no history -> keep all

    def test_select_features_by_ic_drop_negative(self):
        from azalyst_v4_engine import select_features_by_ic, MIN_FEATURES
        features = [f"f{i}" for i in range(30)]
        history = {}
        for i, f in enumerate(features):
            if i < 10:
                history[f] = [-0.05, -0.04, -0.06, -0.03, -0.05]  # negative IC
            else:
                history[f] = [0.03, 0.02, 0.04, 0.03, 0.02]  # positive IC
        result = select_features_by_ic(history, features)
        assert len(result) >= MIN_FEATURES
        assert len(result) == 20  # the 20 positive ones kept

    def test_purged_cv_splits(self):
        from azalyst_v4_engine import PurgedTimeSeriesCV
        cv = PurgedTimeSeriesCV(n_splits=5, gap=48)
        X = np.random.rand(10000, 10)
        splits = list(cv.split(X))
        assert len(splits) >= 4  # may lose last fold due to gap
        for train_idx, val_idx in splits:
            # gap enforced
            assert val_idx[0] - train_idx[-1] >= 48

    def test_detect_regime_fallback(self):
        from azalyst_v4_engine import detect_regime
        regime = detect_regime({}, pd.Timestamp("2025-01-01", tz="UTC"))
        assert regime == "LOW_VOL_GRIND"

    def test_make_xgb_params_cpu(self):
        from azalyst_v4_engine import make_xgb_params
        p = make_xgb_params(None)
        assert "device" not in p
        assert "tree_method" not in p
        assert p["n_estimators"] == 1000

    def test_make_xgb_params_new_cuda(self):
        from azalyst_v4_engine import make_xgb_params
        p = make_xgb_params("new")
        assert p["device"] == "cuda"

    def test_make_xgb_params_old_cuda(self):
        from azalyst_v4_engine import make_xgb_params
        p = make_xgb_params("old")
        assert p["tree_method"] == "gpu_hist"

    def test_simulate_weekly_trades_empty(self):
        from azalyst_v4_engine import simulate_weekly_trades
        trades, ret, longs, shorts = simulate_weekly_trades(
            {}, {}, set(), set()
        )
        assert trades == []
        assert ret == 0.0

    def test_compute_feature_ic_few_symbols(self):
        from azalyst_v4_engine import compute_feature_ic
        # Fewer than 20 symbols -> returns zeros
        result = compute_feature_ic({}, pd.Timestamp("2025-01-01"),
                                     pd.Timestamp("2025-01-08"), ["ret_1d"])
        assert result == {"ret_1d": 0.0}

    def test_fix_timestamp(self):
        from azalyst_v4_engine import _fix_timestamp
        df = pd.DataFrame({"close": [100, 101]},
                          index=pd.to_datetime(["2024-01-01", "2024-01-02"],
                                               utc=True))
        result = _fix_timestamp(df)
        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.tz is not None

    def test_get_date_splits(self):
        from azalyst_v4_engine import get_date_splits
        # Create synthetic symbols dict with 3-year date range
        dates = pd.date_range("2022-06-01", "2025-06-01", freq="5min", tz="UTC")
        sym = {"TEST": pd.DataFrame({"close": np.ones(len(dates))}, index=dates)}
        gmin, gmax, y1_end, y2_end = get_date_splits(sym)
        # Y1 end should be ~1 year in
        assert y1_end > gmin
        assert y2_end > y1_end
        assert y2_end < gmax


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Signal Combiner (azalyst_signal_combiner.py)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSignalCombiner:
    def test_regime_weight_table_exists(self):
        from azalyst_signal_combiner import REGIME_WEIGHT_TABLE
        assert "BULL_TREND" in REGIME_WEIGHT_TABLE
        assert "BEAR_TREND" in REGIME_WEIGHT_TABLE
        assert "HIGH_VOL_LATERAL" in REGIME_WEIGHT_TABLE
        assert "LOW_VOL_GRIND" in REGIME_WEIGHT_TABLE

    def test_weights_sum_to_one(self):
        from azalyst_signal_combiner import REGIME_WEIGHT_TABLE
        for regime, weights in REGIME_WEIGHT_TABLE.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.01, f"{regime} weights sum to {total}"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Integration: DB + Engine roundtrip
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    def test_db_roundtrip_performance(self, tmp_path):
        from azalyst_db import AzalystDB
        db = AzalystDB(str(tmp_path / "integration.db"))
        rid = db.start_run("integration_test", {"version": "v4"})

        # Insert trades
        trades = [
            {"week": w, "symbol": f"SYM{i}USDT", "signal": "BUY",
             "pred_prob": 0.5 + np.random.rand() * 0.2,
             "pnl_percent": (np.random.rand() - 0.5) * 2, "meta_size": 0.8}
            for w in range(1, 5) for i in range(10)
        ]
        db.insert_trades(rid, trades)

        # Insert weekly metrics
        for w in range(1, 5):
            db.insert_weekly_metric(rid, {
                "week": w, "n_trades": 10,
                "week_return_pct": np.random.uniform(-1, 1),
                "ic": np.random.uniform(-0.05, 0.05),
            })

        # Insert SHAP
        db.insert_shap_values(rid, "test", {"ret_1d": 0.05, "rsi_14": 0.03})

        # Insert performance summary
        db.insert_performance_summary(rid, {
            "total_weeks": 4, "total_trades": 40, "retrains": 0,
            "total_return_pct": 2.0, "annualised_pct": 26.0,
            "sharpe": 1.2, "max_drawdown_pct": -3.0,
            "ic_mean": 0.02, "icir": 0.5, "ic_positive_pct": 60.0,
            "var_95": -1.5, "cvar_95": -2.0, "kill_switch_hit": False,
        })
        db.finish_run(rid)

        # Verify full roundtrip
        perf = db.get_performance_summary(rid)
        assert perf["total_trades"] == 40
        assert perf["sharpe"] == 1.2

        all_trades = db.get_trades(rid)
        assert len(all_trades) == 40

        metrics = db.get_weekly_metrics(rid)
        assert len(metrics) == 4

        shap_df = db.get_shap_values(rid, "test")
        assert len(shap_df) == 2

        runs = db.list_runs()
        assert len(runs) == 1
        assert runs.iloc[0]["status"] == "completed"

        db.close()
