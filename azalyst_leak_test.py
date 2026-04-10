"""
Leakage smoke test for pre-training checks.
"""

import numpy as np
from scipy.stats import spearmanr


def run_leak_test(X: np.ndarray, y_ret: np.ndarray, feature_names: list,
                  embargo_bars: int = 48) -> dict:
    results = {}

    y_shuffled = np.random.permutation(y_ret)
    ics = []
    for j in range(min(20, X.shape[1])):
        mask = np.isfinite(X[:, j]) & np.isfinite(y_shuffled)
        if mask.sum() > 100:
            ic, _ = spearmanr(X[mask, j], y_shuffled[mask])
            if np.isfinite(ic):
                ics.append(abs(ic))
    results["shuffled_mean_abs_ic"] = float(np.mean(ics)) if ics else 0.0
    results["shuffled_test_pass"] = results["shuffled_mean_abs_ic"] < 0.02

    leaked = np.roll(y_ret, -1)
    leaked[-1] = np.nan
    mask = np.isfinite(leaked) & np.isfinite(y_ret)
    ic_leaked, _ = spearmanr(leaked[mask], y_ret[mask])
    results["leaked_feature_ic"] = float(ic_leaked)
    results["leaked_test_pass"] = ic_leaked > 0.95

    past = np.roll(y_ret, embargo_bars + 1)
    past[:embargo_bars + 1] = np.nan
    mask = np.isfinite(past) & np.isfinite(y_ret)
    ic_past, _ = spearmanr(past[mask], y_ret[mask])
    results["past_feature_ic"] = float(ic_past)

    return results


if __name__ == "__main__":
    np.random.seed(42)
    n = 10000
    X = np.random.randn(n, 10)
    y = np.random.randn(n) * 0.01
    print(run_leak_test(X, y, [f"f{i}" for i in range(10)]))

