"""
Deflated Sharpe Ratio — Lopez de Prado.
"""

import numpy as np
from scipy.stats import norm


def deflated_sharpe_ratio(sharpe_observed: float,
                          n_returns: int,
                          skew: float,
                          kurtosis: float,
                          n_trials: int) -> dict:
    emc = 0.5772156649
    e_max_sr = (
        np.sqrt(2 * np.log(n_trials))
        - (emc + np.log(np.log(n_trials))) / (2 * np.sqrt(2 * np.log(n_trials)))
    ) if n_trials > 1 else 0.0

    sr_var = (
        1
        - skew * sharpe_observed
        + ((kurtosis - 1) / 4) * sharpe_observed ** 2
    ) / max(n_returns - 1, 1)
    sr_std = np.sqrt(max(sr_var, 1e-10))
    dsr = norm.cdf((sharpe_observed - e_max_sr) / sr_std)

    return {
        "sharpe_observed": sharpe_observed,
        "expected_max_sharpe_null": float(e_max_sr),
        "deflated_sharpe_ratio": float(dsr),
        "p_value": float(1 - dsr),
        "significant_at_95": bool(dsr > 0.95),
        "n_returns": int(n_returns),
        "n_trials": int(n_trials),
    }

