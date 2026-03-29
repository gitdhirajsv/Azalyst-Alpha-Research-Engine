"""
╔══════════════════════════════════════════════════════════════════════╗
  AZALYST  —  TRAINING RESULTS DASHBOARD
  Double-click VIEW_TRAINING.bat to launch this automatically.
  Shows: metrics summary · feature importance · SHAP · weekly IC
         weekly returns · cumulative PnL · drawdown
╚══════════════════════════════════════════════════════════════════════╝
Run from project root or via VIEW_TRAINING.bat
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")          # interactive window (works in Spyder + bare Python)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent
RES     = ROOT / "results"
MODELS  = RES / "models"

FI_FILES = {
    "v4 Base":    RES / "feature_importance_v4_base.csv",
    "v4 W013":    RES / "feature_importance_v4_week013.csv",
    "Y3 W013":    RES / "feature_importance_y3_week013.csv",
    "Y3 W026":    RES / "feature_importance_y3_week026.csv",
    "Y3 W039":    RES / "feature_importance_y3_week039.csv",
}
SHAP_FILES = {
    "v4 Base":  MODELS / "shap" / "shap_importance_v4_base.csv",
    "v4 W013":  MODELS / "shap" / "shap_importance_v4_week013.csv",
}
WEEKLY_CSV   = RES / "weekly_summary_year3.csv"
PNL_CSV      = RES / "backtest_pnl.csv"
PERF_CSV     = RES / "performance_summary.csv"
TRAIN_V4     = RES / "train_summary_v4.json"
TRAIN_BASE   = RES / "train_summary.json"
PERF_Y3      = RES / "performance_year3.json"
IC_CSV       = RES / "ic_analysis.csv"


# ── Helpers ──────────────────────────────────────────────────────────────────
def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _load_csv(path: Path, **kw) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, **kw)
    except Exception:
        return None

def _best_fi() -> tuple[str, pd.DataFrame] | tuple[None, None]:
    """Return the most recent feature-importance file that exists."""
    for label in ["Y3 W039", "Y3 W026", "Y3 W013", "v4 W013", "v4 Base"]:
        df = _load_csv(FI_FILES[label], index_col=0)
        if df is not None and not df.empty:
            return label, df
    return None, None

def _best_shap() -> tuple[str, pd.DataFrame] | tuple[None, None]:
    for label in ["v4 W013", "v4 Base"]:
        df = _load_csv(SHAP_FILES[label])
        if df is not None and not df.empty:
            return label, df
    return None, None


# ── Theme ────────────────────────────────────────────────────────────────────
BG      = "#0e0e12"
PANEL   = "#1a1a24"
ACC1    = "#f07f2a"   # orange  (matches screenshot)
ACC2    = "#4fa8d5"   # blue
ACC3    = "#4caf50"   # green
ACC4    = "#e05252"   # red
TXT     = "#e8e8e8"
GRID    = "#2a2a38"

def _style():
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    PANEL,
        "axes.edgecolor":    GRID,
        "axes.labelcolor":   TXT,
        "axes.titlecolor":   TXT,
        "xtick.color":       TXT,
        "ytick.color":       TXT,
        "grid.color":        GRID,
        "grid.linewidth":    0.6,
        "text.color":        TXT,
        "font.family":       "monospace",
        "font.size":         9,
        "axes.titlesize":    10,
        "axes.titleweight":  "bold",
    })


# ── Build dashboard ──────────────────────────────────────────────────────────
def build_dashboard():
    _style()

    # ── load data ────────────────────────────────────────────────────────────
    tv4      = _load_json(TRAIN_V4)
    tbase    = _load_json(TRAIN_BASE)
    py3      = _load_json(PERF_Y3)
    weekly   = _load_csv(WEEKLY_CSV)
    pnl_df   = _load_csv(PNL_CSV, parse_dates=["timestamp"])
    perf_df  = _load_csv(PERF_CSV)
    fi_label, fi_df   = _best_fi()
    shap_label, shap_df = _best_shap()

    # ── figure layout  (3 rows × 3 cols) ─────────────────────────────────────
    fig = plt.figure(figsize=(18, 13), facecolor=BG)
    fig.suptitle(
        "AZALYST  —  ML TRAINING RESULTS DASHBOARD",
        fontsize=14, fontweight="bold", color=ACC1, y=0.98
    )

    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        left=0.05, right=0.97,
        top=0.93,  bottom=0.06,
        hspace=0.45, wspace=0.38
    )

    # ── Panel 0 (top-left): Training Metrics Summary ──────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_axis_off()

    def _val(d, k, fmt=".5f"):
        v = d.get(k)
        if v is None:
            return "n/a"
        try:
            return f"{float(v):{fmt}}"
        except Exception:
            return str(v)

    rows = [
        ("── TRAIN SUMMARY (v4) ──", "", False),
        ("AUC (CV mean)",   _val(tv4, "mean_auc"),              True),
        ("IC  (CV mean)",   _val(tv4, "mean_ic"),               True),
        ("ICIR",            _val(tv4, "icir"),                  True),
        ("Features used",   str(tv4.get("n_features", "n/a")),  True),
        ("Train rows",      f'{tv4.get("n_rows", 0):,}' if tv4.get("n_rows") else "n/a", True),
        ("GPU",             "CUDA ✓" if tv4.get("use_gpu") else "CPU", True),
        ("", "", False),
        ("── YEAR 3 PERF ──",       "", False),
        ("Total return %",  _val(py3, "total_return_pct", ".2f"), True),
        ("Ann. return %",   _val(py3, "annualised_pct",   ".2f"), True),
        ("Sharpe",          _val(py3, "sharpe",           ".3f"), True),
        ("IC mean",         _val(py3, "ic_mean",          ".5f"), True),
        ("ICIR",            _val(py3, "icir",             ".4f"), True),
        ("Total trades",    str(py3.get("total_trades", "n/a")),  True),
        ("Retrains",        str(py3.get("retrains",      "n/a")),  True),
    ]
    if perf_df is not None and not perf_df.empty:
        r = perf_df.iloc[0]
        rows += [
            ("", "", False),
            ("── BACKTEST OVERALL ──", "", False),
            ("Total ret %",   f'{float(r.get("total_ret_%", 0)):.2f}', True),
            ("Sharpe",        f'{float(r.get("sharpe", 0)):.3f}',       True),
            ("Max DD %",      f'{float(r.get("max_dd_%", 0)):.2f}',     True),
            ("Win rate %",    f'{float(r.get("win_rate_%", 0)):.1f}',   True),
        ]

    y_pos = 0.99
    step  = 0.063
    for label, value, is_data in rows:
        if not is_data:
            ax0.text(0.02, y_pos, label, transform=ax0.transAxes,
                     fontsize=8, color=ACC1, fontweight="bold", va="top")
        else:
            ax0.text(0.04, y_pos, label, transform=ax0.transAxes,
                     fontsize=8.5, color=TXT, va="top")
            col = ACC3 if value not in ("n/a", "CPU") else ACC4
            ax0.text(0.72, y_pos, value, transform=ax0.transAxes,
                     fontsize=8.5, color=col, va="top", ha="right")
        y_pos -= step

    ax0.set_title("Training & Performance Metrics", pad=6)
    ax0.set_facecolor(PANEL)
    for sp in ax0.spines.values():
        sp.set_edgecolor(ACC1)
        sp.set_linewidth(1.2)

    # ── Panel 1 (top-mid): Feature Importance ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    if fi_df is not None:
        col = fi_df.columns[0]
        top = fi_df[col].sort_values(ascending=True).tail(18)
        colors = [ACC1 if i >= len(top) - 5 else ACC2 for i in range(len(top))]
        bars = ax1.barh(top.index, top.values, color=colors, height=0.7)
        ax1.set_title(f"Feature Importance  ({fi_label})", pad=6)
        ax1.set_xlabel("Importance")
        ax1.grid(axis="x", alpha=0.4)
        ax1.tick_params(axis="y", labelsize=7)
    else:
        ax1.text(0.5, 0.5, "No feature importance data", ha="center", va="center",
                 color=TXT, transform=ax1.transAxes)
        ax1.set_title("Feature Importance", pad=6)

    # ── Panel 2 (top-right): SHAP Importance ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    if shap_df is not None:
        shap_top = shap_df.nlargest(18, "mean_abs_shap")
        shap_top_s = shap_top.set_index("feature")["mean_abs_shap"].sort_values()
        colors2 = [ACC3 if i >= len(shap_top_s) - 5 else ACC2 for i in range(len(shap_top_s))]
        ax2.barh(shap_top_s.index, shap_top_s.values, color=colors2, height=0.7)
        ax2.set_title(f"SHAP Importance  ({shap_label})", pad=6)
        ax2.set_xlabel("|SHAP value|")
        ax2.grid(axis="x", alpha=0.4)
        ax2.tick_params(axis="y", labelsize=7)
    else:
        ax2.text(0.5, 0.5, "No SHAP data found", ha="center", va="center",
                 color=TXT, transform=ax2.transAxes)
        ax2.set_title("SHAP Importance", pad=6)

    # ── Panel 3 (mid-left): Weekly IC ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    if weekly is not None:
        ic = weekly["ic"]
        colors_ic = [ACC3 if v >= 0 else ACC4 for v in ic]
        ax3.bar(weekly["week"], ic, color=colors_ic, width=0.7)
        ax3.axhline(0, color=TXT, lw=0.8, ls="--")
        mean_ic = ic.mean()
        ax3.axhline(mean_ic, color=ACC1, lw=1.2, ls="--",
                    label=f"Mean IC {mean_ic:.4f}")
        ax3.set_title("Weekly Information Coefficient (IC)", pad=6)
        ax3.set_xlabel("Week")
        ax3.set_ylabel("IC")
        ax3.legend(fontsize=8, loc="upper right")
        ax3.grid(axis="y", alpha=0.4)
    else:
        ax3.text(0.5, 0.5, "No weekly summary data", ha="center", va="center",
                 color=TXT, transform=ax3.transAxes)
        ax3.set_title("Weekly IC", pad=6)

    # ── Panel 4 (mid-mid): Weekly Returns ────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    if weekly is not None:
        wr = weekly["week_return_pct"]
        colors_wr = [ACC3 if v >= 0 else ACC4 for v in wr]
        ax4.bar(weekly["week"], wr, color=colors_wr, width=0.7)
        ax4.axhline(0, color=TXT, lw=0.8, ls="--")
        cum = (1 + wr / 100).cumprod() - 1
        ax4b = ax4.twinx()
        ax4b.plot(weekly["week"], cum * 100, color=ACC2, lw=1.5,
                  label="Cum %")
        ax4b.set_ylabel("Cum Return %", color=ACC2, fontsize=8)
        ax4b.tick_params(axis="y", labelcolor=ACC2, labelsize=7)
        ax4.set_title("Weekly Returns & Cumulative PnL", pad=6)
        ax4.set_xlabel("Week")
        ax4.set_ylabel("Week Return %")
        ax4.grid(axis="y", alpha=0.4)
    else:
        ax4.text(0.5, 0.5, "No weekly data", ha="center", va="center",
                 color=TXT, transform=ax4.transAxes)
        ax4.set_title("Weekly Returns", pad=6)

    # ── Panel 5 (mid-right): Turnover ────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    if weekly is not None and "turnover_pct" in weekly.columns:
        ax5.plot(weekly["week"], weekly["turnover_pct"], color=ACC1, lw=1.5)
        ax5.fill_between(weekly["week"], weekly["turnover_pct"], alpha=0.25, color=ACC1)
        ax5.set_title("Turnover % per Week", pad=6)
        ax5.set_xlabel("Week")
        ax5.set_ylabel("Turnover %")
        ax5.grid(axis="y", alpha=0.4)
    else:
        ax5.text(0.5, 0.5, "No turnover data", ha="center", va="center",
                 color=TXT, transform=ax5.transAxes)
        ax5.set_title("Turnover %", pad=6)

    # ── Panel 6+7 (bottom span 2 cols): Backtest PnL curve ───────────────────
    ax6 = fig.add_subplot(gs[2, :2])
    if pnl_df is not None and "cum_ret" in pnl_df.columns:
        ts = pnl_df["timestamp"]
        cum  = pnl_df["cum_ret"] * 100
        bm   = pnl_df.get("cum_benchmark", pd.Series([0] * len(pnl_df))) * 100
        dd   = pnl_df.get("drawdown", pd.Series([0] * len(pnl_df))) * 100

        ax6.plot(ts, cum, color=ACC3, lw=1.5, label="Strategy")
        ax6.plot(ts, bm,  color=ACC2, lw=1,   ls="--", label="Benchmark", alpha=0.8)
        ax6.fill_between(ts, dd, 0, color=ACC4, alpha=0.3, label="Drawdown")
        ax6.axhline(0, color=TXT, lw=0.6, ls="--")
        ax6.set_title("Backtest Cumulative Return & Drawdown (Year 1–3)", pad=6)
        ax6.set_xlabel("Date")
        ax6.set_ylabel("Cumulative Return %")
        ax6.legend(fontsize=8, loc="upper left")
        ax6.grid(alpha=0.4)
        ax6.tick_params(axis="x", labelsize=7)
    else:
        ax6.text(0.5, 0.5, "No backtest PnL data found", ha="center", va="center",
                 color=TXT, transform=ax6.transAxes)
        ax6.set_title("Backtest PnL", pad=6)

    # ── Panel 7 (bottom-right): Model AUC / IC comparison ────────────────────
    ax7 = fig.add_subplot(gs[2, 2])

    model_labels = []
    model_aucs   = []
    model_ics    = []

    for src_label, src_json in [
        ("v4 Base", TRAIN_V4),
        ("Y3 Base", TRAIN_BASE),
    ]:
        d = _load_json(src_json)
        if d:
            model_labels.append(src_label)
            model_aucs.append(d.get("mean_auc", 0))
            model_ics.append(abs(d.get("mean_ic", 0)))

    if model_labels:
        x = np.arange(len(model_labels))
        w = 0.35
        ax7.bar(x - w/2, model_aucs, w, label="AUC",       color=ACC1)
        ax7.bar(x + w/2, model_ics,  w, label="|IC|",      color=ACC2)
        ax7.set_xticks(x)
        ax7.set_xticklabels(model_labels, fontsize=8)
        ax7.axhline(0.5, color=ACC4, lw=0.8, ls="--", label="AUC baseline 0.5")
        ax7.set_title("Model AUC & |IC| Comparison", pad=6)
        ax7.set_ylabel("Score")
        ax7.legend(fontsize=8)
        ax7.grid(axis="y", alpha=0.4)
    else:
        ax7.text(0.5, 0.5, "No model comparison data", ha="center", va="center",
                 color=TXT, transform=ax7.transAxes)
        ax7.set_title("Model Comparison", pad=6)

    # ── Footer ────────────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.005,
        "Azalyst Alpha Research Engine  |  VIEW_TRAINING.py  |  all data from results/",
        ha="center", fontsize=7.5, color="#666680"
    )

    plt.show()
    print("[VIEW_TRAINING] Dashboard open — close the window to exit.")


if __name__ == "__main__":
    print("Loading Azalyst Training Dashboard ...")
    build_dashboard()
