# Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    AZALYST ALPHA RESEARCH ENGINE                 │
│                                                                  │
│  ┌──────────────┐    ┌──────────────────┐    ┌───────────────┐   │
│  │  DATA LAYER  │───▶│  FACTOR ENGINE v2│───▶│  VALIDATION   │   │
│  │  Polars +    │    │  35 factors      │    │  Style Neut.  │   │
│  │  DuckDB      │    │  Cross-section   │    │  Fama-MacBeth │   │
│  └──────────────┘    └──────────────────┘    └───────────────┘   │
│          │                                          │            │
│          ▼                                          ▼            │
│  ┌──────────────┐    ┌──────────────────┐    ┌───────────────┐   │
│  │   STATARB    │    │   REGIME DETECT  │    │   ML SCORING  │   │
│  │ Cointegration│    │   GMM + Breadth  │    │   LGBM + CUDA │   │
│  │ Pairs Engine │    │   BTC Microstr.  │    │   Pump/Return │   │
│  └──────────────┘    └──────────────────┘    └───────────────┘   │
│          │                    │                       │          │
│          └────────────────────┴───────────────────────┘          │
│                               ▼                                  │
│                   ┌───────────────────────┐                      │
│                   │    SIGNAL COMBINER    │                      │
│                   │   Regime-adaptive     │                      │
│                   │   weighted fusion     │                      │
│                   └───────────────────────┘                      │
│                               ▼                                  │
│                   ┌───────────────────────┐                      │
│                   │      signals.csv      │                      │
│                   │   Ranked per symbol   │                      │
│                   └───────────────────────┘                      │
└──────────────────────────────────────────────────────────────────┘
```

Above, each stage reads as a separate subsystem:

- **Data Layer:** Parallel Polars ingestion plus DuckDB queries build the wide panels (5m → 1h if requested) that feed every downstream module.
- **Factor Engine v2:** Computes 35 cross-sectional signals and ranks them with Spearman IC, ICIR, and decay metrics.
- **Validation:** Style neutralization (BTC beta, sectors, liquidity) and Fama-MacBeth plus BH correction cull overfitting tails.
- **StatArb:** Stores cointegration pairs, half-lives, and live z-scores for mean-reversion signals.
- **Regime Detector:** Trains a four-component Gaussian Mixture Model on BTC 4h data to classify Bull/Bear/High-Vol/Quiet markets.
- **ML Scoring:** GPU-accelerated LightGBM learns PumpDumpDetector, ReturnPredictor, and regime-aware forward-return models.
- **Signal Combiner:** Fuses all signals with regime-aware weights before writing `signals.csv` (ranked per coin).
