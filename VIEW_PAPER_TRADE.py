"""Quick status view for the paper-trade runner."""
import json
from pathlib import Path
import pandas as pd

STATE = Path(__file__).parent / "paper_trade_state"
POS = STATE / "positions.json"
EQ = STATE / "equity_curve.csv"
TRADES = STATE / "trade_log.csv"


def main():
    if not POS.exists():
        print("No state yet. Run RUN_PAPER_TRADE.bat first.")
        return

    with open(POS, "r", encoding="utf-8") as f:
        state = json.load(f)

    print("=" * 70)
    print("AZALYST v7 — PAPER TRADE STATUS")
    print("=" * 70)
    print(f"  realised equity   : ${state['equity']:,.2f}")
    print(f"  peak equity       : ${state['peak_equity']:,.2f}")
    print(f"  paused            : {state['paused']}")
    print(f"  last rebalance    : {state.get('last_rebalance', 'never')}")
    print(f"  last regime       : {state.get('last_regime', '-')}")
    print(f"  open positions    : {len(state.get('positions', {}))}")
    print()

    pos = state.get("positions", {})
    if pos:
        rows = []
        for sym, p in pos.items():
            rows.append({
                "symbol": sym,
                "side": p["side"],
                "entry": p["entry_price"],
                "mark": p.get("mark_price", "-"),
                "size_usd": p["size_usd"],
                "scale": p["scale"],
                "unreal_pnl": p.get("unrealised_pnl", "-"),
                "opened_at": p["opened_at"],
            })
        df = pd.DataFrame(rows).sort_values(["side", "unreal_pnl"],
                                            ascending=[True, False],
                                            na_position="last")
        print("OPEN POSITIONS")
        print("-" * 70)
        print(df.to_string(index=False))
        print()

    if EQ.exists():
        eq = pd.read_csv(EQ)
        print("EQUITY CURVE (last 10 rows)")
        print("-" * 70)
        print(eq.tail(10).to_string(index=False))
        print()

    if TRADES.exists():
        tr = pd.read_csv(TRADES)
        print(f"TRADE LOG — {len(tr)} events total, last 15:")
        print("-" * 70)
        print(tr.tail(15).to_string(index=False))


if __name__ == "__main__":
    main()
