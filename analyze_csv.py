import pandas as pd

df = pd.read_csv('results/weekly_summary_v4.csv')
print(f'Total rows: {len(df)}')
print(f'Week range: {df["week"].min()} to {df["week"].max()}')
print(f'Total return: {df["week_return_pct"].sum():.4f}%')
print(f'Kill-switch weeks: {(df["regime"] == "KILL_SWITCH").sum()}')
print(f'Regimes: {df["regime"].unique()}')
print(f'\nRegime distribution:')
print(df['regime'].value_counts())
