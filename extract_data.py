import pandas as pd
import json
import os

def extract_data():
    data = {}
    
    # Root directory
    root = "D:/Azalyst Alpha Research Engine"
    output_dir = os.path.join(root, "azalyst_output")
    
    # 1. Performance Metrics
    try:
        perf_df = pd.read_csv(os.path.join(root, "performance_metrics.csv"))
        data['performance'] = perf_df.to_dict(orient='records')
    except Exception as e:
        data['performance_error'] = str(e)
        
    # 2. Latest Trades
    try:
        trades_df = pd.read_csv(os.path.join(root, "paper_trades.csv"))
        # Get last 20
        data['latest_trades'] = trades_df.tail(20).to_dict(orient='records')
        # Regime distribution
        data['regime_distribution'] = trades_df['regime'].value_counts().to_dict()
    except Exception as e:
        data['trades_error'] = str(e)
        
    # 3. Factor Analysis
    try:
        ic_df = pd.read_csv(os.path.join(output_dir, "ic_analysis.csv"))
        # Sort by ICIR absolute value and get top 15
        ic_df['abs_ICIR'] = ic_df['ICIR'].abs()
        top_factors = ic_df.sort_values('abs_ICIR', ascending=False).head(15)
        data['top_factors'] = top_factors.drop(columns=['abs_ICIR']).to_dict(orient='records')
    except Exception as e:
        data['ic_error'] = str(e)
        
    # Save to JSON
    with open(os.path.join(root, "dashboard_data.json"), 'w') as f:
        json.dump(data, f, indent=4)
    print("Data extraction complete: dashboard_data.json")

if __name__ == "__main__":
    extract_data()
