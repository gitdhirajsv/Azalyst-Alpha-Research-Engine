"""
Azalyst — Spyder Live Monitor (RTX 2050 local run)
Run this in Spyder (F5) while azalyst_local_gpu.py runs in a terminal.
Auto-refreshes every 5 seconds with live 4-panel chart.
"""
import os, time, json, warnings
import pandas as pd, numpy as np
import matplotlib, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore")

RESULTS_DIR  = r"./results"
REFRESH_SECS = 5

def load_data():
    w, t, p = pd.DataFrame(), pd.DataFrame(), {}
    try: w = pd.read_csv(os.path.join(RESULTS_DIR,"weekly_summary_year3.csv"))
    except: pass
    try: t = pd.read_csv(os.path.join(RESULTS_DIR,"all_trades_year3.csv"))
    except: pass
    try:
        with open(os.path.join(RESULTS_DIR,"performance_year3.json")) as f: p=json.load(f)
    except: pass
    return w, t, p

def draw(fig, axes, w, t, p):
    for row in axes:
        for ax in row: ax.clear()
    nw = len(w)

    ax1 = axes[0][0]
    if nw > 0:
        rets = w['week_return_pct'].fillna(0)/100
        cum  = ((1+rets).cumprod()-1)*100
        ax1.plot(w['week'], cum, '#1f77b4', linewidth=2)
        ax1.fill_between(w['week'], cum, alpha=0.12, color='#1f77b4')
        ax1.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        ax1.set_title(f"Cumulative Return | Total={p.get('total_return_pct',cum.iloc[-1] if len(cum)>0 else 0):.1f}%  Ann={p.get('annualised_pct',0):.1f}%", fontweight='bold', fontsize=9)
        ax1.set_xlabel('Week #'); ax1.set_ylabel('%'); ax1.grid(True,alpha=0.25)
    else:
        ax1.text(0.5,0.5,'Waiting for data...',ha='center',va='center',color='gray',fontsize=12)
        ax1.set_title('Cumulative Return',fontweight='bold')

    ax2 = axes[0][1]
    if nw > 0:
        ic = w['ic'].fillna(0)
        ax2.bar(w['week'],ic,color=['#2ca02c' if v>0 else '#d62728' for v in ic],alpha=0.75,width=0.8)
        if len(ic)>2:
            ax2.axhline(ic.mean(),color='navy',linewidth=1.5,linestyle='--',label=f"Mean IC={ic.mean():.4f}")
            ax2.legend(fontsize=8)
        ax2.axhline(0,color='black',linewidth=0.6)
        ax2.set_title(f"Weekly IC | ICIR={p.get('icir',0):.4f}",fontweight='bold',fontsize=9)
        ax2.set_xlabel('Week #'); ax2.set_ylabel('Spearman IC'); ax2.grid(True,alpha=0.25)
    else:
        ax2.text(0.5,0.5,'Waiting...',ha='center',va='center',color='gray',fontsize=12)
        ax2.set_title('Weekly IC',fontweight='bold')

    ax3 = axes[1][0]
    if nw > 2:
        wr = w['week_return_pct'].dropna()
        ax3.hist(wr,bins=min(25,max(8,len(wr)//3)),color='#ff7f0e',alpha=0.72,edgecolor='black',linewidth=0.4)
        ax3.axvline(wr.mean(),color='red',linewidth=1.8,linestyle='--',label=f'Mean {wr.mean():.2f}%')
        ax3.axvline(wr.median(),color='green',linewidth=1.2,linestyle=':',label=f'Median {wr.median():.2f}%')
        ax3.set_title(f"Weekly Returns | Sharpe={p.get('sharpe',0):.3f}",fontweight='bold',fontsize=9)
        ax3.set_xlabel('Return (%)'); ax3.set_ylabel('Count'); ax3.legend(fontsize=8); ax3.grid(True,alpha=0.25)
    else:
        ax3.text(0.5,0.5,'Need >2 weeks...',ha='center',va='center',color='gray',fontsize=12)
        ax3.set_title('Weekly Return Distribution',fontweight='bold')

    ax4 = axes[1][1]; ax4.axis('off')
    if p:
        lines = [f"GPU           : {p.get('gpu','RTX 2050')}",
                 f"Weeks run     : {p.get('total_weeks',nw)}",
                 f"Total trades  : {p.get('total_trades',len(t)):,}",
                 f"Retrains done : {p.get('retrains',0)}",
                 "",
                 f"Total Return  : {p.get('total_return_pct',0):+.2f}%",
                 f"Annualised    : {p.get('annualised_pct',0):+.2f}%",
                 f"Sharpe        : {p.get('sharpe',0):.4f}",
                 f"IC Mean       : {p.get('ic_mean',0):.5f}",
                 f"ICIR          : {p.get('icir',0):.4f}",
                 f"IC Positive % : {p.get('ic_positive_pct',0):.1f}%"]
    elif nw > 0:
        lines = [f"GPU           : RTX 2050 (running)",
                 f"Weeks so far  : {nw}",f"Trades so far : {len(t):,}","",
                 f"Latest week   : {w['week_return_pct'].iloc[-1]:+.3f}%",
                 f"Latest IC     : {w['ic'].iloc[-1]:+.4f}",
                 f"4w avg return : {w['week_return_pct'].tail(4).mean():+.3f}%",
                 "","Pipeline running..."]
    else:
        lines = ["Pipeline starting...","","Data will appear here","once week 1 completes.","",f"Watching: {RESULTS_DIR}"]
    ax4.text(0.05,0.95,'\n'.join(lines),transform=ax4.transAxes,va='top',ha='left',
             family='monospace',fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5',facecolor='#f0f4f8',alpha=0.8))
    ax4.set_title('Live Status',fontweight='bold',fontsize=9)
    fig.suptitle(f'Azalyst v2 — Live Monitor | RTX 2050 | {time.strftime("%H:%M:%S")}',fontsize=13,fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.96])

print("="*60)
print("  Azalyst Spyder Live Monitor")
print(f"  Watching: {RESULTS_DIR}")
print(f"  Refresh : every {REFRESH_SECS}s")
print("  Press the Stop button in Spyder to end")
print("="*60)

plt.ion()
fig, axes = plt.subplots(2,2,figsize=(15,9))
try: fig.canvas.manager.set_window_title("Azalyst RTX 2050 — Live Monitor")
except: pass

for i in range(9999):
    try:
        w, t, p = load_data()
        draw(fig, axes, w, t, p)
        fig.canvas.draw(); fig.canvas.flush_events()
        plt.pause(REFRESH_SECS)
    except KeyboardInterrupt:
        print("\n  Monitor stopped."); break
    except Exception as e:
        print(f"  [WARN] {e}"); time.sleep(REFRESH_SECS)
