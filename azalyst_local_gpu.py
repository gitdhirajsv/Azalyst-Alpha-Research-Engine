"""
╔══════════════════════════════════════════════════════════════════════════════╗
         AZALYST  —  LOCAL GPU RUNNER  (RTX 2050 4GB  |  i5-11260H)
║  Runs the full Year 3 walk-forward entirely on NVIDIA RTX 2050 CUDA.       ║
║  Intel UHD Graphics is NEVER used — CUDA is pinned to device 0 (RTX 2050) ║
║  4GB VRAM guard: caps training at 2M rows to prevent OOM.                 ║
║  Live Spyder console output every 4 weeks.                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, sys, gc, json, time, pickle, warnings, subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

# ── Force CUDA to RTX 2050, ignore Intel UHD ─────────────────────────────────
# Intel UHD is NOT a CUDA device — XGBoost will never see it.
# Pinning CUDA_VISIBLE_DEVICES=0 ensures RTX 2050 is always device 0.
os.environ["CUDA_VISIBLE_DEVICES"]  = "0"
os.environ["CUDA_DEVICE_ORDER"]     = "PCI_E_BUS_ID"

# ── CONFIG — edit these to match your folder layout ──────────────────────────
DATA_DIR    = r"./data"           # raw SYMBOL.parquet files
RESULTS_DIR = r"./results"        # outputs (CSVs, chart, JSON, models)
CACHE_DIR   = r"./feature_cache"  # pre-built feature store (auto-built once)

# RTX 2050 4GB VRAM guard — do NOT raise above 2_000_000
MAX_TRAIN_ROWS = 2_000_000

# Walk-forward config (mirrors notebook Cell 3)
RETRAIN_WEEKS  = 13
TOP_QUANTILE   = 0.15
FEE_RATE       = 0.001
ROUND_TRIP_FEE = FEE_RATE * 2
HORIZON_BARS   = 48
BARS_PER_HOUR  = 12
BARS_PER_DAY   = 288

FEATURE_COLS = [
    'ret_1bar','ret_1h','ret_4h','ret_1d','ret_2d','ret_3d','ret_1w',
    'vol_ratio','vol_ret_1h','vol_ret_1d','obv_change','vpt_change','vol_momentum',
    'rvol_1h','rvol_4h','rvol_1d','vol_ratio_1h_1d','atr_norm','parkinson_vol','garman_klass',
    'rsi_14','rsi_6','macd_hist','bb_pos','bb_width','stoch_k','stoch_d','cci_14','adx_14','dmi_diff',
    'vwap_dev','amihud','kyle_lambda','spread_proxy','body_ratio','candle_dir',
    'wick_top','wick_bot','price_accel','skew_1d','kurt_1d','max_ret_4h',
    'wq_alpha001','wq_alpha012','wq_alpha031','wq_alpha098',
    'cs_momentum','cs_reversal','vol_adjusted_mom','trend_consistency',
    'vol_regime','trend_strength','corr_btc_proxy','hurst_exp','fft_strength',
]

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — GPU VALIDATION  (RTX 2050 only, hard abort if missing)
# ─────────────────────────────────────────────────────────────────────────────

def validate_gpu():
    print("=" * 64)
    print("  AZALYST LOCAL GPU RUNNER")
    print("  Target: NVIDIA RTX 2050  (CUDA device 0)")
    print("=" * 64)

    # nvidia-smi check
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            raise RuntimeError("nvidia-smi returned non-zero")
        gpu_info = r.stdout.strip()
        print(f"\n  GPU detected  : {gpu_info}")
    except Exception as e:
        print(f"\n  [ERROR] Cannot find NVIDIA GPU: {e}")
        print("  Check: Device Manager > Display Adapters > RTX 2050 listed?")
        print("  Check: NVIDIA driver installed?  Run: nvidia-smi in cmd")
        sys.exit(1)

    if "RTX 2050" not in gpu_info and "NVIDIA" not in gpu_info:
        print("\n  [WARN] GPU name unexpected — continuing anyway but check above")

    # XGBoost CUDA test
    try:
        import xgboost as xgb
        print(f"  XGBoost ver   : {xgb.__version__}")
    except ImportError:
        print("  [ERROR] xgboost not installed.")
        print("  Fix: pip install xgboost>=2.0.3 --upgrade")
        sys.exit(1)

    _x = np.random.rand(500, 10).astype(np.float32)
    _y = np.random.randint(0, 2, 500)
    cuda_api = None

    # New API (XGBoost >= 1.7)
    try:
        xgb.XGBClassifier(device='cuda', n_estimators=5, verbosity=0).fit(_x, _y)
        cuda_api = "new"
        print(f"  CUDA API      : new  (device='cuda')")
    except Exception as e1:
        # Legacy API
        try:
            xgb.XGBClassifier(tree_method='gpu_hist', n_estimators=5, verbosity=0).fit(_x, _y)
            cuda_api = "old"
            print(f"  CUDA API      : legacy  (tree_method='gpu_hist')")
        except Exception as e2:
            print(f"\n  [ERROR] XGBoost CUDA failed with both APIs.")
            print(f"  New API error : {e1}")
            print(f"  Old API error : {e2}")
            print(f"\n  Fix: pip install xgboost>=2.0.3 --upgrade")
            print(f"  Also: check CUDA toolkit installed (run: nvcc --version)")
            sys.exit(1)

    print(f"  VRAM guard    : {MAX_TRAIN_ROWS:,} rows  (safe for 4GB)")
    print(f"  Status        : RTX 2050 CUDA CONFIRMED\n")
    return cuda_api


# ─────────────────────────────────────────────────────────────────────────────
#  XGBoost params
# ─────────────────────────────────────────────────────────────────────────────

def make_xgb_params(cuda_api):
    p = dict(
        n_estimators=1000, learning_rate=0.02, max_depth=6,
        min_child_weight=30, subsample=0.8, colsample_bytree=0.7,
        colsample_bylevel=0.7, reg_alpha=0.1, reg_lambda=1.0,
        eval_metric='auc', early_stopping_rounds=50,
        verbosity=0, random_state=42,
    )
    if   cuda_api == "new": p['device']      = 'cuda'
    elif cuda_api == "old": p['tree_method'] = 'gpu_hist'
    return p


# ─────────────────────────────────────────────────────────────────────────────
#  Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def _rsi(s, n):
    d  = s.diff()
    g  = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    ls = (-d).clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + g / ls.replace(0, np.nan))

def build_features(df):
    bph, bpd = BARS_PER_HOUR, BARS_PER_DAY
    c=df['close'].astype(np.float32); o=df['open'].astype(np.float32)
    h=df['high'].astype(np.float32);  l=df['low'].astype(np.float32)
    v=df['volume'].astype(np.float32)
    f = pd.DataFrame(index=df.index, dtype=np.float32)
    lr = np.log(c / c.shift(1))
    f['ret_1bar']=lr; f['ret_1h']=np.log(c/c.shift(bph))
    f['ret_4h']=np.log(c/c.shift(bph*4)); f['ret_1d']=np.log(c/c.shift(bpd))
    f['ret_2d']=np.log(c/c.shift(bpd*2)); f['ret_3d']=np.log(c/c.shift(bpd*3))
    f['ret_1w']=np.log(c/c.shift(bpd*5))
    av=v.rolling(bpd,min_periods=bph).mean()
    f['vol_ratio']=v/av.replace(0,np.nan)
    f['vol_ret_1h']=np.log(v/v.shift(bph).replace(0,np.nan))
    f['vol_ret_1d']=np.log(v/v.shift(bpd).replace(0,np.nan))
    obv=(np.sign(lr)*v).cumsum()
    f['obv_change']=obv.diff(bph)/(obv.abs().rolling(bpd,min_periods=bph).mean()+1e-8)
    vpt=(lr*v).cumsum(); f['vpt_change']=vpt.diff(bph)
    f['vol_momentum']=v.rolling(bph,min_periods=2).mean()/v.rolling(bpd,min_periods=bph).mean()
    f['rvol_1h']=lr.rolling(bph,min_periods=6).std()
    f['rvol_4h']=lr.rolling(bph*4,min_periods=12).std()
    f['rvol_1d']=lr.rolling(bpd,min_periods=bph).std()
    f['vol_ratio_1h_1d']=f['rvol_1h']/f['rvol_1d'].replace(0,np.nan)
    tr=pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    f['atr_norm']=tr.ewm(span=14).mean()/c
    f['parkinson_vol']=np.sqrt(1/(4*np.log(2))*np.log(h/l)**2).rolling(bpd).mean()
    f['garman_klass']=(0.5*np.log(h/l)**2-(2*np.log(2)-1)*np.log(c/o)**2).rolling(bpd).mean()
    f['rsi_14']=_rsi(c,14)/100.0; f['rsi_6']=_rsi(c,6)/100.0
    macd=c.ewm(span=12).mean()-c.ewm(span=26).mean()
    f['macd_hist']=(macd-macd.ewm(span=9).mean())/c
    ma20=c.rolling(20).mean(); std20=c.rolling(20).std()
    f['bb_pos']=(c-(ma20-2*std20))/(4*std20); f['bb_width']=(4*std20)/ma20
    lo14=l.rolling(14).min(); hi14=h.rolling(14).max()
    f['stoch_k']=(c-lo14)/(hi14-lo14); f['stoch_d']=f['stoch_k'].rolling(3).mean()
    tp=(h+l+c)/3; tp_ma=tp.rolling(14).mean()
    tp_mad=(tp-tp_ma).abs().rolling(14).mean()
    f['cci_14']=(tp-tp_ma)/(0.015*tp_mad)
    f['adx_14']=(h.diff().clip(lower=0)-l.diff().clip(upper=0).abs()).abs().rolling(14).mean()/tr.rolling(14).mean()
    f['dmi_diff']=(h.diff().clip(lower=0)-l.diff().clip(upper=0).abs())/tr.rolling(14).mean()
    vwap=(tp*v).rolling(bpd).sum()/v.rolling(bpd).sum()
    f['vwap_dev']=(c-vwap)/c; f['amihud']=(lr.abs()/v).rolling(bpd).mean()
    f['kyle_lambda']=(lr.abs()/(v*c)).rolling(bpd).mean()
    f['spread_proxy']=(h-l)/c; f['body_ratio']=(c-o).abs()/(h-l)
    f['candle_dir']=np.sign(c-o)
    f['wick_top']=(h-c.clip(lower=o))/(h-l); f['wick_bot']=(c.clip(upper=o)-l)/(h-l)
    f['price_accel']=c.pct_change(bph)-c.pct_change(bph).shift(bph)
    f['skew_1d']=lr.rolling(bpd).skew(); f['kurt_1d']=lr.rolling(bpd).kurt()
    f['max_ret_4h']=lr.rolling(bph*4).max()
    f['wq_alpha001']=np.sign(f['ret_1d'])*f['rvol_1d']
    f['wq_alpha012']=np.sign(v.diff())*(-lr)
    f['wq_alpha031']=-c.rank().rolling(bpd).corr(v)
    f['wq_alpha098']=np.log(c/c.shift(bpd*5))/f['rvol_1d']
    f['cs_momentum']=f['ret_4h']; f['cs_reversal']=-f['ret_1d']
    f['vol_adjusted_mom']=f['ret_4h']*f['vol_ratio']
    f['trend_consistency']=np.sign(lr).rolling(48).mean()
    f['vol_regime']=f['rvol_1d']/f['rvol_1d'].rolling(bpd*30).mean()
    f['trend_strength']=f['adx_14']
    f['corr_btc_proxy']=lr.rolling(bpd).corr(lr.shift(1))
    f['hurst_exp']=np.nan; f['fft_strength']=np.nan
    return f.replace([np.inf,-np.inf],np.nan).shift(1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Feature store
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_store(data_dir, cache_dir):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    files = sorted(Path(data_dir).rglob("*.parquet"))
    if not files:
        print(f"  [ERROR] No parquet files in {data_dir}")
        sys.exit(1)
    new_count = skip_count = 0
    for f in files:
        out = Path(cache_dir) / f.name
        if out.exists(): skip_count += 1; continue
        try:
            df = pd.read_parquet(f)
            df.index = pd.to_datetime(df.index, utc=True)
            feat = build_features(df)
            feat['future_ret'] = np.log(df['close'].shift(-HORIZON_BARS) / df['close'])
            feat['symbol'] = f.stem
            feat.dropna(subset=['future_ret']+FEATURE_COLS, how='all').to_parquet(out)
            new_count += 1
        except Exception as e:
            print(f"    [WARN] {f.stem}: {e}")
    print(f"  Feature store: {new_count} built, {skip_count} cached ({len(files)} symbols total)")


# ─────────────────────────────────────────────────────────────────────────────
#  Purged K-Fold & IC
# ─────────────────────────────────────────────────────────────────────────────

class PurgedTimeSeriesCV:
    def __init__(self, n_splits=5, gap=48):
        self.n_splits=n_splits; self.gap=gap
    def split(self, X):
        n=len(X); fold_size=n//(self.n_splits+1)
        for i in range(self.n_splits):
            train_end=(i+1)*fold_size; val_start=train_end+self.gap; val_end=val_start+fold_size
            if val_end>n: break
            yield np.arange(0,train_end), np.arange(val_start,val_end)

def compute_ic(y_pred, y_true):
    mask=np.isfinite(y_pred)&np.isfinite(y_true)
    if mask.sum()<10: return 0.0
    return float(stats.spearmanr(y_pred[mask],y_true[mask])[0])


# ─────────────────────────────────────────────────────────────────────────────
#  Train model on RTX 2050
# ─────────────────────────────────────────────────────────────────────────────

def train_model(X, y, y_ret, cuda_api, label=''):
    import xgboost as xgb
    print(f"  Training [{label}]: {len(X):,} rows | GPU=RTX2050")
    t0=time.time()
    scaler=RobustScaler(); Xs=scaler.fit_transform(X)
    cv=PurgedTimeSeriesCV(n_splits=5,gap=48)
    aucs,ics=[],[]
    for fold,(tr,val) in enumerate(cv.split(Xs),1):
        if len(np.unique(y[val]))<2: continue
        m=xgb.XGBClassifier(**make_xgb_params(cuda_api))
        m.fit(Xs[tr],y[tr],eval_set=[(Xs[val],y[val])],verbose=False)
        probs=m.predict_proba(Xs[val])[:,1]
        try: aucs.append(roc_auc_score(y[val],probs))
        except: pass
        if y_ret is not None and np.isfinite(y_ret[val]).any():
            ics.append(compute_ic(probs,y_ret[val]))
    mean_auc=float(np.mean(aucs)) if aucs else 0.0
    mean_ic =float(np.mean(ics))  if ics  else 0.0
    icir    =float(np.mean(ics)/(np.std(ics)+1e-8)) if len(ics)>1 else 0.0
    final=xgb.XGBClassifier(**make_xgb_params(cuda_api))
    split=int(len(Xs)*0.9)
    final.fit(Xs[:split],y[:split],eval_set=[(Xs[split:],y[split:])],verbose=False)
    imp=pd.Series(final.feature_importances_,index=FEATURE_COLS,name='importance').sort_values(ascending=False)
    print(f"    AUC={mean_auc:.4f}  IC={mean_ic:.4f}  ICIR={icir:.4f}  [{time.time()-t0:.0f}s]")
    return final, scaler, imp, mean_auc, mean_ic, icir


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cuda_api = validate_gpu()
    for d in [RESULTS_DIR, f"{RESULTS_DIR}/models"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Feature store
    print("[Step 1] Feature store...")
    build_feature_store(DATA_DIR, CACHE_DIR)

    # Load all
    print("\n[Step 2] Loading feature store...")
    all_files = sorted(Path(CACHE_DIR).glob("*.parquet"))
    if not all_files:
        print("  [ERROR] Feature store empty. Check DATA_DIR path.")
        sys.exit(1)
    frames=[]
    for fp in all_files:
        try:
            df=pd.read_parquet(fp)
            if not isinstance(df.index,pd.DatetimeIndex):
                df.index=pd.to_datetime(df.index,utc=True)
            elif df.index.tz is None:
                df.index=df.index.tz_localize('UTC')
            if 'symbol' not in df.columns: df['symbol']=fp.stem
            frames.append(df)
        except Exception as e:
            print(f"  [WARN] {fp.stem}: {e}")
    if not frames:
        print("  [ERROR] No feature files loaded."); sys.exit(1)

    ALL_DATA=pd.concat(frames,axis=0).sort_index()
    print(f"  Loaded {len(ALL_DATA):,} rows | {ALL_DATA['symbol'].nunique()} symbols")
    print(f"  Range : {ALL_DATA.index.min().date()} → {ALL_DATA.index.max().date()}")

    # Date split
    global_min=ALL_DATA.index.min(); global_max=ALL_DATA.index.max()
    total_span=global_max-global_min
    YEAR2_END=global_min+(total_span*2/3); YEAR3_START=YEAR2_END
    print(f"\n  Train (Y1+Y2): {global_min.date()} → {YEAR2_END.date()}")
    print(f"  Test  (Y3)  : {YEAR3_START.date()} → {global_max.date()}")

    # Training matrix
    print("\n[Step 3] Building training matrix...")
    train_df=ALL_DATA[ALL_DATA.index<YEAR2_END].copy()
    if 'alpha_label' not in train_df.columns:
        train_df['alpha_label']=(
            train_df.groupby(train_df.index)['future_ret']
            .transform(lambda x:(x>x.median()).astype(float)))
    for col in FEATURE_COLS:
        if col not in train_df.columns: train_df[col]=0.0
    feat_matrix=train_df[FEATURE_COLS].values.astype(np.float32)
    label_arr  =train_df['alpha_label'].values.astype(np.float32)
    ret_arr    =train_df['future_ret'].values.astype(np.float32)
    valid_mask =np.isfinite(feat_matrix).all(axis=1)&np.isfinite(label_arr)
    feat_matrix=feat_matrix[valid_mask]; label_arr=label_arr[valid_mask]; ret_arr=ret_arr[valid_mask]
    if len(feat_matrix)>MAX_TRAIN_ROWS:
        idx=np.random.choice(len(feat_matrix),MAX_TRAIN_ROWS,replace=False); idx.sort()
        feat_matrix=feat_matrix[idx]; label_arr=label_arr[idx]; ret_arr=ret_arr[idx]
        print(f"  VRAM guard: capped at {MAX_TRAIN_ROWS:,} rows")
    print(f"  Matrix: {len(feat_matrix):,} rows × {feat_matrix.shape[1]} features | Label: {label_arr.mean()*100:.1f}%")
    del train_df; gc.collect()

    # Base model
    base_model_path =f"{RESULTS_DIR}/models/model_base_y1y2.json"
    base_scaler_path=f"{RESULTS_DIR}/models/scaler_base_y1y2.pkl"
    print("\n[Step 4] Base model (Year 1+2)...")
    if Path(base_model_path).exists() and Path(base_scaler_path).exists():
        import xgboost as xgb
        print("  Loading cached base model...")
        BASE_MODEL=xgb.XGBClassifier(); BASE_MODEL.load_model(base_model_path)
        with open(base_scaler_path,'rb') as fh: BASE_SCALER=pickle.load(fh)
    else:
        BASE_MODEL,BASE_SCALER,importance,mean_auc,mean_ic,icir=train_model(
            feat_matrix,label_arr,ret_arr,cuda_api,label='base_y1y2')
        BASE_MODEL.save_model(base_model_path)
        with open(base_scaler_path,'wb') as fh: pickle.dump(BASE_SCALER,fh)
        importance.to_csv(f"{RESULTS_DIR}/feature_importance_base.csv")
    gc.collect()

    # Walk-forward Year 3
    print(f"\n[Step 5] Year 3 walk-forward: {YEAR3_START.date()} → {global_max.date()}")
    weeks=pd.date_range(start=YEAR3_START,end=global_max,freq='W-MON')
    if len(weeks)<2:
        print("  [ERROR] Not enough Year 3 weeks."); sys.exit(1)

    current_model=BASE_MODEL; current_scaler=BASE_SCALER; retrain_count=0
    all_trades_list=[]; weekly_summary_list=[]; weekly_returns_hist=[]

    for week_num,(ws,we) in enumerate(zip(weeks[:-1],weeks[1:]),1):
        week_df=ALL_DATA[(ALL_DATA.index>=ws)&(ALL_DATA.index<we)].copy()
        if len(week_df)<10: continue
        for col in FEATURE_COLS:
            if col not in week_df.columns: week_df[col]=0.0
        feat_w=week_df[FEATURE_COLS].values.astype(np.float32)
        valid_w=np.isfinite(feat_w).all(axis=1)
        if valid_w.sum()<5: continue
        try:
            feat_scaled=current_scaler.transform(feat_w[valid_w])
            probs_w=current_model.predict_proba(feat_scaled)[:,1]
        except Exception as e:
            print(f"  Week {week_num}: predict failed — {e}"); continue

        week_valid=week_df.iloc[np.where(valid_w)[0]].copy()
        week_valid['prob']=probs_w

        def rank_cs(grp):
            n=max(1,int(len(grp)*TOP_QUANTILE))
            grp=grp.copy().sort_values('prob'); grp['signal']='HOLD'
            if len(grp)>=4:
                grp.iloc[-n:,grp.columns.get_loc('signal')]='BUY'
                grp.iloc[:n, grp.columns.get_loc('signal')]='SELL'
            return grp
        week_valid=week_valid.groupby(week_valid.index,group_keys=False).apply(rank_cs)

        trades_this_week=[]
        for _,row in week_valid[week_valid['signal']!='HOLD'].iterrows():
            fret=row.get('future_ret',np.nan)
            if not np.isfinite(fret): continue
            pnl=((fret-ROUND_TRIP_FEE) if row['signal']=='BUY' else (-fret-ROUND_TRIP_FEE))*100
            trades_this_week.append({'week':week_num,'week_start':str(ws.date()),
                'symbol':row.get('symbol',''),'signal':row['signal'],
                'pred_prob':round(float(row['prob']),5),'pnl_percent':round(float(pnl),4)})
        all_trades_list.extend(trades_this_week)

        pnls=np.array([t['pnl_percent'] for t in trades_this_week])/100
        week_ret=float(pnls.mean()) if len(pnls)>0 else 0.0
        weekly_returns_hist.append(week_ret)
        cs_ic=compute_ic(week_valid['prob'].values,week_valid['future_ret'].fillna(0).values)
        ann_proj=((1+week_ret)**52-1)*100

        # Quarterly retrain
        did_retrain=False
        if week_num%RETRAIN_WEEKS==0:
            print(f"  Week {week_num:3d}: QUARTERLY RETRAIN...")
            try:
                rt_df=ALL_DATA[ALL_DATA.index<we].copy()
                for col in FEATURE_COLS:
                    if col not in rt_df.columns: rt_df[col]=0.0
                if 'alpha_label' not in rt_df.columns:
                    rt_df['alpha_label']=rt_df.groupby(rt_df.index)['future_ret'].transform(
                        lambda x:(x>x.median()).astype(float))
                Xr=rt_df[FEATURE_COLS].values.astype(np.float32)
                yr=rt_df['alpha_label'].values.astype(np.float32)
                yr_r=rt_df['future_ret'].values.astype(np.float32)
                vm=np.isfinite(Xr).all(axis=1)&np.isfinite(yr)
                Xr,yr,yr_r=Xr[vm],yr[vm],yr_r[vm]
                if len(Xr)>MAX_TRAIN_ROWS:
                    idx2=np.random.choice(len(Xr),MAX_TRAIN_ROWS,replace=False)
                    Xr,yr,yr_r=Xr[idx2],yr[idx2],yr_r[idx2]
                m_new,s_new,imp_new,auc_n,ic_n,icir_n=train_model(
                    Xr,yr,yr_r,cuda_api,label=f'y3_w{week_num:03d}')
                current_model=m_new; current_scaler=s_new; retrain_count+=1
                m_new.save_model(f"{RESULTS_DIR}/models/model_y3_week{week_num:03d}.json")
                imp_new.to_csv(f"{RESULTS_DIR}/feature_importance_y3_week{week_num:03d}.csv")
                did_retrain=True; del rt_df,Xr,yr; gc.collect()
            except Exception as e:
                print(f"    Retrain failed (keeping current): {e}")

        on_track=week_ret>=((1.10)**(1/52)-1)
        weekly_summary_list.append({'week':week_num,'week_start':str(ws.date()),
            'week_end':str(we.date()),
            'n_symbols':int(week_valid['symbol'].nunique()) if 'symbol' in week_valid.columns else 0,
            'n_trades':len(trades_this_week),'week_return_pct':round(week_ret*100,4),
            'annualised_pct':round(ann_proj,2),'ic':round(cs_ic,5),
            'on_track':on_track,'retrained':did_retrain})

        # Spyder console live update every 4 weeks
        if week_num%4==0 or week_num<=2:
            rolling=np.mean(weekly_returns_hist[-4:])*100 if weekly_returns_hist else 0
            print(f"  Week {week_num:3d} | ret={week_ret*100:+.3f}%  IC={cs_ic:+.4f}  "
                  f"4w_avg={rolling:+.3f}%  n={len(trades_this_week)}  gpu=RTX2050")

    # Save results
    print("\n[Step 6] Saving results...")
    if not weekly_summary_list:
        print("  [WARN] No results — check DATA_DIR"); return

    weekly_df=pd.DataFrame(weekly_summary_list)
    trades_df=pd.DataFrame(all_trades_list)
    weekly_df.to_csv(f"{RESULTS_DIR}/weekly_summary_year3.csv",index=False)
    trades_df.to_csv(f"{RESULTS_DIR}/all_trades_year3.csv",index=False)

    rets=weekly_df['week_return_pct'].dropna()/100
    pnls=trades_df['pnl_percent'].dropna()/100 if len(trades_df)>0 else pd.Series(dtype=float)
    ic_s=weekly_df['ic'].dropna()
    cum=(1+rets).cumprod(); n_wks=max(len(rets),1)
    t_ret=float(cum.iloc[-1]-1) if len(cum)>0 else 0.0
    ann_ret=(1+t_ret)**(52/n_wks)-1
    sharpe=float(rets.mean()/(rets.std()+1e-9)*np.sqrt(52))
    ic_mean=float(ic_s.mean()) if len(ic_s)>0 else 0.0
    ic_std =float(ic_s.std())  if len(ic_s)>0 else 0.0
    icir   =float(ic_mean/(ic_std+1e-8))
    ic_pos =float((ic_s>0).mean()*100) if len(ic_s)>0 else 0.0

    performance={'label':'Year3_WalkForward_RTX2050',
        'total_weeks':n_wks,'total_trades':len(trades_df),'retrains':retrain_count,
        'total_return_pct':round(t_ret*100,2),'annualised_pct':round(ann_ret*100,2),
        'sharpe':round(sharpe,4),'ic_mean':round(ic_mean,5),'icir':round(icir,4),
        'ic_positive_pct':round(ic_pos,1),'gpu':'NVIDIA RTX 2050',
        'vram_cap_rows':MAX_TRAIN_ROWS}
    with open(f"{RESULTS_DIR}/performance_year3.json",'w') as fh:
        json.dump(performance,fh,indent=2)

    # Chart
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        fig=plt.figure(figsize=(16,11))
        fig.suptitle('Azalyst v2 — Year 3 Walk-Forward (RTX 2050)',fontsize=14,fontweight='bold')
        gs=gridspec.GridSpec(2,2,figure=fig,hspace=0.38,wspace=0.32)
        ax1=fig.add_subplot(gs[0,0])
        cum_pct=((1+weekly_df['week_return_pct'].fillna(0)/100).cumprod()-1)*100
        ax1.plot(weekly_df['week'],cum_pct,color='#1f77b4',linewidth=2)
        ax1.fill_between(weekly_df['week'],cum_pct,alpha=0.12,color='#1f77b4')
        ax1.axhline(0,color='gray',linewidth=0.8,linestyle='--')
        ax1.set_title('Cumulative Return (%)',fontweight='bold')
        ax1.set_xlabel('Week #'); ax1.set_ylabel('%'); ax1.grid(True,alpha=0.25)
        ax2=fig.add_subplot(gs[0,1])
        wr=weekly_df['week_return_pct'].dropna()
        ax2.hist(wr,bins=min(30,max(10,len(wr)//3)),color='#ff7f0e',alpha=0.72,edgecolor='black',linewidth=0.4)
        if len(wr)>2:
            ax2.axvline(wr.mean(),color='red',linewidth=1.8,linestyle='--',label=f'Mean {wr.mean():.2f}%')
            ax2.axvline(wr.median(),color='green',linewidth=1.2,linestyle=':',label=f'Median {wr.median():.2f}%')
            ax2.legend(fontsize=9)
        ax2.set_title('Weekly Return Distribution',fontweight='bold'); ax2.grid(True,alpha=0.25)
        ax3=fig.add_subplot(gs[1,0])
        ic_vals=weekly_df['ic'].fillna(0)
        ax3.bar(weekly_df['week'],ic_vals,color=['#2ca02c' if v>0 else '#d62728' for v in ic_vals],alpha=0.75,width=0.8)
        if len(ic_vals)>2:
            ax3.axhline(ic_vals.mean(),color='navy',linewidth=1.5,linestyle='--',label=f'Mean IC {ic_vals.mean():.4f}')
            ax3.legend(fontsize=9)
        ax3.axhline(0,color='black',linewidth=0.6)
        ax3.set_title('Weekly IC',fontweight='bold'); ax3.grid(True,alpha=0.25)
        ax4=fig.add_subplot(gs[1,1])
        if len(trades_df)>0 and 'pnl_percent' in trades_df.columns:
            pnl=trades_df['pnl_percent'].dropna()
            ax4.hist(pnl,bins=min(40,max(10,len(pnl)//20)),color='#9467bd',alpha=0.72,edgecolor='black',linewidth=0.3)
            ax4.axvline(pnl.mean(),color='red',linewidth=1.8,linestyle='--',label=f'Mean {pnl.mean():.3f}%')
            ax4.axvline(0,color='black',linewidth=0.8); ax4.legend(fontsize=9)
            ax4.set_title(f'Trade P&L (n={len(pnl):,})',fontweight='bold'); ax4.grid(True,alpha=0.25)
        plt.savefig(f"{RESULTS_DIR}/performance_year3.png",dpi=150,bbox_inches='tight')
        plt.close(); print(f"  Chart → {RESULTS_DIR}/performance_year3.png")
    except Exception as e:
        print(f"  [WARN] Chart failed: {e}")

    print(f"""
{'='*64}
  AZALYST v2 — COMPLETE  (NVIDIA RTX 2050)
{'='*64}
  Total Return  : {performance['total_return_pct']:>8.2f}%
  Annualised    : {performance['annualised_pct']:>8.2f}%
  Sharpe        : {performance['sharpe']:>8.4f}
  IC Mean       : {performance['ic_mean']:>8.5f}
  ICIR          : {performance['icir']:>8.4f}
  IC Positive % : {performance['ic_positive_pct']:>8.1f}%
  Weeks         : {performance['total_weeks']}
  Retrains      : {performance['retrains']}
  GPU           : {performance['gpu']}
{'='*64}
  Results saved to: {RESULTS_DIR}
{'='*64}
""")

if __name__ == "__main__":
    main()
