"""
Save V10 + V11 configs as equity curve CSVs for the dashboard.
Also saves SPY buy-and-hold for comparison.

V10 configs: original 13-instrument, trend+carry signals (backward compat)
V11 configs: expanded 33-instrument universe, 7 signal families

Run once: python scripts/save_v10_dashboard.py
Then launch: streamlit run dashboard/app.py
"""
import sys, os, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from src.data.loader import load_bars_from_csv, resample_bars, compute_returns
from src.data.norgate_loader import load_norgate_daily, load_all_norgate
from src.features.volatility import ewma_volatility
from src.execution.costs import CostModel
from src.engine import VectorizedBacktestEngine
from src.data.contracts import load_instrument_configs
from src.portfolio.risk_overlay import DrawdownLadder
from src.reporting.reports import ReportGenerator
from src.signals.base import vol_scale, weekly_rebal
from src.signals.trend import majority_vote_tsm
from src.signals.carry import carry_signal
from src.signals.mean_reversion import mean_reversion_signal
from src.signals.cross_sectional import cross_sectional_momentum
from src.signals.lead_lag import lead_lag_signal
from src.signals.vol_breakout import vol_breakout_signal
from src.signals.seasonality import seasonality_signal
from src.validation.regime_oos import RegimeConditionedOOS

# ============================================================
# ASSET CLASS DEFINITIONS
# ============================================================
ASSET_CLASSES = {
    'INDEX': ['ES', 'NQ', 'FDAX', 'NKD', 'LFT', 'HSI', 'YAP'],
    'DEBT':  ['ZN', 'ZB', 'ZF', 'FGBL'],
    'ENERGY': ['CL', 'NG', 'HO', 'RB'],
    'METALS': ['GC', 'SI', 'HG', 'PL', 'PA'],
    'AGS':   ['ZC', 'ZS', 'ZW', 'KC', 'CT', 'SB', 'CC'],
    'FX':    ['6E', '6J', '6B', '6A', '6C'],
}

# V10 universe (backward compat)
V10_DATA_MAP = {
    "ES":     "ES_full_1min_continuous_UNadjusted_13wjmr/ES_full_1min_continuous_UNadjusted.txt",
    "NQ":     "data_external/indices/nasdaq100/USATECHIDXUSD_D1.csv",
    "FDAX":   "data_external/indices/dax30/DEUIDXEUR_D1.csv",
    "GC":     "data_external/commodities/gold/XAUUSD_D1.csv",
    "CL":     "data_external/commodities/brent/BRENTCMDUSD_D1.csv",
    "SI":     "data_external/commodities/silver/XAGUSD_D1.csv",
    "6E":     "data_external/forex/eurusd/EURUSD_D1.csv",
    "6J":     "data_external/forex/usdjpy/USDJPY_D1.csv",
    "6B":     "data_external/forex/gbpusd/GBPUSD_D1.csv",
    "AUDUSD": "data_external/forex/audusd/AUDUSD_D1.csv",
    "USDCAD": "data_external/forex/usdcad/USDCAD_D1.csv",
    "EURJPY": "data_external/forex/eurjpy/EURJPY_D1.csv",
    "GBPJPY": "data_external/forex/gbpjpy/GBPJPY_D1.csv",
}
V10_EQUITY_SYMS = {'ES', 'NQ', 'FDAX'}

# V11 expanded universe: all 33 instruments from Norgate
V11_SYMBOLS = sorted(set(
    sym for syms in ASSET_CLASSES.values() for sym in syms
))
V11_EQUITY_SYMS = set(ASSET_CLASSES['INDEX'])

# ============================================================
# DATA LOADING
# ============================================================
print("Loading instruments config...")
instruments = load_instrument_configs(os.path.join(PROJECT_ROOT, "config", "instruments.yaml"))

# --- V10 data (legacy paths) ---
print("\nLoading V10 data (legacy paths)...")
v10_close = {}
v10_returns = {}
for sym, data_file in V10_DATA_MAP.items():
    path = os.path.join(PROJECT_ROOT, data_file)
    if not os.path.exists(path):
        continue
    try:
        raw = load_bars_from_csv(path)
        if sym == "ES":
            raw = resample_bars(raw, "1D")
        if hasattr(raw.index, 'dayofweek'):
            raw = raw[raw.index.dayofweek < 5]
        raw.index = raw.index.normalize()
        raw = raw[~raw.index.duplicated(keep='first')]
        v10_close[sym] = raw["close"]
        v10_returns[sym] = compute_returns(raw["close"], method="log")
        print(f"  {sym:>6}: {len(raw)} bars")
    except Exception as e:
        print(f"  SKIP {sym}: {e}")

v10_close_df = pd.DataFrame(v10_close)
v10_returns_df = pd.DataFrame(v10_returns)

# --- V11 data (Norgate) ---
print("\nLoading V11 data (Norgate)...")
v11_close, v11_returns = load_all_norgate(V11_SYMBOLS, project_root=PROJECT_ROOT)
v11_close_df = pd.DataFrame(v11_close)
v11_returns_df = pd.DataFrame(v11_returns)
print(f"  Loaded {len(v11_close)} instruments from Norgate")

# ============================================================
# V10 SIGNAL COMPUTATION (original code path, unchanged)
# ============================================================
print("\nComputing V10 signals...")
v10_trend_w = {}
v10_carry_w = {}
v10_blend_w = {}
for sym in v10_close:
    c, r = v10_close[sym], v10_returns[sym]
    mv = weekly_rebal(majority_vote_tsm(c))
    cs = weekly_rebal(carry_signal(c))
    v10_trend_w[sym] = vol_scale(mv, r)
    v10_carry_w[sym] = vol_scale(cs, r)
    v10_blend_w[sym] = (v10_trend_w[sym] + v10_carry_w[sym]) / 2

# V10 regime signals
print("Computing V10 regime signals...")
from src.features.regime import RegimeTagger
v10_eq_bear_probs = {}
for sym in V10_EQUITY_SYMS:
    if sym not in v10_close:
        continue
    try:
        tagger = RegimeTagger(vol_com=60, trend_windows=[21, 63, 252])
        regime = tagger.tag(v10_close[sym], v10_returns[sym])
        v10_eq_bear_probs[sym] = regime["bear_probability"]
    except Exception:
        pass
v10_portfolio_bear = pd.DataFrame(v10_eq_bear_probs).mean(axis=1) if v10_eq_bear_probs else pd.Series(0.5, index=v10_returns_df.index)

# ============================================================
# V11 SIGNAL COMPUTATION (all 7 signal families)
# ============================================================
print("\nComputing V11 signals (7 families)...")

# Signal 1: Trend (majority vote)
print("  Signal 1: Trend...")
v11_trend_w = {}
for sym in v11_close:
    c, r = v11_close[sym], v11_returns[sym]
    mv = weekly_rebal(majority_vote_tsm(c))
    v11_trend_w[sym] = vol_scale(mv, r)

# Signal 2: Carry
print("  Signal 2: Carry...")
v11_carry_w = {}
for sym in v11_close:
    c, r = v11_close[sym], v11_returns[sym]
    cs = weekly_rebal(carry_signal(c))
    v11_carry_w[sym] = vol_scale(cs, r)

# Signal 1+2 blend
v11_blend_w = {}
for sym in v11_close:
    v11_blend_w[sym] = (v11_trend_w[sym] + v11_carry_w[sym]) / 2

# Signal 3: Mean Reversion
print("  Signal 3: Mean Reversion...")
v11_meanrev_w = {}
for sym in v11_close:
    c, r = v11_close[sym], v11_returns[sym]
    mr = weekly_rebal(mean_reversion_signal(c), period=2)
    v11_meanrev_w[sym] = vol_scale(mr, r)

# Signal 4: Cross-Sectional Momentum
print("  Signal 4: Cross-Sectional Momentum...")
xsmom_raw = cross_sectional_momentum(v11_close, lookback=63, rebal_period=21)
v11_xsmom_w = {}
for sym in v11_close:
    if sym in xsmom_raw:
        v11_xsmom_w[sym] = vol_scale(xsmom_raw[sym], v11_returns[sym])
    else:
        v11_xsmom_w[sym] = pd.Series(0.0, index=v11_returns[sym].index)

# Signal 5: Lead-Lag (experimental)
print("  Signal 5: Lead-Lag...")
ll_raw = lead_lag_signal(v11_returns, asset_classes=ASSET_CLASSES)
v11_leadlag_w = {}
for sym in v11_close:
    if sym in ll_raw:
        v11_leadlag_w[sym] = vol_scale(weekly_rebal(ll_raw[sym], period=5), v11_returns[sym])
    else:
        v11_leadlag_w[sym] = pd.Series(0.0, index=v11_returns[sym].index)

# Signal 6: Vol Breakout v2
print("  Signal 6: Vol Breakout...")
v11_volbrk_w = {}
for sym in v11_close:
    c, r = v11_close[sym], v11_returns[sym]
    vb = weekly_rebal(vol_breakout_signal(c, r))
    v11_volbrk_w[sym] = vol_scale(vb, r)

# Signal 7: Seasonality
print("  Signal 7: Seasonality...")
v11_season_w = {}
for sym in v11_close:
    r = v11_returns[sym]
    ss = seasonality_signal(r, min_years=10)
    v11_season_w[sym] = vol_scale(weekly_rebal(ss, period=21), r)

# V11 regime signals
print("Computing V11 regime signals...")
v11_eq_bear_probs = {}
for sym in V11_EQUITY_SYMS:
    if sym not in v11_close:
        continue
    try:
        tagger = RegimeTagger(vol_com=60, trend_windows=[21, 63, 252])
        regime = tagger.tag(v11_close[sym], v11_returns[sym])
        v11_eq_bear_probs[sym] = regime["bear_probability"]
    except Exception:
        pass
v11_portfolio_bear = pd.DataFrame(v11_eq_bear_probs).mean(axis=1) if v11_eq_bear_probs else pd.Series(0.5, index=v11_returns_df.index)

# ============================================================
# PORTFOLIO BUILDERS
# ============================================================
ROLLING_WINDOW = 756
REBAL_FREQ = 63
cost_model = CostModel(slippage_base_bps=1.5, slippage_size_coefficient=5.0, impact_power=0.5)


def build_adaptive_portfolio(portfolio_vol_target=0.25, max_leverage=4.0,
                              use_regime=True, vol_com=21,
                              min_sharpe_threshold=0.0, use_carry_blend=True):
    """V10 portfolio builder (unchanged for backward compat)."""
    all_close = v10_close
    all_returns = v10_returns
    all_trend_w = v10_trend_w
    all_carry_w = v10_carry_w
    all_blend_w = v10_blend_w
    returns_df = v10_returns_df
    eq_bear_probs = v10_eq_bear_probs
    portfolio_bear = v10_portfolio_bear
    EQUITY_SYMS = V10_EQUITY_SYMS

    all_dates = returns_df.dropna(how='all').index.sort_values()
    trailing_sh = {}
    for sym in all_close:
        r = all_returns[sym]
        t_pnl = (all_trend_w[sym] * r)
        c_pnl = (all_carry_w[sym] * r)
        b_pnl = (all_blend_w[sym] * r)
        trailing_sh[sym] = {
            'trend': t_pnl.rolling(ROLLING_WINDOW, min_periods=252).apply(
                lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0, raw=True),
            'carry': c_pnl.rolling(ROLLING_WINDOW, min_periods=252).apply(
                lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0, raw=True),
            'blend': b_pnl.rolling(ROLLING_WINDOW, min_periods=252).apply(
                lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0, raw=True),
        }

    syms = list(all_close.keys())
    port_w = pd.DataFrame(0.0, index=all_dates, columns=syms)
    current_selections = {}
    current_budgets = {}
    current_universe = []
    last_rebal = None

    for i, date in enumerate(all_dates):
        need_rebal = (last_rebal is None or (i - last_rebal) >= REBAL_FREQ)
        if need_rebal and i >= ROLLING_WINDOW:
            new_selections = {}
            new_sharpes = {}
            for sym in all_close:
                if sym not in trailing_sh:
                    continue
                t_sh = trailing_sh[sym]['trend'].iloc[i] if i < len(trailing_sh[sym]['trend']) else 0
                c_sh = trailing_sh[sym]['carry'].iloc[i] if i < len(trailing_sh[sym]['carry']) else 0
                b_sh = trailing_sh[sym]['blend'].iloc[i] if i < len(trailing_sh[sym]['blend']) else 0
                if pd.isna(t_sh): t_sh = 0
                if pd.isna(c_sh): c_sh = 0
                if pd.isna(b_sh): b_sh = 0
                if use_carry_blend and t_sh > 0 and c_sh > 0 and b_sh > max(t_sh, c_sh) * 0.9:
                    best_type, best_sh = 'blend', b_sh
                elif t_sh >= c_sh:
                    best_type, best_sh = 'trend', t_sh
                else:
                    best_type, best_sh = 'carry', c_sh
                new_selections[sym] = best_type
                new_sharpes[sym] = best_sh

            current_universe = [s for s, sh in new_sharpes.items() if sh > min_sharpe_threshold]
            if len(current_universe) < 2:
                sorted_syms = sorted(new_sharpes.keys(), key=lambda s: new_sharpes[s], reverse=True)
                current_universe = sorted_syms[:max(2, len(sorted_syms) // 2)]

            current_selections = {s: new_selections[s] for s in current_universe}
            sh_vals = {s: max(new_sharpes[s], 0.01) for s in current_universe}
            total_sh = sum(sh_vals.values())
            current_budgets = {s: sh_vals[s] / total_sh for s in current_universe}
            last_rebal = i

        for sym in current_universe:
            if sym not in current_selections:
                continue
            sig_type = current_selections[sym]
            w_source = {'trend': all_trend_w, 'carry': all_carry_w, 'blend': all_blend_w}[sig_type]
            if date in w_source[sym].index:
                w_val = w_source[sym].loc[date]
                if not pd.isna(w_val):
                    if use_regime and sym in EQUITY_SYMS and sym in eq_bear_probs:
                        if date in eq_bear_probs[sym].index:
                            bp = eq_bear_probs[sym].loc[date]
                            if bp > 0.55:
                                w_val = w_val * (0.30 if w_val > 0 else 0.70)
                    if use_regime and date in portfolio_bear.index:
                        bp_port = portfolio_bear.loc[date]
                        if bp_port > 0.25:
                            bear_s = max(0.40, 1.0 - (bp_port - 0.25) / 0.50 * 0.60)
                            w_val *= bear_s
                    port_w.at[date, sym] = w_val * current_budgets.get(sym, 0)

    port_w = port_w.iloc[ROLLING_WINDOW:]
    r_aligned = returns_df.reindex(port_w.index).fillna(0)[syms]
    port_ret = (port_w * r_aligned).sum(axis=1)
    port_vol = ewma_volatility(port_ret, com=vol_com)
    scale = (portfolio_vol_target / port_vol.clip(lower=0.005)).clip(upper=max_leverage)
    port_w = port_w.multiply(scale, axis=0)
    port_w = port_w.ewm(halflife=5).mean()

    gross = port_w.abs().sum(axis=1)
    excess = gross > max_leverage
    if excess.any():
        port_w.loc[excess] = port_w.loc[excess].div(gross[excess] / max_leverage, axis=0)

    margin_util = pd.Series(0.0, index=port_w.index)
    for col in port_w.columns:
        if col in instruments:
            margin_util += port_w[col].abs() * instruments[col].margin_init_pct
    margin_excess = margin_util > 0.50
    if margin_excess.any():
        margin_scale = 0.50 / margin_util[margin_excess]
        port_w.loc[margin_excess] = port_w.loc[margin_excess].multiply(margin_scale, axis=0)

    return port_w


def build_fixed_portfolio(portfolio_vol_target=0.20, max_leverage=4.0, use_regime=True, vol_com=21):
    """V10 fixed equal-weight portfolio (unchanged for backward compat)."""
    all_close = v10_close
    all_trend_w = v10_trend_w
    returns_df = v10_returns_df
    eq_bear_probs = v10_eq_bear_probs
    portfolio_bear = v10_portfolio_bear
    EQUITY_SYMS = V10_EQUITY_SYMS

    syms = list(all_close.keys())
    N = len(syms)
    common_idx = None
    for sym in syms:
        idx = all_trend_w[sym].dropna().index
        common_idx = idx if common_idx is None else common_idx.intersection(idx)
    if common_idx is None or len(common_idx) < 252:
        return pd.DataFrame()

    budget = 1.0 / N
    port_w = pd.DataFrame(index=common_idx, columns=syms, dtype=float)
    for s in syms:
        w = all_trend_w[s].reindex(common_idx).fillna(0)
        if use_regime:
            sized = w.copy()
            if s in EQUITY_SYMS and s in eq_bear_probs:
                bp = eq_bear_probs[s]
                is_bear = bp > 0.55
                sized[is_bear & (w > 0)] *= 0.30
                sized[is_bear & (w < 0)] *= 0.70
            bp_port = portfolio_bear.reindex(w.index).fillna(0.5)
            bear_scale = np.where(bp_port > 0.25,
                                  np.clip(1.0 - (bp_port - 0.25) / 0.50 * 0.60, 0.40, 1.0), 1.0)
            sized *= bear_scale
            w = sized
        port_w[s] = w * budget

    r_aligned = returns_df.reindex(common_idx).fillna(0)[syms]
    port_ret = (port_w * r_aligned).sum(axis=1)
    port_vol = ewma_volatility(port_ret, com=vol_com)
    scale = (portfolio_vol_target / port_vol.clip(lower=0.005)).clip(upper=max_leverage)
    port_w = port_w.multiply(scale, axis=0)
    port_w = port_w.ewm(halflife=5).mean()

    gross = port_w.abs().sum(axis=1)
    excess = gross > max_leverage
    if excess.any():
        port_w.loc[excess] = port_w.loc[excess].div(gross[excess] / max_leverage, axis=0)

    margin_util = pd.Series(0.0, index=port_w.index)
    for col in port_w.columns:
        if col in instruments:
            margin_util += port_w[col].abs() * instruments[col].margin_init_pct
    margin_excess = margin_util > 0.50
    if margin_excess.any():
        margin_scale = 0.50 / margin_util[margin_excess]
        port_w.loc[margin_excess] = port_w.loc[margin_excess].multiply(margin_scale, axis=0)

    return port_w


def build_adaptive_portfolio_v11(
    signal_dict: dict[str, dict[str, pd.Series]],
    portfolio_vol_target: float = 0.25,
    max_leverage: float = 4.0,
    use_regime: bool = True,
    vol_com: int = 21,
    min_sharpe_threshold: float = 0.0,
):
    """V11 generalized portfolio builder.

    Accepts arbitrary signal families via signal_dict.
    Trailing Sharpe selection iterates over signal_dict keys.

    Parameters
    ----------
    signal_dict : dict[str, dict[str, pd.Series]]
        Outer key = signal name (e.g. 'trend', 'carry', 'meanrev').
        Inner dict = {symbol: vol-scaled signal Series}.
    portfolio_vol_target : float
        Target portfolio vol.
    max_leverage : float
        Maximum gross leverage.
    use_regime : bool
        Whether to apply regime overlay.
    vol_com : int
        EWMA com for portfolio vol targeting.
    min_sharpe_threshold : float
        Minimum trailing Sharpe to include an instrument.
    """
    all_close = v11_close
    all_returns = v11_returns
    returns_df = v11_returns_df
    eq_bear_probs = v11_eq_bear_probs
    portfolio_bear = v11_portfolio_bear
    EQUITY_SYMS = V11_EQUITY_SYMS

    sig_names = list(signal_dict.keys())
    all_dates = returns_df.dropna(how='all').index.sort_values()

    # Compute trailing Sharpe for each signal family x instrument
    trailing_sh = {}
    for sym in all_close:
        trailing_sh[sym] = {}
        r = all_returns[sym]
        for sig_name in sig_names:
            sig_w = signal_dict[sig_name].get(sym)
            if sig_w is None:
                trailing_sh[sym][sig_name] = pd.Series(0.0, index=r.index)
                continue
            pnl = sig_w * r
            trailing_sh[sym][sig_name] = pnl.rolling(ROLLING_WINDOW, min_periods=252).apply(
                lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0, raw=True)

    syms = list(all_close.keys())
    port_w = pd.DataFrame(0.0, index=all_dates, columns=syms)
    current_selections = {}
    current_budgets = {}
    current_universe = []
    last_rebal = None

    for i, date in enumerate(all_dates):
        need_rebal = (last_rebal is None or (i - last_rebal) >= REBAL_FREQ)
        if need_rebal and i >= ROLLING_WINDOW:
            new_selections = {}
            new_sharpes = {}
            for sym in all_close:
                if sym not in trailing_sh:
                    continue
                best_type = sig_names[0]
                best_sh = 0
                for sig_name in sig_names:
                    sh_series = trailing_sh[sym][sig_name]
                    sh_val = sh_series.iloc[i] if i < len(sh_series) else 0
                    if pd.isna(sh_val):
                        sh_val = 0
                    if sh_val > best_sh:
                        best_type = sig_name
                        best_sh = sh_val
                new_selections[sym] = best_type
                new_sharpes[sym] = best_sh

            current_universe = [s for s, sh in new_sharpes.items() if sh > min_sharpe_threshold]
            if len(current_universe) < 2:
                sorted_syms = sorted(new_sharpes.keys(), key=lambda s: new_sharpes[s], reverse=True)
                current_universe = sorted_syms[:max(2, len(sorted_syms) // 2)]

            current_selections = {s: new_selections[s] for s in current_universe}
            # Equal-weight budgets (lesson learned: Sharpe weighting overfits)
            budget = 1.0 / len(current_universe)
            current_budgets = {s: budget for s in current_universe}
            last_rebal = i

        for sym in current_universe:
            if sym not in current_selections:
                continue
            sig_name = current_selections[sym]
            sig_w = signal_dict[sig_name].get(sym)
            if sig_w is None:
                continue
            if date in sig_w.index:
                w_val = sig_w.loc[date]
                if not pd.isna(w_val):
                    if use_regime and sym in EQUITY_SYMS and sym in eq_bear_probs:
                        if date in eq_bear_probs[sym].index:
                            bp = eq_bear_probs[sym].loc[date]
                            if bp > 0.55:
                                w_val = w_val * (0.30 if w_val > 0 else 0.70)
                    if use_regime and date in portfolio_bear.index:
                        bp_port = portfolio_bear.loc[date]
                        if bp_port > 0.25:
                            bear_s = max(0.40, 1.0 - (bp_port - 0.25) / 0.50 * 0.60)
                            w_val *= bear_s
                    port_w.at[date, sym] = w_val * current_budgets.get(sym, 0)

    port_w = port_w.iloc[ROLLING_WINDOW:]
    r_aligned = returns_df.reindex(port_w.index).fillna(0)[syms]
    port_ret = (port_w * r_aligned).sum(axis=1)
    port_vol = ewma_volatility(port_ret, com=vol_com)
    scale = (portfolio_vol_target / port_vol.clip(lower=0.005)).clip(upper=max_leverage)
    port_w = port_w.multiply(scale, axis=0)
    port_w = port_w.ewm(halflife=5).mean()

    gross = port_w.abs().sum(axis=1)
    excess = gross > max_leverage
    if excess.any():
        port_w.loc[excess] = port_w.loc[excess].div(gross[excess] / max_leverage, axis=0)

    margin_util = pd.Series(0.0, index=port_w.index)
    for col in port_w.columns:
        if col in instruments:
            margin_util += port_w[col].abs() * instruments[col].margin_init_pct
    margin_limit = 0.80  # V11: 80% margin limit (diversified portfolio supports it)
    margin_excess = margin_util > margin_limit
    if margin_excess.any():
        margin_scale = margin_limit / margin_util[margin_excess]
        port_w.loc[margin_excess] = port_w.loc[margin_excess].multiply(margin_scale, axis=0)

    return port_w


# ============================================================
# BACKTEST & SAVE
# ============================================================
reporter = ReportGenerator(output_dir=os.path.join(PROJECT_ROOT, "output"))
dd_ladder = DrawdownLadder()
regime_oos = RegimeConditionedOOS(min_observations_per_bucket=20)

def backtest_and_save(wdf, run_id, initial_equity=1_000_000, close_df=None, returns_df=None):
    """Backtest with DD ladder and save equity curve + strategy data + risk timeseries."""
    if close_df is None:
        close_df = v10_close_df
    if returns_df is None:
        returns_df = v10_returns_df

    common = wdf.dropna(how='all').index.intersection(returns_df.dropna(how='all').index)
    w = wdf.loc[common].fillna(0)
    r = returns_df.loc[common].fillna(0)
    syms = [s for s in w.columns if s in r.columns]
    active = [s for s in syms if w[s].abs().sum() > 0]
    w = w[active]; r = r[active]
    meta_dict = {s: instruments[s] for s in active if s in instruments}
    prices = pd.DataFrame(close_df)[active].reindex(common).ffill()

    engine = VectorizedBacktestEngine(cost_model=cost_model, initial_equity=initial_equity)
    result = engine.run_with_dd_ladder(
        weights=w, returns=r, dd_ladder=dd_ladder,
        metadata=meta_dict, prices=prices,
    )

    equity = initial_equity * (1 + result.returns).cumprod()
    df = pd.DataFrame({
        'timestamp': result.returns.index,
        'equity': equity.values,
        'returns': result.returns.values,
    })

    out_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"equity_curve_portfolio_adjusted_{run_id}.csv")
    df.to_csv(path, index=False)

    strat = pd.DataFrame(index=common)
    strat.index.name = 'date'
    adj_w = result.weights
    for sym in active:
        strat[f'{sym}_signal'] = adj_w[sym] if sym in adj_w.columns else 0
        strat[f'{sym}_return'] = (adj_w[sym] if sym in adj_w.columns else 0) * r[sym]
    strat_path = os.path.join(out_dir, f"strategy_data_{run_id}.csv")
    strat.to_csv(strat_path)

    dd_multipliers = result.metadata.get("dd_multipliers", pd.Series(1.0, index=common))
    gross_leverage = result.weights.abs().sum(axis=1)
    margin_util = pd.Series(0.0, index=common)
    for col in result.weights.columns:
        if col in meta_dict:
            margin_util += result.weights[col].abs() * meta_dict[col].margin_init_pct
    dd_depth = (equity.cummax() - equity) / equity.cummax()
    composite_risk = (dd_depth + margin_util).clip(upper=1.0) / 2.0
    reporter.save_risk_timeseries(
        dd_multipliers=dd_multipliers.reindex(common).fillna(1.0),
        margin_util=margin_util,
        composite_risk=composite_risk,
        leverage=gross_leverage.reindex(common).fillna(0.0),
        run_id=run_id,
    )

    total = (1 + result.returns).prod() - 1
    ann_ret = result.metrics["annualized_return"]
    sharpe = result.metrics["sharpe_ratio"]
    dd = result.metrics["max_drawdown"]
    calmar = ann_ret / abs(dd) if dd != 0 else 0
    gl_p95 = gross_leverage.quantile(0.95)
    mu_max = margin_util.max()
    print(f"  {run_id}: Sh={sharpe:.3f} Ann={ann_ret:.1%} Tot={total:.0%} DD={dd:.1%} Cal={calmar:.3f}")
    print(f"    Lev p95={gl_p95:.1f}x  Margin max={mu_max:.1%}  -> {path}")
    return result


def save_regime_report(run_id: str, result, ref_prices_df: pd.DataFrame,
                       ref_returns: pd.Series):
    """Generate and save a regime report for a backtest result.

    Uses a reference index (e.g., ES) OHLCV for regime tagging, then
    conditions portfolio returns on vol/trend regimes.
    """
    try:
        port_returns = result.returns
        # Generate regime labels from reference index
        common = ref_prices_df.index.intersection(port_returns.index)
        if len(common) < 500:
            print(f"  Regime report: skipped (only {len(common)} common days)")
            return
        prices_aligned = ref_prices_df.loc[common]
        rets_aligned = ref_returns.reindex(common).fillna(0)

        # Need OHLCV columns for regime tagger
        required = {"open", "high", "low", "close"}
        if not required.issubset(set(prices_aligned.columns)):
            print(f"  Regime report: skipped (missing OHLCV columns)")
            return

        regime_report_obj = regime_oos.generate_report(
            returns=port_returns.reindex(common).fillna(0),
            prices=prices_aligned,
            fold_id=0,
        )
        report = reporter.generate_regime_report([regime_report_obj], run_id=run_id)
        n_bd = len(report.get("regime_breakdowns", []))
        print(f"  Regime report: {n_bd} breakdowns saved")
    except Exception as e:
        print(f"  Regime report: failed ({e})")


# ============================================================
# GENERATE SPY BUY-AND-HOLD
# ============================================================
def save_spy_buyhold():
    """Save S&P 500 buy-and-hold using actual SPY ETF data (dividend-adjusted)."""
    # Check for cached SPY CSV first (avoids network dependency)
    spy_cache = os.path.join(PROJECT_ROOT, "data_external", "spy_daily.csv")
    spy_returns = None

    if os.path.exists(spy_cache):
        print(f"  Loading cached SPY data from {spy_cache}")
        spy_df = pd.read_csv(spy_cache, parse_dates=["Date"], index_col="Date")
        spy_close = spy_df["Close"].squeeze()
        spy_returns = np.log(spy_close / spy_close.shift(1)).dropna()
        print(f"  SPY data: {len(spy_returns)} days from cache")

    if spy_returns is None:
        try:
            import yfinance as yf
            import socket
            socket.setdefaulttimeout(15)  # 15s timeout for network
            spy_raw = yf.download('SPY', start='1993-01-01', end='2027-01-01', progress=False, timeout=15)
            spy_close = spy_raw['Close'].squeeze()
            spy_returns = np.log(spy_close / spy_close.shift(1)).dropna()
            print(f"  SPY data: {len(spy_returns)} days via yfinance")
            # Cache for next run
            os.makedirs(os.path.dirname(spy_cache), exist_ok=True)
            spy_raw.to_csv(spy_cache)
            print(f"  Cached to {spy_cache}")
        except Exception as e:
            print(f"  yfinance failed ({e}), falling back to ES Norgate data")

    if spy_returns is None:
        if "ES" in v11_close:
            spy_returns = v11_returns["ES"]
        else:
            es_path = os.path.join(PROJECT_ROOT, V10_DATA_MAP["ES"])
            raw = load_bars_from_csv(es_path)
            raw = resample_bars(raw, "1D")
            if hasattr(raw.index, 'dayofweek'):
                raw = raw[raw.index.dayofweek < 5]
            raw.index = raw.index.normalize()
            raw = raw[~raw.index.duplicated(keep='first')]
            spy_returns = compute_returns(raw["close"], method="log")

    spy_equity = 1_000_000 * (1 + spy_returns).cumprod()
    df = pd.DataFrame({
        'timestamp': spy_returns.index,
        'equity': spy_equity.values,
        'returns': spy_returns.values,
    })
    df = df.dropna()

    out_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "equity_curve_portfolio_adjusted_spy_buyhold.csv")
    df.to_csv(path, index=False)

    total = (1 + spy_returns.dropna()).prod() - 1
    n = len(spy_returns.dropna())
    ann = ((1 + spy_returns.dropna()).prod() ** (252 / n)) - 1 if n > 10 else 0
    sh = spy_returns.mean() / spy_returns.std() * np.sqrt(252) if spy_returns.std() > 0 else 0
    print(f"  spy_buyhold: Ann={ann:.1%} Sh={sh:.2f} Tot={total:.0%} -> {path}")


# ============================================================
# MAIN
# ============================================================
print("\n=== Saving V10 configs ===\n")

# V10 Config 1: Conservative
print("Config 1: V10 Conservative (Roll Reg Carry vol=20%)")
w = build_adaptive_portfolio(portfolio_vol_target=0.20, use_regime=True, use_carry_blend=True)
backtest_and_save(w, "v10_conservative")

# V10 Config 2: Balanced
print("\nConfig 2: V10 Balanced (Roll Reg Carry thresh=0.20 vol=25%)")
w = build_adaptive_portfolio(portfolio_vol_target=0.25, use_regime=True, use_carry_blend=True, min_sharpe_threshold=0.20)
backtest_and_save(w, "v10_balanced")

# V10 Config 3: Aggressive
print("\nConfig 3: V10 Aggressive (Roll NoReg Carry vol=30%)")
w = build_adaptive_portfolio(portfolio_vol_target=0.30, use_regime=False, use_carry_blend=True)
backtest_and_save(w, "v10_aggressive")

# V10 Config 4: Baseline
print("\nConfig 4: V10 Baseline (Fixed EqWt Trend vol=20%)")
w = build_fixed_portfolio(portfolio_vol_target=0.20, use_regime=True)
backtest_and_save(w, "v10_baseline")

# V10 Leverage tiers
for lev in [2, 4, 8, 10]:
    print(f"\nLeverage Tier: Conservative {lev}x")
    w = build_adaptive_portfolio(
        portfolio_vol_target=0.20, max_leverage=float(lev),
        use_regime=True, use_carry_blend=True,
    )
    backtest_and_save(w, f"v10_lev_{lev}x")

# SPY Buy & Hold
print("\nSPY Buy & Hold:")
save_spy_buyhold()

# ============================================================
# V10 REGIME REPORTS
# ============================================================
# Load ES OHLCV reference for regime tagging (V10 uses 1-min resampled)
print("\nGenerating V10 regime reports...")
_es_path = os.path.join(PROJECT_ROOT, V10_DATA_MAP["ES"])
if os.path.exists(_es_path):
    _es_raw = load_bars_from_csv(_es_path)
    _es_raw = resample_bars(_es_raw, "1D")
    if hasattr(_es_raw.index, 'dayofweek'):
        _es_raw = _es_raw[_es_raw.index.dayofweek < 5]
    _es_raw.index = _es_raw.index.normalize()
    _es_raw = _es_raw[~_es_raw.index.duplicated(keep='first')]
    v10_ref_prices = _es_raw[["open", "high", "low", "close"]]
    v10_ref_returns = compute_returns(_es_raw["close"], method="log")

    for run_id in ["v10_conservative", "v10_balanced", "v10_aggressive", "v10_baseline"]:
        eq_path = os.path.join(PROJECT_ROOT, "output", f"equity_curve_portfolio_adjusted_{run_id}.csv")
        if os.path.exists(eq_path):
            _eq = pd.read_csv(eq_path, parse_dates=["timestamp"]).set_index("timestamp")
            _rets = _eq["returns"].dropna()

            class _FakeResult:
                def __init__(self, r): self.returns = r
            save_regime_report(run_id, _FakeResult(_rets), v10_ref_prices, v10_ref_returns)

# ============================================================
# V11 CONFIGS (expanded universe + signal families)
# ============================================================
if len(v11_close) >= 10:
    print("\n\n=== Saving V11 configs ===\n")

    # Prepare ES reference OHLCV from Norgate for V11 regime reports
    v11_ref_prices = None
    v11_ref_returns = None
    if "ES" in v11_close:
        _es_df = load_norgate_daily("ES", project_root=PROJECT_ROOT)
        v11_ref_prices = _es_df[["open", "high", "low", "close"]]
        v11_ref_returns = v11_returns["ES"]

    sig_dict_all = {
        'trend': v11_trend_w,
        'carry': v11_carry_w,
        'blend': v11_blend_w,
        'meanrev': v11_meanrev_w,
        'xsmom': v11_xsmom_w,
        'leadlag': v11_leadlag_w,
        'volbrk': v11_volbrk_w,
        'season': v11_season_w,
    }
    sig_dict_tc = {'trend': v11_trend_w, 'carry': v11_carry_w, 'blend': v11_blend_w}

    # V11 Config 1: Expanded trend+carry only (Phase 1 isolate)
    print("V11 Config 1: Expanded Trend+Carry (32 instruments, 50% vol, 8x lev)")
    w = build_adaptive_portfolio_v11(sig_dict_tc, portfolio_vol_target=0.50, max_leverage=8.0)
    result_v11_1 = backtest_and_save(w, "v11_expanded_trendcarry", close_df=v11_close_df, returns_df=v11_returns_df)
    if v11_ref_prices is not None:
        save_regime_report("v11_expanded_trendcarry", result_v11_1, v11_ref_prices, v11_ref_returns)

    # V11 Config 2: All signals — moderate (Sh=1.45, ~11.5% CAGR, DD~-16%)
    print("\nV11 Config 2: All Signals — Moderate (35% vol, 6x lev)")
    w = build_adaptive_portfolio_v11(sig_dict_all, portfolio_vol_target=0.35, max_leverage=6.0)
    result_v11_2 = backtest_and_save(w, "v11_all_moderate", close_df=v11_close_df, returns_df=v11_returns_df)
    if v11_ref_prices is not None:
        save_regime_report("v11_all_moderate", result_v11_2, v11_ref_prices, v11_ref_returns)

    # V11 Config 3: All signals — growth (sweet spot: Sh=1.42, ~15% CAGR, DD~-22%)
    print("\nV11 Config 3: All Signals — Growth (50% vol, 8x lev)")
    w = build_adaptive_portfolio_v11(sig_dict_all, portfolio_vol_target=0.50, max_leverage=8.0)
    result_v11_3 = backtest_and_save(w, "v11_all_growth", close_df=v11_close_df, returns_df=v11_returns_df)
    if v11_ref_prices is not None:
        save_regime_report("v11_all_growth", result_v11_3, v11_ref_prices, v11_ref_returns)

    # V11 Config 4: All signals — aggressive (Sh=1.34, ~16% CAGR, DD~-26%)
    print("\nV11 Config 4: All Signals — Aggressive (55% vol, 9x lev)")
    w = build_adaptive_portfolio_v11(sig_dict_all, portfolio_vol_target=0.55, max_leverage=9.0)
    result_v11_4 = backtest_and_save(w, "v11_all_aggressive", close_df=v11_close_df, returns_df=v11_returns_df)
    if v11_ref_prices is not None:
        save_regime_report("v11_all_aggressive", result_v11_4, v11_ref_prices, v11_ref_returns)

else:
    print("\n\nSKIPPING V11 configs: insufficient Norgate data (run scripts/export_norgate_data.py first)")

print("\n=== Done! Launch dashboard: streamlit run dashboard/app.py ===")
