"""
Export all instruments from Norgate Data API to data_norgate/ as CSV files.
Decouples backtesting from NDU running.

Run once (with NDU running): python scripts/export_norgate_data.py
"""
import sys, os, json
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import norgatedata
except ImportError:
    print("ERROR: norgatedata package not installed. Run: pip install norgatedata")
    sys.exit(1)

import pandas as pd

# Full symbol map: internal name -> Norgate back-adjusted continuous symbol
SYMBOL_MAP = {
    # INDEX
    "ES":   "&ES_CCB",
    "NQ":   "&NQ_CCB",
    "FDAX": "&FDAX_CCB",
    "NKD":  "&NKD_CCB",
    "LFT":  "&LFT_CCB",
    "HSI":  "&HSI_CCB",
    "YAP":  "&YAP_CCB",
    # DEBT
    "ZN":   "&ZN_CCB",
    "ZB":   "&ZB_CCB",
    "ZF":   "&ZF_CCB",
    "FGBL": "&FGBL_CCB",
    # ENERGY
    "CL":   "&CL_CCB",
    "NG":   "&NG_CCB",
    "HO":   "&HO_CCB",
    "RB":   "&RB_CCB",
    # METALS
    "GC":   "&GC_CCB",
    "SI":   "&SI_CCB",
    "HG":   "&HG_CCB",
    "PL":   "&PL_CCB",
    "PA":   "&PA_CCB",
    # AGS
    "ZC":   "&ZC_CCB",
    "ZS":   "&ZS_CCB",
    "ZW":   "&ZW_CCB",
    "KC":   "&KC_CCB",
    "CT":   "&CT_CCB",
    "SB":   "&SB_CCB",
    "CC":   "&CC_CCB",
    # FX
    "6E":   "&6E_CCB",
    "6J":   "&6J_CCB",
    "6B":   "&6B_CCB",
    "6A":   "&6A_CCB",
    "6C":   "&6C_CCB",
}

ASSET_CLASS_MAP = {
    "ES": "INDEX", "NQ": "INDEX", "FDAX": "INDEX", "NKD": "INDEX",
    "LFT": "INDEX", "HSI": "INDEX", "YAP": "INDEX",
    "ZN": "DEBT", "ZB": "DEBT", "ZF": "DEBT", "FGBL": "DEBT",
    "CL": "ENERGY", "NG": "ENERGY", "HO": "ENERGY", "RB": "ENERGY",
    "GC": "METALS", "SI": "METALS", "HG": "METALS", "PL": "METALS", "PA": "METALS",
    "ZC": "AGS", "ZS": "AGS", "ZW": "AGS", "KC": "AGS",
    "CT": "AGS", "SB": "AGS", "CC": "AGS",
    "6E": "FX", "6J": "FX", "6B": "FX", "6A": "FX", "6C": "FX",
}

OUT_DIR = os.path.join(PROJECT_ROOT, "data_norgate")
os.makedirs(OUT_DIR, exist_ok=True)


def export_symbol(internal: str, norgate_sym: str) -> dict | None:
    """Export a single symbol from Norgate to CSV. Returns metadata dict or None."""
    try:
        pricedata = norgatedata.price_timeseries(
            norgate_sym,
            stock_price_adjustment_setting=norgatedata.StockPriceAdjustmentType.TOTALRETURN,
            padding_setting=norgatedata.PaddingType.NONE,
            timeseriesformat='pandas-dataframe',
        )
    except Exception as e:
        print(f"  SKIP {internal} ({norgate_sym}): {e}")
        return None

    if pricedata is None or len(pricedata) == 0:
        print(f"  SKIP {internal} ({norgate_sym}): no data returned")
        return None

    # Normalize columns to lowercase
    pricedata.columns = [c.lower() for c in pricedata.columns]

    # Keep only OHLCV
    keep_cols = []
    for target in ["open", "high", "low", "close", "volume"]:
        for col in pricedata.columns:
            if col == target:
                keep_cols.append(col)
                break
    pricedata = pricedata[keep_cols]

    # Ensure DatetimeIndex
    if not isinstance(pricedata.index, pd.DatetimeIndex):
        pricedata.index = pd.to_datetime(pricedata.index)
    pricedata.index.name = "datetime"
    pricedata = pricedata.sort_index()
    pricedata = pricedata[~pricedata.index.duplicated(keep='first')]

    # Save to CSV
    out_path = os.path.join(OUT_DIR, f"{internal}_daily.csv")
    pricedata.to_csv(out_path)

    first_date = str(pricedata.index[0].date())
    last_date = str(pricedata.index[-1].date())
    n_bars = len(pricedata)

    print(f"  {internal:>5}: {n_bars:>6} bars  {first_date} -> {last_date}")

    return {
        "symbol": internal,
        "norgate_symbol": norgate_sym,
        "asset_class": ASSET_CLASS_MAP.get(internal, "UNKNOWN"),
        "first_date": first_date,
        "last_date": last_date,
        "n_bars": n_bars,
        "file": f"{internal}_daily.csv",
    }


def main():
    print(f"\n=== Exporting {len(SYMBOL_MAP)} instruments from Norgate ===\n")

    manifest = {}
    success = 0
    fail = 0

    for internal, norgate_sym in sorted(SYMBOL_MAP.items()):
        meta = export_symbol(internal, norgate_sym)
        if meta:
            manifest[internal] = meta
            success += 1
        else:
            fail += 1

    # Write manifest
    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n=== Done: {success} exported, {fail} failed ===")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
