#!/usr/bin/env python3
"""
VFD CSV Merger — 2024 vs 2025 (pairwise rows)

Usage:
  python vfd_csv_merger.py --y2024 prices_2024.csv --y2025 prices_2025.csv --out VFD_2024_vs_2025.csv

What it does:
- Reads two CSVs (2024 + 2025).
- Normalizes columns (tries to auto-detect; see COL_GUESS).
- Builds a canonical model key: (Voltage, Capacity_kW, Series, Model).
- Outputs TWO rows per model (Year=2024, then Year=2025).
- Enforces ordering:
    VoltageOrder: 1φ 200-220V → 3φ 200-220V → 3φ 380-480V
    SeriesOrder : D → E → F → A → HEL
    Capacity    : ascending (float)
    Year        : 2024, then 2025
- For HEL models, if 2024 is missing, it adds a blank 2024 row (price empty).
- If any field is missing, attempts to infer from Model string.
"""
import argparse
import re
from pathlib import Path
import pandas as pd

SERIES_ORDER = {'D': 1, 'E': 2, 'F': 3, 'A': 4, 'HEL': 5}
VOLTAGE_ORDER = {'1φ 200-220V': 0, '3φ 200-220V': 1, '3φ 380-480V': 2}

# Candidate column names for auto-detection
COL_GUESS = {
    'model': ['Model', 'MODEL', 'Model Name', 'Model_Name', 'Model (Rated Current)'],
    'price': ['Price (BDT)', 'Price_BDT', 'Price', 'LP (BDT)', 'List Price', 'LP_BDT'],
    'capacity': ['Capacity (KW)', 'Capacity_kW', 'Capacity', 'kW'],
    'series': ['Series'],
    'voltage': ['Voltage', 'Voltage Group', 'Volt'],
    'rated_current': ['Rated Current', 'Rated_Current', 'RatedCurrent', 'RC (A)', 'RC_A']
}

CAP_RE = re.compile(r'(\d+(?:\.\d+)?)\s*[Kk]\b')
RC_RE = re.compile(r'(\d+(?:\.\d+)?)\s*[Aa]\b')

def pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def detect_series(model: str) -> str:
    if not isinstance(model, str):
        return ''
    m = model.upper()
    if 'HEL' in m:
        return 'HEL'
    # Prefer explicit FR-* prefix
    for s in ['D','E','F','A']:
        if f'FR-{s}' in m:
            return s
    # Fallback: look for -D7xx / -E8xx etc
    m2 = re.search(r'FR-([DEFA])', m)
    if m2:
        return m2.group(1)
    # Fallback based on numeric family hints
    if '720' in m: return 'D' if 'FR-D' in m else 'E' if 'FR-E' in m else 'E'
    if '740' in m: return 'D' if 'FR-D' in m else 'E'
    if '820' in m: return 'A' if 'FR-A' in m else 'E'
    if '840' in m: return 'F' if 'FR-F' in m else 'A' if 'FR-A' in m else 'E'
    return ''

def detect_voltage(model: str, default=''):
    if not isinstance(model, str):
        return default
    m = model.upper()
    if '20S' in m:
        return '1φ 200-220V'
    if '720' in m or '820' in m:
        return '3φ 200-220V'
    if '740' in m or '840' in m or 'HEL' in m:
        return '3φ 380-480V'
    return default

def parse_capacity(model: str, fallback=None):
    # Prefer regex in the model string like '3.7K' or '75K'
    if isinstance(model, str):
        hit = CAP_RE.search(model)
        if hit:
            try:
                return float(hit.group(1))
            except:
                pass
    return fallback

def parse_rc(text: str, fallback=None):
    # Tries to parse "(216A)" or "216 A"
    if isinstance(text, str):
        hit = RC_RE.search(text)
        if hit:
            try:
                return float(hit.group(1))
            except:
                pass
    return fallback

def normalize(df: pd.DataFrame, year: int) -> pd.DataFrame:
    df = df.copy()
    # Model
    c_model = pick_col(df, COL_GUESS['model'])
    if not c_model:
        raise ValueError("Couldn't find a 'Model' column. Please rename or pass a CSV with a Model column.")
    df.rename(columns={c_model: 'Model'}, inplace=True)
    df['Model'] = df['Model'].astype(str).str.strip()

    # Price
    c_price = pick_col(df, COL_GUESS['price'])
    if not c_price:
        raise ValueError("Couldn't find a 'Price (BDT)' column. Please include one of: " + ', '.join(COL_GUESS['price']))
    df.rename(columns={c_price: 'Price (BDT)'}, inplace=True)

    # Capacity
    c_cap = pick_col(df, COL_GUESS['capacity'])
    if c_cap and c_cap != 'Capacity (kW)':
        df.rename(columns={c_cap: 'Capacity (kW)'}, inplace=True)
    if 'Capacity (kW)' not in df.columns:
        df['Capacity (kW)'] = df['Model'].apply(parse_capacity)

    # Series
    c_series = pick_col(df, COL_GUESS['series'])
    if c_series and c_series != 'Series':
        df.rename(columns={c_series: 'Series'}, inplace=True)
    if 'Series' not in df.columns:
        df['Series'] = df['Model'].apply(detect_series)

    # Voltage
    c_volt = pick_col(df, COL_GUESS['voltage'])
    if c_volt and c_volt != 'Voltage':
        df.rename(columns={c_volt: 'Voltage'}, inplace=True)
    if 'Voltage' not in df.columns:
        df['Voltage'] = df['Model'].apply(detect_voltage)

    # Rated Current
    c_rc = pick_col(df, COL_GUESS['rated_current'])
    if c_rc and c_rc != 'Rated Current':
        df.rename(columns={c_rc: 'Rated Current'}, inplace=True)
    if 'Rated Current' not in df.columns:
        # Try to parse from Model like "FR-F840-110K-1 (216A)"
        df['Rated Current'] = df['Model'].apply(parse_rc)

    # Year
    df['Year'] = int(year)

    # Keep only relevant columns (preserve if present)
    cols = ['Voltage', 'Capacity (kW)', 'Series', 'Model', 'Rated Current', 'Year', 'Price (BDT)']
    for c in cols:
        if c not in df.columns:
            df[c] = ''
    df = df[cols].copy()
    return df

def build_pairs(df24: pd.DataFrame, df25: pd.DataFrame) -> pd.DataFrame:
    # Union of model keys
    key_cols = ['Voltage', 'Capacity (kW)', 'Series', 'Model']
    # Prefer non-null RC from either year
    rc_map = (
        pd.concat([df25[['Model','Rated Current']], df24[['Model','Rated Current']]])
        .dropna(subset=['Rated Current'])
        .drop_duplicates('Model')
        .set_index('Model')['Rated Current']
        .to_dict()
    )

    keys = (
        pd.concat([df24[key_cols], df25[key_cols]])
        .drop_duplicates()
        .copy()
    )

    # Ordering helpers
    keys['VoltageOrder'] = keys['Voltage'].map(VOLTAGE_ORDER).fillna(99).astype(int)
    keys['SeriesOrder']  = keys['Series'].map(SERIES_ORDER).fillna(99).astype(int)
    # Missing capacities go to the end
    keys['CapOrder'] = keys['Capacity (kW)'].astype(float).fillna(10_000.0)

    keys = keys.sort_values(['VoltageOrder', 'CapOrder', 'SeriesOrder', 'Model'], kind='mergesort')

    out_rows = []
    for _, k in keys.iterrows():
        k_mask_24 = (df24[key_cols] == k[key_cols]).all(axis=1)
        k_mask_25 = (df25[key_cols] == k[key_cols]).all(axis=1)

        row24 = df24[k_mask_24].head(1).copy()
        row25 = df25[k_mask_25].head(1).copy()

        base = {
            'Voltage': k['Voltage'],
            'Capacity (kW)': k['Capacity (kW)'],
            'Series': k['Series'],
            'Model': k['Model'],
            'Rated Current': rc_map.get(k['Model'], None)
        }

        # Always emit 2024 then 2025
        if row24.empty:
            out_rows.append({**base, 'Year': 2024, 'Price (BDT)': ''})
        else:
            r = row24.iloc[0].to_dict()
            r.update({'Year': 2024})
            out_rows.append({**base, **r})

        if row25.empty:
            out_rows.append({**base, 'Year': 2025, 'Price (BDT)': ''})
        else:
            r = row25.iloc[0].to_dict()
            r.update({'Year': 2025})
            out_rows.append({**base, **r})

    df_out = pd.DataFrame(out_rows)
    # Final stable sort including Year (2024 then 2025)
    df_out['VoltageOrder'] = df_out['Voltage'].map(VOLTAGE_ORDER).fillna(99).astype(int)
    df_out['SeriesOrder']  = df_out['Series'].map(SERIES_ORDER).fillna(99).astype(int)
    df_out['CapOrder']     = df_out['Capacity (kW)'].astype(float).fillna(10_000.0)
    df_out['YearOrder']    = df_out['Year'].map({2024: 0, 2025: 1}).astype(int)

    df_out = df_out.sort_values(
        ['VoltageOrder', 'CapOrder', 'SeriesOrder', 'Model', 'YearOrder'],
        kind='mergesort'
    ).drop(columns=['VoltageOrder','SeriesOrder','CapOrder','YearOrder'])

    return df_out

def main():
    ap = argparse.ArgumentParser(description="Merge 2024/2025 VFD price CSVs into pairwise rows with strict ordering.")
    ap.add_argument("--y2024", required=True, help="CSV path for 2024 list")
    ap.add_argument("--y2025", required=True, help="CSV path for 2025 list (HEL may be 2025-only)")
    ap.add_argument("--out", default="VFD_2024_vs_2025.csv", help="Output CSV path")
    args = ap.parse_args()

    df24_raw = pd.read_csv(args.y2024)
    df25_raw = pd.read_csv(args.y2025)

    df24 = normalize(df24_raw, 2024)
    df25 = normalize(df25_raw, 2025)

    # Build pair rows
    df_out = build_pairs(df24, df25)

    # Write
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"✓ Wrote: {out_path.resolve()}  (rows: {len(df_out)})")

if __name__ == "__main__":
    main()
