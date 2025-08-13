#!/usr/bin/env python3
"""
sisl_vfd_report_order.py

Enforces SISL VFD report ordering rules and "two rows per model" (2024 first, 2025 second).
Also permanently removes any SL/Serial column.

Usage:
    python sisl_vfd_report_order.py --in "VFD PRICE LIST — 2024 vs 2025_Ref_Report.csv" --out "VFD_PRICE_LIST_2024_vs_2025_TWO_ROWS_PER_MODEL.csv"
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_capacity_kw(val):
    """Extract numeric kW from Capacity like '37 kW' or from model token '-37K-' if needed."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:k|kw|kW)?', s, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m2 = re.search(r'-(\d+(?:\.\d+)?)\s*[Kk]\b', s)
    if m2:
        return float(m2.group(1))
    return None


def detect_series(model):
    """Map model strings to series buckets D(S), D, E, F, A, HEL."""
    if pd.isna(model):
        return None
    m = str(model)
    if 'HEL' in m:
        return 'HEL'
    if 'D720S' in m:
        return 'D(S)'
    if 'D720' in m or 'D740' in m:
        return 'D'
    if any(tag in m for tag in ['E820','E720','E840','E740']):
        return 'E'
    if 'F840' in m:
        return 'F'
    if 'A820' in m or 'A840' in m:
        return 'A'
    return None


def voltage_cat(model):
    """
    1P-200V (special S models) -> 3P-200V (D720/E720/E820/A820) -> 3P-400V (D740/E740/E840/F840/A840/HEL).
    """
    m = str(model) if not pd.isna(model) else ''
    if 'D720S' in m:
        return '1P-200V'
    if any(tag in m for tag in ['D720-', 'E720', 'E820', 'A820']):
        return '3P-200V'
    if any(tag in m for tag in ['D740', 'E740', 'E840', 'F840', 'A840', 'HEL']):
        return '3P-400V'
    return 'Other'


def e_subrank(model):
    """
    For E-series sorting inside a capacity bucket:
      - 200V: E720 before E820
      - 400V: E740 before E840
    """
    m = str(model)
    if 'E720' in m: 
        return 0
    if 'E820' in m: 
        return 1
    if 'E740' in m: 
        return 0
    if 'E840' in m: 
        return 1
    return 2


def normalize_year(y):
    """Return int year if possible, else NaN."""
    try:
        return int(str(y).strip())
    except Exception:
        return np.nan


def enforce_two_rows_per_model(input_csv: Path, output_csv: Path) -> None:
    df = pd.read_csv(input_csv)

    # Normalize Year and derive order helpers
    if 'Year' not in df.columns:
        raise ValueError("Input CSV must include a 'Year' column.")
    df['Year'] = df['Year'].apply(normalize_year)

    # Remove any SL/Serial columns permanently
    drop_candidates = [c for c in df.columns if c.strip().lower() in 
                       ['sl', 'sl#', 'serial', 'serial#', 'sl no', 'sl no.','slno','sl.']]
    df = df.drop(columns=drop_candidates, errors='ignore')

    # Derive ordering keys
    df['cap_kw'] = df['Capacity'].apply(parse_capacity_kw) if 'Capacity' in df.columns else np.nan
    df['Series'] = df['Model'].apply(detect_series) if 'Model' in df.columns else None
    df['VoltCat'] = df['Model'].apply(voltage_cat) if 'Model' in df.columns else None

    volt_order = {'1P-200V': 0, '3P-200V': 1, '3P-400V': 2, 'Other': 3}
    series_order = {'D(S)': 0, 'D': 1, 'E': 2, 'F': 3, 'A': 4, 'HEL': 5}

    df['volt_rank'] = df['VoltCat'].map(volt_order).fillna(3).astype(int)
    df['series_rank'] = df['Series'].map(series_order).fillna(6).astype(int)
    df['e_subrank'] = df['Model'].apply(e_subrank).astype(int)

    # Establish model order (ignore years for now)
    model_order_df = (df
        .sort_values(['volt_rank','cap_kw','series_rank','e_subrank','Model'], kind='mergesort')
        [['Model','Capacity','volt_rank','cap_kw','series_rank','e_subrank']]
        .drop_duplicates(subset=['Model'])
        .reset_index(drop=True))

    # Preserve only known output columns (keep in original order if present)
    preferred_cols = ['Capacity','Model','Year','Price','15%','20%','25%','30%','35%','COGS','COGS + GP%']
    output_cols = [c for c in preferred_cols if c in df.columns]
    # Add any extra columns that may exist in input but not in preferred list, keeping their relative order
    extras = [c for c in df.columns if c not in output_cols and c not in
              ['cap_kw','Series','VoltCat','volt_rank','series_rank','e_subrank']]
    output_cols += extras

    # Build exactly two rows per model (2024 then 2025). Blank placeholders if missing.
    pairs = []
    for _, row in model_order_df.iterrows():
        model = row['Model']
        cap = row['Capacity']
        sub = df[df['Model'] == model].copy()

        r24 = sub[sub['Year'] == 2024].head(1)
        r25 = sub[sub['Year'] == 2025].head(1)

        if r24.empty:
            r24 = pd.DataFrame([{col: np.nan for col in output_cols}])
            if 'Capacity' in output_cols: r24.at[0,'Capacity'] = cap
            if 'Model' in output_cols: r24.at[0,'Model'] = model
            if 'Year' in output_cols: r24.at[0,'Year'] = 2024

        if r25.empty:
            r25 = pd.DataFrame([{col: np.nan for col in output_cols}])
            if 'Capacity' in output_cols: r25.at[0,'Capacity'] = cap
            if 'Model' in output_cols: r25.at[0,'Model'] = model
            if 'Year' in output_cols: r25.at[0,'Year'] = 2025

        r24 = r24[output_cols]
        r25 = r25[output_cols]

        pairs.append(pd.concat([r24, r25], ignore_index=True))

    out_df = pd.concat(pairs, ignore_index=True)
    out_df.to_csv(output_csv, index=False)


def main():
    ap = argparse.ArgumentParser(description="Apply SISL VFD ordering and two-rows-per-model rule.")
    ap.add_argument('--in', dest='input_csv', required=True, help='Path to input CSV')
    ap.add_argument('--out', dest='output_csv', required=True, help='Path to output CSV')
    args = ap.parse_args()

    in_path = Path(args.input_csv)
    out_path = Path(args.output_csv)

    if not in_path.exists():
        print(f"ERROR: Input CSV not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    enforce_two_rows_per_model(in_path, out_path)
    print(f"✅ Wrote: {out_path}")


if __name__ == '__main__':
    main()
