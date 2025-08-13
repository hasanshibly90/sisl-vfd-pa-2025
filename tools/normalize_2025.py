#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Normalize 2025 CSV so the generator can read it.

Reads:
  data/prices_2024.csv   (must have capacity+model)
  data/prices_2025.csv   (you said it has ['Model Name','COGS'])

Writes:
  data/prices_2025_normalized.csv with columns:
    - Capacity (KW)
    - Model (Rated Current)
    - Price (BDT)
Also writes data/missing_2025_capacity.csv if any capacities can't be found.
"""
import sys
from pathlib import Path
import pandas as pd

Y24 = Path("data/prices_2024.csv")
Y25 = Path("data/prices_2025.csv")
OUT = Path("data/prices_2025_normalized.csv")
MISS = Path("data/missing_2025_capacity.csv")

CAP_24_CAND   = ["Capacity (KW)", "Capacity (kW)", "Capacity", "kW"]
MODEL_24_CAND = ["Model (Rated Current)", "Model", "MODEL", "Model Name"]
MODEL_25_CAND = ["Model Name", "Model (Rated Current)", "Model", "MODEL"]
PRICE_25_CAND = ["Price (BDT)", "COGS", "List Price", "LP (BDT)", "LP_BDT", "Price"]

def pick_col(df, choices):
    for c in choices:
        if c in df.columns:
            return c
    return None

def norm(s): 
    return str(s).strip().upper()

if not Y24.exists() or not Y25.exists():
    print("ERROR: expected data/prices_2024.csv and data/prices_2025.csv")
    sys.exit(1)

# Read 2024
df24 = pd.read_csv(Y24, dtype=str, keep_default_na=False)
cap24 = pick_col(df24, CAP_24_CAND)
mod24 = pick_col(df24, MODEL_24_CAND)
if not cap24 or not mod24:
    print(f"ERROR: 2024 file missing capacity/model columns. Found: {list(df24.columns)}")
    sys.exit(1)

# Build capacity lookup keyed by 2024 model (normalized)
cap_map = {norm(m): c for m, c in zip(df24[mod24], df24[cap24]) if str(m).strip()}

# Read 2025 (your columns are likely 'Model Name' + 'COGS')
df25 = pd.read_csv(Y25, dtype=str, keep_default_na=False)
mod25 = pick_col(df25, MODEL_25_CAND)
prc25 = pick_col(df25, PRICE_25_CAND)
if not mod25 or not prc25:
    print(f"ERROR: 2025 file missing model/price columns. Found: {list(df25.columns)}")
    sys.exit(1)

# Build normalized 2025 with required headers
out = pd.DataFrame({
    "Capacity (KW)": df25[mod25].map(lambda x: cap_map.get(norm(x), "")),
    "Model (Rated Current)": df25[mod25],  # keep 2025 model text verbatim
    "Price (BDT)": df25[prc25],            # use your 2025 price/COGS
})

# Save rows with missing capacity for you to fill, if any
missing_mask = out["Capacity (KW)"].astype(str).str.strip().eq("")
if missing_mask.any():
    out[missing_mask].to_csv(MISS, index=False, encoding="utf-8-sig")
    print(f"NOTE: {missing_mask.sum()} rows missing capacity → {MISS}")

# Write normalized file
out.to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"✓ Wrote: {OUT}")
