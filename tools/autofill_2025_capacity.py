#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autofill missing Capacity (KW) in data/prices_2025_normalized.csv
Heuristics:
  1) Parse tokens like "0.75K", "11K", "400K" (or "0.75 kW") from the 2025 model text
  2) Fallback fuzzy match to 2024 model list (normalized) to copy capacity
Outputs:
  - data/prices_2025_normalized_filled.csv (preferred input for the generator)
  - data/missing_2025_capacity_after_autofill.csv (still-missing rows, if any)
"""
from pathlib import Path
import re
import pandas as pd
from difflib import get_close_matches

Y24 = Path("data/prices_2024.csv")
N25 = Path("data/prices_2025_normalized.csv")
OUT = Path("data/prices_2025_normalized_filled.csv")
MISS = Path("data/missing_2025_capacity_after_autofill.csv")

CAP24_CAND   = ["Capacity (KW)", "Capacity (kW)", "Capacity", "kW"]
MOD24_CAND   = ["Model (Rated Current)", "Model", "MODEL", "Model Name"]
MOD25_COL    = "Model (Rated Current)"   # in normalized file
CAP25_COL    = "Capacity (KW)"

def pick_col(df, choices):
    for c in choices:
        if c in df.columns:
            return c
    return None

def norm(s: str) -> str:
    # strip non-alphanumerics, uppercase
    return re.sub(r"[^A-Za-z0-9]+", "", str(s)).upper()

def parse_cap_from_model(model: str) -> str:
    if not model:
        return ""
    m = re.search(r"(\d+(?:\.\d+)?)\s*[kK]\b", model)   # 0.75K / 11k / 400k
    if m:
        return f"{m.group(1)}K"
    m2 = re.search(r"(\d+(?:\.\d+)?)\s*(?:kW|KW)\b", model)  # 0.75 kW
    if m2:
        return f"{m2.group(1)}K"
    return ""

# Load inputs
if not Y24.exists() or not N25.exists():
    raise SystemExit("Expected data/prices_2024.csv and data/prices_2025_normalized.csv")

df24 = pd.read_csv(Y24, dtype=str, keep_default_na=False)
cap24 = pick_col(df24, CAP24_CAND)
mod24 = pick_col(df24, MOD24_CAND)
if not cap24 or not mod24:
    raise SystemExit(f"2024 file missing capacity/model headers. Found: {list(df24.columns)}")

df25 = pd.read_csv(N25, dtype=str, keep_default_na=False)
if MOD25_COL not in df25.columns or CAP25_COL not in df25.columns:
    raise SystemExit(f"{N25} missing expected columns. Found: {list(df25.columns)}")

# Build lookup from 2024
cap_map_24 = {}
norm_keys_24 = []
for m, c in zip(df24[mod24], df24[cap24]):
    key = norm(m)
    if key:
        cap_map_24[key] = c
        norm_keys_24.append(key)

filled_from_regex = 0
filled_from_fuzzy = 0

# Pass 1: regex extraction from 2025 model text
missing_mask = df25[CAP25_COL].astype(str).str.strip().eq("")
for idx in df25[missing_mask].index:
    guess = parse_cap_from_model(df25.at[idx, MOD25_COL])
    if guess:
        df25.at[idx, CAP25_COL] = guess
        filled_from_regex += 1

# Pass 2: fuzzy/contains match to 2024 models (for any still empty)
missing_mask = df25[CAP25_COL].astype(str).str.strip().eq("")
for idx in df25[missing_mask].index:
    key25 = norm(df25.at[idx, MOD25_COL])
    if not key25:
        continue
    # Try direct lookup
    if key25 in cap_map_24:
        df25.at[idx, CAP25_COL] = cap_map_24[key25]
        filled_from_fuzzy += 1
        continue
    # Try close matches
    matches = get_close_matches(key25, norm_keys_24, n=1, cutoff=0.86)
    if matches:
        df25.at[idx, CAP25_COL] = cap_map_24[matches[0]]
        filled_from_fuzzy += 1

# Save results
df25.to_csv(OUT, index=False, encoding="utf-8-sig")
still_missing = df25[df25[CAP25_COL].astype(str).str.strip().eq("")]
if not still_missing.empty:
    still_missing.to_csv(MISS, index=False, encoding="utf-8-sig")

print(f"✓ Wrote: {OUT}")
print(f"  Autofilled from regex: {filled_from_regex}")
print(f"  Autofilled from fuzzy: {filled_from_fuzzy}")
print(f"  Still missing: {len(still_missing)}{' → ' + str(MISS) if len(still_missing) else ''}")
