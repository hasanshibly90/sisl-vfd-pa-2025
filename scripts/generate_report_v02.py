# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd, numpy as np, re

ROOT = Path(".")
COGS_2024 = ROOT / "VFD_COGS_2024_Last.csv"
COGS_2025 = ROOT / "VFD_COGS_2025_Final.csv"
LIST_2024 = ROOT / "VFD_List_Price_2024_Final.csv"

def read_smart(path: Path) -> pd.DataFrame:
    # auto-detect delimiter, keep strings
    df = pd.read_csv(path, engine="python", sep=None, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    return df

MODEL_RX = re.compile(r"FR-[A-Z][A-Z0-9]*-?[0-9.]+K", re.I)

def guess_model_col(df: pd.DataFrame) -> str | None:
    # 1) try obvious header names
    for name in df.columns:
        n = name.lower().replace(" ", "")
        if n in {"model","modelname","modelno","modelnumber","model_name"}:
            return name
    # 2) fallback: column with most FR-...K matches
    best = (0.0, None)
    for name in df.columns:
        s = df[name].astype(str)
        ratio = s.str.contains(MODEL_RX, na=False).mean()
        if ratio > best[0]:
            best = (ratio, name)
    return best[1] if best[0] >= 0.05 else None

def find_col(df: pd.DataFrame, *keywords) -> str | None:
    # score headers by keyword presence
    best = (0, None)
    for name in df.columns:
        key = name.lower().replace(" ", "")
        score = sum(k in key for k in keywords)
        if score > best[0]:
            best = (score, name)
    return best[1] if best[0] > 0 else None

def numify(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")

def extract_series(model: str) -> str | None:
    m = re.search(r"FR-([A-Z]+)", str(model).upper())
    return m.group(1) if m else None

def extract_kw_from_model(model: str):
    m = re.search(r"-([0-9]+(?:\.[0-9]+)?)K", str(model).upper())
    return float(m.group(1)) if m else np.nan

# Load files
c24 = read_smart(COGS_2024)
c25 = read_smart(COGS_2025)
lp24 = read_smart(LIST_2024)

# Normalize model column name to "Model"
for name, df in [("COGS_2024", c24), ("COGS_2025", c25), ("LIST_2024", lp24)]:
    mcol = guess_model_col(df)
    if not mcol:
        raise SystemExit(f"Could not detect a Model column in {name}. Headers: {list(df.columns)}")
    if mcol != "Model":
        df.rename(columns={mcol: "Model"}, inplace=True)
    df["Model"] = df["Model"].astype(str).str.strip()

# Model universe
models = sorted(set(c24["Model"]) | set(c25["Model"]) | set(lp24["Model"]))
df = pd.DataFrame({"Model": models})

# Derive Series and Capacity
df["Series"] = df["Model"].map(extract_series)
df["Capacity (kW)"] = df["Model"].map(extract_kw_from_model)

# Map source columns
c24_cogs = find_col(c24, "cogs", "cost")
c25_cogs = find_col(c25, "cogs", "cost")
lp24_price = find_col(lp24, "list", "price", "lp", "bdt")

if c24_cogs: df = df.merge(c24[["Model", c24_cogs]].rename(columns={c24_cogs: "COGS_2024"}), on="Model", how="left")
if c25_cogs: df = df.merge(c25[["Model", c25_cogs]].rename(columns={c25_cogs: "COGS_2025"}), on="Model", how="left")
if lp24_price: df = df.merge(lp24[["Model", lp24_price]].rename(columns={lp24_price: "List_2024"}), on="Model", how="left")

# Numeric conversion
for col in ["COGS_2024", "COGS_2025", "List_2024"]:
    if col in df.columns:
        df[col] = numify(df[col])

# Base price (2025 list can be merged later; fallback to 2024)
df["List_2025"] = np.nan
df["BasePrice"] = df["List_2025"].fillna(df.get("List_2024"))

# Discount tiers
for pct in (15, 20, 25, 30, 35):
    df[f"Disc_{pct}%"] = (df["BasePrice"] * (1 - pct/100)).round(2)

# Profit metrics
df["COGS_for_calc"] = df.get("COGS_2025")
if "COGS_for_calc" not in df or df["COGS_for_calc"].isna().all():
    df["COGS_for_calc"] = df.get("COGS_2024")
df["Gross_Profit_at_Base"] = (df["BasePrice"] - df["COGS_for_calc"]).round(2)
df["Margin_%_at_Base"] = np.where(
    df["BasePrice"].gt(0),
    (df["Gross_Profit_at_Base"] / df["BasePrice"] * 100).round(2),
    np.nan
)

# Order: by Capacity, then Series D,E,F,A,HEL, then Model
order_map = {"D": 1, "E": 2, "F": 3, "A": 4, "HEL": 5}
df["_ord"] = df["Series"].map(order_map).fillna(99)
df = df.sort_values(["Capacity (kW)", "_ord", "Model"]).drop(columns="_ord").reset_index(drop=True)
df.insert(0, "SL#", range(1, len(df) + 1))

cols = [
    "SL#", "Model", "Capacity (kW)", "Series",
    "List_2024", "List_2025", "COGS_2024", "COGS_2025",
    "Gross_Profit_at_Base", "Margin_%_at_Base",
    "Disc_15%", "Disc_20%", "Disc_25%", "Disc_30%", "Disc_35%",
]

Path("outputs").mkdir(exist_ok=True)
out = Path("outputs/VFD_PRICE_LIST_2024_vs_2025_v02.csv")
df[[c for c in cols if c in df.columns]].to_csv(out, index=False, encoding="utf-8-sig")
print(f"Wrote {out.resolve()}")
