# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd, numpy as np, re

# ----------- FILES -----------
ROOT = Path(".")
COGS_2024 = ROOT / "VFD_COGS_2024_Last.csv"
COGS_2025 = ROOT / "VFD_COGS_2025_Final.csv"
LIST_2024 = ROOT / "VFD_List_Price_2024_Final.csv"
LIST_2025 = ROOT / "VFD_List_Price_2025_Final.csv"   # used for ORDER + pricing

# ----------- HELPERS -----------
MODEL_RX = re.compile(r"(FR-[A-Z0-9\-]+-[0-9]+(?:\.[0-9]+)?K)", re.I)

def read_smart(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, engine="python", sep=None, dtype=str, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    return df

def numify(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")

def guess_model_col(df: pd.DataFrame):
    # obvious header names first
    for name in df.columns:
        key = name.lower().replace(" ","")
        if key in {"model","modelname","modelno","modelnumber","model_name","model(ratedcurrent)"}:
            return name
    # fallback: column with most FR-...K matches
    best = (0.0, None)
    for name in df.columns:
        ratio = df[name].astype(str).str.contains(MODEL_RX, na=False).mean()
        if ratio > best[0]:
            best = (ratio, name)
    return best[1] if best[0] >= 0.05 else df.columns[0]  # last resort

def extract_all_models(text: str):
    """Return a list of FR-...K codes found left->right (pair-aware)."""
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    text_up = text.upper()
    hits = [m for m in MODEL_RX.findall(text_up)]
    if not hits and " / " in text_up:
        # crude split fallback
        hits = [s.strip() for s in text_up.split(" / ") if s.strip().startswith("FR-")]
    return hits

def extract_series(model: str):
    m = re.search(r"FR-([A-Z]+)", str(model).upper())
    return m.group(1) if m else None

def extract_kw(model: str):
    m = re.search(r"-([0-9]+(?:\.[0-9]+)?)K", str(model).upper())
    return float(m.group(1)) if m else np.nan

def find_col(df: pd.DataFrame, *keywords):
    best = (0, None)
    for name in df.columns:
        key = name.lower().replace(" ", "")
        score = sum(k in key for k in keywords)
        if score > best[0]:
            best = (score, name)
    return best[1] if best[0] > 0 else None

# ----------- LOAD DATA -----------
c24 = read_smart(COGS_2024)
c25 = read_smart(COGS_2025)
lp24 = read_smart(LIST_2024)
lp25 = read_smart(LIST_2025)

# Normalize model column to "Model" for these tables (merges will use it)
for name, df in [("COGS_2024", c24), ("COGS_2025", c25), ("LIST_2024", lp24), ("LIST_2025", lp25)]:
    mcol = guess_model_col(df)
    if mcol != "Model":
        df.rename(columns={mcol: "Model"}, inplace=True)
    df["Model"] = df["Model"].astype(str).str.strip()

# Build the universe of models
models = sorted(set(c24["Model"]) | set(c25["Model"]) | set(lp24["Model"]) | set(lp25["Model"]))
df = pd.DataFrame({"Model": models})
df["Series"] = df["Model"].map(extract_series)
df["Capacity (kW)"] = df["Model"].map(extract_kw)

# Bring numeric columns where present
c24_cogs = find_col(c24, "cogs","cost")
c25_cogs = find_col(c25, "cogs","cost")
lp24_price = find_col(lp24, "list","price","lp","bdt")
lp25_price = find_col(lp25, "list","price","lp","bdt")

if c24_cogs: df = df.merge(c24[["Model", c24_cogs]].rename(columns={c24_cogs:"COGS_2024"}), on="Model", how="left")
if c25_cogs: df = df.merge(c25[["Model", c25_cogs]].rename(columns={c25_cogs:"COGS_2025"}), on="Model", how="left")
if lp24_price: df = df.merge(lp24[["Model", lp24_price]].rename(columns={lp24_price:"List_2024"}), on="Model", how="left")
if lp25_price: df = df.merge(lp25[["Model", lp25_price]].rename(columns={lp25_price:"List_2025"}), on="Model", how="left")

# Numeric conversion
for col in ["COGS_2024","COGS_2025","List_2024","List_2025"]:
    if col in df.columns:
        df[col] = numify(df[col])

# Base price: prefer 2025 (if present), else 2024
df["BasePrice"] = df.get("List_2025").where(df.get("List_2025").notna(), df.get("List_2024"))

# Discounts (15–35%)
for pct in (15,20,25,30,35):
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

# ----------- ORDERING FROM 2025 LIST (PAIR-AWARE) -----------
# Build an order sequence by reading *all* models in the 2025 Model column (left->right)
lp25_model_col = guess_model_col(lp25)
seen = set(); order_keys = []
for val in lp25[lp25_model_col].fillna("").astype(str):
    for code in extract_all_models(val):
        if code and code not in seen:
            seen.add(code)
            order_keys.append(code)
order_index = {k:i for i,k in enumerate(order_keys)}

# Map the df's Model to that order. If a Model doesn't appear in 2025 list, fall back later.
df["_order_idx"] = df["Model"].map(order_index).astype("float")

# Secondary fallbacks: by capacity, then by Series order D->E->F->A->HEL, then Model
series_rank = {"D":1,"E":2,"F":3,"A":4,"HEL":5}
df["_series_rank"] = df["Series"].map(series_rank).fillna(99)
df["_cap"] = df["Capacity (kW)"]
df["_model"] = df["Model"]

df = df.sort_values(by=["_order_idx","_cap","_series_rank","_model"], na_position="last").drop(columns=["_order_idx","_cap","_series_rank","_model"])

# Optional sanity print
matched = df["Model"].isin(order_index).sum()
print(f"[ordering] matched {matched} of {len(df)} models to 2025 list order.")

# ----------- OUTPUT (no SL#) -----------
cols = ["Model","Capacity (kW)","Series",
        "List_2024","List_2025","COGS_2024","COGS_2025",
        "Gross_Profit_at_Base","Margin_%_at_Base",
        "Disc_15%","Disc_20%","Disc_25%","Disc_30%","Disc_35%"]
out = ROOT / "outputs" / "VFD_PRICE_LIST_2024_vs_2025_v03.csv"
out.parent.mkdir(exist_ok=True)
df[[c for c in cols if c in df.columns]].to_csv(out, index=False, encoding="utf-8-sig")
print(f"Wrote {out.resolve()}")
