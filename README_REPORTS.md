# Automated VFD Price Report

This repo generates **SISL_VFD_PA_2024 vs 2025_v01** (CSV + Excel) from the 2024 and 2025 CSVs.

## Place source CSVs
Put your files here (names can vary, but must include the year in the filename):
```
data/
  ...2024...csv
  ...2025...csv
```

## Run locally
```bash
pip install -r requirements.txt
python tools/generate_report.py              # auto-detects data/*2024*.csv and *2025*.csv
# or specify explicit files
python tools/generate_report.py --y2024 data/your_2024.csv --y2025 data/your_2025.csv
```

Outputs are saved to:
```
reports/SISL_VFD_PA_2024 vs 2025_v01.csv
reports/SISL_VFD_PA_2024 vs 2025_v01.xlsx
```

## GitHub Actions (automatic)
- Triggers when you push changes to `data/` or `tools/`
- You can also run it manually (**Actions → Build VFD Price Report → Run workflow**)
- Runs every day at **09:00 Asia/Dhaka** (03:00 UTC)
- Uses the built-in `GITHUB_TOKEN` with `contents: write` to commit artifacts back to the repo
