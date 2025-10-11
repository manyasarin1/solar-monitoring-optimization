import os
import numpy as np
import pandas as pd
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(BASE_DIR, "../features")
DATA_PROC_DIR = os.path.join(BASE_DIR, "../data_proc")
OUTPUT_DIR = os.path.join(BASE_DIR, "data_ready")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🔍 Scanning folders for all city–quarter–scenario datasets...\n")

cities = ["chennai", "delhi", "jaipur", "leh"]
quarters = ["q1", "q2", "q3", "q4"]
scenarios = ["clean", "sparse", "noisy", "rural"]

def read_flexible_csv(fpath):
    """Auto-detects delimiter, skips NASA header, and normalizes column names."""
    # Find where the header starts
    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    start_line = 0
    for i, line in enumerate(lines):
        if "YEAR" in line:
            start_line = i
            break

    # Try comma or whitespace separator
    try:
        df = pd.read_csv(fpath, skiprows=start_line, sep=",")
        if len(df.columns) == 1:
            df = pd.read_csv(fpath, skiprows=start_line, delim_whitespace=True)
    except Exception:
        df = pd.read_csv(fpath, skiprows=start_line, delim_whitespace=True)

    # Clean column names — remove units and brackets
    df.columns = [re.sub(r"\s*\[.*?\]", "", c).strip() for c in df.columns]
    return df

for city in cities:
    for q in quarters:
        for s in scenarios:
            fname = f"{city}_{q}_{s}.csv"
            fpaths = [
                os.path.join(FEATURES_DIR, fname),
                os.path.join(DATA_PROC_DIR, fname),
            ]
            fpath = next((fp for fp in fpaths if os.path.exists(fp)), None)
            if not fpath:
                print(f"❌ Missing: {fname}")
                continue

            try:
                df = read_flexible_csv(fpath)
            except Exception as e:
                print(f"⚠️ Error reading {fname}: {e}")
                continue

            needed_cols = ["ALLSKY_SFC_SW_DWN", "T2M", "WS10M", "PS", "WSC"]
            cols = [c for c in needed_cols if c in df.columns]

            if not cols:
                print(f"⚠️ Missing usable columns in {fname}")
                print("   → Found columns:", df.columns.tolist()[:10])
                continue

            arr = df[cols].to_numpy(dtype=np.float32)
            outname = fname.replace(".csv", ".npz")
            outpath = os.path.join(OUTPUT_DIR, outname)
            np.savez_compressed(outpath, data=arr)
            print(f"✅ Saved {outname} → from {fname}")

print("\n🎉 All clean + simulated datasets converted — check pinn/data_ready/")
