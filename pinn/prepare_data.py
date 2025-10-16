import os
import pandas as pd
import numpy as np

# ============================
# CONFIGURATION
# ============================
DATA_DIR = "../features"        # where your clean/noisy/sparse/rural CSVs are
OUTPUT_DIR = "./data_ready"     # where NPZ files will be saved
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🔍 Scanning folders for all city–quarter–scenario datasets...\n")

# ============================
# TARGET COLUMN NAMES (order)
# ============================
# [Solar, Temp, Wind, Pressure, PanelTemp, Efficiency]
EXPECTED_COLS = ["SW", "T2M", "WS10M", "PS", "Tp", "eta"]

# ============================
# MAIN LOOP
# ============================
for fname in os.listdir(DATA_DIR):
    if not fname.endswith(".csv"):
        continue

    fpath = os.path.join(DATA_DIR, fname)
    df = pd.read_csv(fpath, skiprows=18, header=None)  # treat all rows as data

    # Clean up empty columns
    df = df.dropna(axis=1, how='all')
    df = df.select_dtypes(include=[np.number])

    if df.shape[1] < 6:
        print(f"⚠️ Not enough usable columns in {fname}")
        continue

    # Rename first 6 columns to standard names
    df = df.iloc[:, :6]
    df.columns = EXPECTED_COLS

    # Split into inputs and outputs
    X = df.iloc[:, :4].to_numpy(dtype=np.float32)
    y = df.iloc[:, 4:].to_numpy(dtype=np.float32)

    outname = fname.replace(".csv", ".npz")
    outpath = os.path.join(OUTPUT_DIR, outname)
    np.savez_compressed(outpath, X=X, y=y)

    print(f"✅ Saved {outname}  →  {outpath}")

print("\n🎉 All clean + simulated datasets converted successfully — check pinn/data_ready/")
