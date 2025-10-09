# make_scenarios.py
# Auto-generates Scenarios 2–4 for all city-quarter CSVs

import os, numpy as np, pandas as pd

# === CONFIG ===
INPUT_DIR = "../data_proc"          # where original clean files are
OUTPUT_DIR = "../data_processed"    # all outputs go here
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(42)

# list of all city-quarter combos
CITIES  = ["CHENNAI", "DELHI", "JAIPUR", "LEH"]
QUARTERS = ["Q1", "Q2", "Q3", "Q4"]

def add_noise(data, temp_noise=1, rel_noise=0.1, bias=0):
    noisy = data.copy()
    noisy["T2M"] = noisy["T2M"] + np.random.uniform(-temp_noise, temp_noise, len(noisy)) + bias
    for c in ["ALLSKY_SFC_SW_DWN","WS10M","PS","WSC"]:
        noisy[c] = noisy[c] * np.random.uniform(1-rel_noise, 1+rel_noise, len(noisy))
    return noisy

def drop_rows(data, frac):
    return data.sample(frac=1-frac, random_state=42).sort_index()

# process one file
def process_file(city, quarter):
    name = f"{city}_{quarter}.csv"
    path = os.path.join(INPUT_DIR, name)
    if not os.path.exists(path):
        print(f"⚠️ Skipping {name} — file not found")
        return
    print(f"Processing {name} ...")

    # read data (skip NASA header)
    df = pd.read_csv(path, skiprows=18)
    df.columns = ["YEAR","MO","DY","HR","ALLSKY_SFC_SW_DWN","T2M","WS10M","PS","WSC"]

    # scenario 2 — sparse
    sparse = drop_rows(df, 0.30)
    sparse = add_noise(sparse, temp_noise=1, rel_noise=0.10)
    sparse.to_csv(f"{OUTPUT_DIR}/{city.lower()}_{quarter.lower()}_sparse.csv", index=False)

    # scenario 3 — noisy
    noisy = add_noise(df, temp_noise=3, rel_noise=0.20, bias=0.5)
    noisy.to_csv(f"{OUTPUT_DIR}/{city.lower()}_{quarter.lower()}_noisy.csv", index=False)

    # scenario 4 — rural
    rural = drop_rows(df, 0.50)
    rural = add_noise(rural, temp_noise=1, rel_noise=0.10, bias=0.5)
    rural.to_csv(f"{OUTPUT_DIR}/{city.lower()}_{quarter.lower()}_rural.csv", index=False)

    print(f"✅ Done: {city} {quarter}")

# main loop
for city in CITIES:
    for q in QUARTERS:
        process_file(city, q)

print("\n🎯 All 16 files processed! Check data_processed/")
