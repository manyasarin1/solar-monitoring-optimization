import pandas as pd
import os

# --- Folders setup ---
CLEAN_DIR = "data_proc"           # clean NASA data
DOCTORED_DIR = "data_processed"   # sparse/noisy/rural data
OUTPUT_DIR = "features"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def add_features(path, file):
    """Reads a CSV, adds derived features Tp & eta, saves to features/ folder"""
    file_path = os.path.join(path, file)

    # Define expected NASA column names
    colnames = [
        "YEAR", "MO", "DY", "HR",
        "ALLSKY_SFC_SW_DWN",  # Solar radiation (MJ/m^2/day)
        "T2M",                # Air temperature (°C)
        "WS10M",              # Wind speed (m/s)
        "PS",                 # Surface pressure (kPa)
        "WSC"                 # Wind scaling/control
    ]

    # 🧩 Handle clean NASA data with header lines
    if "data_proc" in path:
        df = pd.read_csv(file_path, skiprows=18, names=colnames)
    else:
        df = pd.read_csv(file_path)

    # 🧠 Ensure key columns exist
    if not {"ALLSKY_SFC_SW_DWN", "T2M"}.issubset(df.columns):
        print(f"⚠️ Skipping {file} — missing required columns.")
        return

    # 🌡 Derive physical features
    df["Tp"] = df["T2M"] + 0.03 * (df["ALLSKY_SFC_SW_DWN"] / 10)
    df["eta"] = 0.18 * (1 - 0.0045 * (df["Tp"] - 25))
    df["eta"] = df["eta"].clip(0, 0.24)

    # 💾 Save processed data
    out_path = os.path.join(OUTPUT_DIR, file)
    df.to_csv(out_path, index=False)
    print(f"✅ done: {file}")


# --- Process all clean NASA data ---
print("\n🌞 Processing clean NASA data...")
for f in os.listdir(CLEAN_DIR):
    if f.endswith(".csv"):
        add_features(CLEAN_DIR, f)

# --- Process all doctored sparse/noisy/rural data ---
print("\n🌡 Processing sparse/noisy/rural data...")
for f in os.listdir(DOCTORED_DIR):
    if f.endswith(".csv"):
        add_features(DOCTORED_DIR, f)

print("\n🎯 All feature files saved in 'features/' ✅")
