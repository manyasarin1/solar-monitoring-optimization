import os
import pandas as pd

print("🔍 CHECKING PINNS LOCATION:")
print("="*40)

# Check the exact path you mentioned
pinns_paths = [
    'pinn/analysis_result/pin_summary_metrics.csv',
    'pinn/analysis_result/pinn_summary_metrics.csv',
    'pinn/analysis_results/pin_summary_metrics.csv', 
    'pinn/analysis_results/pinn_summary_metrics.csv'
]

for path in pinns_paths:
    if os.path.exists(path):
        print(f"✅ FOUND PINNS: {path}")
        pinns_df = pd.read_csv(path)
        print(f"   📊 Samples: {len(pinns_df)}")
        print(f"   📋 Columns: {list(pinns_df.columns)}")
        print(f"   🔢 First 3 rows:")
        print(pinns_df.head(3).to_string())
        break
else:
    print("❌ PINNs not found in specified location")
    print("📁 Let me check what's in pinn folder:")
    if os.path.exists('pinn'):
        for root, dirs, files in os.walk('pinn'):
            for file in files:
                if 'summary' in file.lower() or 'metric' in file.lower():
                    print(f"   📄 {os.path.join(root, file)}")

# Also search for ML results
print(f"\n🔍 Searching for ML results...")
ml_files = []
for root, dirs, files in os.walk('.'):
    for file in files:
        if any(keyword in file.lower() for keyword in ['scenario', 'test', 'result', 'rmse']):
            if file.endswith(('.json', '.csv')):
                ml_files.append(os.path.join(root, file))

print("📊 Potential ML result files:")
for file in ml_files[:10]:  # Show first 10
    print(f"   📄 {file}")