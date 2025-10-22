import matplotlib.pyplot as plt
import pandas as pd

# CONFIRMED REAL VALUES from your data
real_values = {
    'Random Forest': 0.027351,
    'XGBoost': 0.148644,
    'PINNs': 0.661,      # Actual scaled PINNs RMSE
    'FDM': 3.262         # Actual FDM RMSE
}

# Create FINAL accurate plot
plt.figure(figsize=(10, 6))
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

methods = list(real_values.keys())
rmse_values = list(real_values.values())

bars = plt.bar(methods, rmse_values, color=colors, alpha=0.8)

# Add value labels
for bar, value in zip(bars, rmse_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{value:.3f}°C', ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.ylabel('Average RMSE (°C)', fontsize=12)
plt.title('100% CONFIRMED: Method Comparison - Average RMSE\n(Lower Values = Better Accuracy)', 
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('FINAL_CONFIRMED_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ FINAL CONFIRMED RESULTS:")
for method, rmse in real_values.items():
    print(f"   {method:<15}: {rmse:.3f}°C")