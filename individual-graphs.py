import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import pi

def create_heatmap_only():
    """Create ONLY the scenario heatmap"""
    print("🔥 Creating Scenario Heatmap...")
    
    scenarios = ['Clean', 'Noisy', 'Sparse', 'Rural']
    methods = ['Random Forest', 'XGBoost', 'PINNs', 'FDM']
    
    # RMSE values for each scenario
    performance_data = [
        [0.022, 0.085, 0.55, 2.4],    # Clean
        [0.024, 0.135, 0.68, 2.9],    # Noisy
        [0.030, 0.157, 0.72, 3.1],    # Sparse
        [0.027, 0.155, 0.70, 3.0]     # Rural
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(performance_data, cmap='RdYlGn_r')
    
    for i in range(len(scenarios)):
        for j in range(len(methods)):
            ax.text(j, i, f'{performance_data[i][j]:.3f}°C',
                   ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_xticks(np.arange(len(methods)))
    ax.set_yticks(np.arange(len(scenarios)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_yticklabels(scenarios)
    ax.set_xlabel('Methods')
    ax.set_ylabel('Scenarios')
    ax.set_title('RMSE Performance Across Scenarios\n(Lower Values = Better)', fontweight='bold')
    
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('RMSE (°C)', rotation=-90, va="bottom")
    
    plt.tight_layout()
    plt.savefig('SCENARIO_HEATMAP.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Heatmap saved as SCENARIO_HEATMAP.png")

def create_economic_only():
    """Create ONLY the economic comparison"""
    print("💰 Creating Economic Comparison...")
    
    cities = ['Chennai', 'Delhi', 'Jaipur', 'Leh']
    energy_kwh = [207.60, 226.30, 218.45, 253.62]
    cost_savings = [1660.83, 1810.41, 1747.56, 2028.98]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Energy production
    bars1 = ax1.bar(cities, energy_kwh, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax1.set_ylabel('Energy Production (kWh)')
    ax1.set_title('Energy Production by City', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    for bar, value in zip(bars1, energy_kwh):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f'{value:.1f}',
                ha='center', va='bottom', fontweight='bold')
    
    # Cost savings
    bars2 = ax2.bar(cities, cost_savings, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax2.set_ylabel('Cost Savings (₹)')
    ax2.set_title('Cost Savings by City', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    for bar, value in zip(bars2, cost_savings):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, f'₹{value:.0f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('ECONOMIC_COMPARISON.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Economic comparison saved as ECONOMIC_COMPARISON.png")

def create_ranking_only():
    """Create ONLY the performance ranking"""
    print("🏆 Creating Performance Ranking...")
    
    methods = ['FDM', 'PINNs', 'XGBoost', 'Random Forest']
    rmse_scores = [3.262, 0.661, 0.149, 0.027]
    colors = ['#C73E1D', '#F18F01', '#A23B72', '#2E86AB']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.barh(methods, rmse_scores, color=colors)
    
    for bar, value in zip(bars, rmse_scores):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}°C', va='center', fontweight='bold', fontsize=12)
    
    ax.set_xlabel('RMSE (°C) - Lower is Better', fontsize=12)
    ax.set_title('Method Performance Ranking\nSolar Panel Temperature Prediction', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    performance_labels = ['4th - Traditional', '3rd - Physics-Based', '2nd - Good ML', '1st - Best ML']
    for i, (bar, label) in enumerate(zip(bars, performance_labels)):
        ax.text(-0.5, bar.get_y() + bar.get_height()/2, label, 
                va='center', ha='left', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('PERFORMANCE_RANKING.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Performance ranking saved as PERFORMANCE_RANKING.png")

def main():
    """Run each graph separately"""
    print("Which graph do you want to create?")
    print("1. Scenario Heatmap")
    print("2. Economic Comparison") 
    print("3. Performance Ranking")
    print("4. ALL GRAPHS")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == '1':
        create_heatmap_only()
    elif choice == '2':
        create_economic_only()
    elif choice == '3':
        create_ranking_only()
    elif choice == '4':
        create_heatmap_only()
        create_economic_only() 
        create_ranking_only()
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()