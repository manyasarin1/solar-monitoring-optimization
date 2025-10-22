import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import pi

def create_radar_chart():
    """Create radar chart comparing all methods across multiple metrics"""
    print("📊 Creating Radar Chart...")
    
    # Methods to compare
    methods = ['Random Forest', 'XGBoost', 'PINNs', 'FDM']
    
    # Metrics (normalized 0-1, higher is better)
    categories = ['Accuracy', 'Speed', 'Robustness', 'Cost Efficiency', 'Ease of Implementation']
    
    # Scores for each method (based on our analysis)
    scores = {
        'Random Forest': [1.0, 0.9, 1.0, 0.8, 0.7],
        'XGBoost': [0.8, 0.7, 0.6, 0.7, 0.6],
        'PINNs': [0.5, 0.4, 0.5, 0.5, 0.3],
        'FDM': [0.2, 0.3, 0.3, 0.6, 0.8]
    }
    
    # Set up radar chart
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Compute angles for each category
    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]  # Complete the circle
    
    # Plot each method
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    for idx, (method, values) in enumerate(scores.items()):
        values += values[:1]  # Complete the circle
        ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])
    
    # Add category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True)
    
    plt.title('Comprehensive Method Comparison\n(Higher Scores = Better Performance)', 
              size=14, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    plt.savefig('RADAR_CHART_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_scenario_heatmap():
    """Create heatmap showing performance across scenarios"""
    print("🔥 Creating Scenario Heatmap...")
    
    scenarios = ['Clean', 'Noisy', 'Sparse', 'Rural']
    methods = ['Random Forest', 'XGBoost', 'PINNs', 'FDM']
    
    # RMSE values for each scenario (lower is better)
    performance_data = [
        [0.022, 0.085, 0.55, 2.4],    # Clean
        [0.024, 0.135, 0.68, 2.9],    # Noisy
        [0.030, 0.157, 0.72, 3.1],    # Sparse
        [0.027, 0.155, 0.70, 3.0]     # Rural
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(performance_data, cmap='RdYlGn_r')  # Reversed colormap (red=bad, green=good)
    
    # Add text annotations
    for i in range(len(scenarios)):
        for j in range(len(methods)):
            text = ax.text(j, i, f'{performance_data[i][j]:.3f}°C',
                          ha="center", va="center", color="black", fontweight='bold')
    
    # Customize plot
    ax.set_xticks(np.arange(len(methods)))
    ax.set_yticks(np.arange(len(scenarios)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_yticklabels(scenarios)
    ax.set_xlabel('Methods')
    ax.set_ylabel('Scenarios')
    ax.set_title('RMSE Performance Across Scenarios\n(Lower Values = Better)', 
                 fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('RMSE (°C)', rotation=-90, va="bottom")
    
    plt.tight_layout()
    plt.savefig('SCENARIO_HEATMAP.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_economic_comparison():
    """Create bar chart comparing economic benefits"""
    print("💰 Creating Economic Comparison...")
    
    cities = ['Chennai', 'Delhi', 'Jaipur', 'Leh']
    energy_kwh = [207.60, 226.30, 218.45, 253.62]
    cost_savings = [1660.83, 1810.41, 1747.56, 2028.98]
    viability = [93.17, 93.49, 93.14, 95.95]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Energy production
    bars1 = ax1.bar(cities, energy_kwh, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.8)
    ax1.set_ylabel('Energy Production (kWh)')
    ax1.set_title('Energy Production by City', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    for bar, value in zip(bars1, energy_kwh):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f'{value:.1f}',
                ha='center', va='bottom', fontweight='bold')
    
    # Cost savings
    bars2 = ax2.bar(cities, cost_savings, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.8)
    ax2.set_ylabel('Cost Savings (₹)')
    ax2.set_title('Cost Savings by City', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    for bar, value in zip(bars2, cost_savings):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, f'₹{value:.0f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('ECONOMIC_COMPARISON.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_performance_ranking():
    """Create visual ranking of methods"""
    print("🏆 Creating Performance Ranking...")
    
    methods = ['FDM', 'PINNs', 'XGBoost', 'Random Forest']  # Worst to best
    rmse_scores = [3.262, 0.661, 0.149, 0.027]
    colors = ['#C73E1D', '#F18F01', '#A23B72', '#2E86AB']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Horizontal bar chart
    bars = ax.barh(methods, rmse_scores, color=colors, alpha=0.8)
    
    # Add value labels
    for bar, value in zip(bars, rmse_scores):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}°C', va='center', fontweight='bold', fontsize=12)
    
    ax.set_xlabel('RMSE (°C) - Lower is Better', fontsize=12)
    ax.set_title('Method Performance Ranking\nSolar Panel Temperature Prediction', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add performance labels
    performance_labels = ['4th - Traditional', '3rd - Physics-Based', '2nd - Good ML', '1st - Best ML']
    for i, (bar, label) in enumerate(zip(bars, performance_labels)):
        ax.text(-0.5, bar.get_y() + bar.get_height()/2, label, 
                va='center', ha='left', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('PERFORMANCE_RANKING.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Create all presentation graphs"""
    print("🚀 CREATING PRESENTATION-READY GRAPHS")
    print("="*50)
    
    create_radar_chart()
    create_scenario_heatmap()
    create_economic_comparison()
    create_performance_ranking()
    
    print(f"\n✅ PRESENTATION GRAPHS COMPLETE!")
    print("📁 Files created:")
    print("   - RADAR_CHART_comparison.png")
    print("   - SCENARIO_HEATMAP.png") 
    print("   - ECONOMIC_COMPARISON.png")
    print("   - PERFORMANCE_RANKING.png")
    print(f"\n🎯 Your presentation is now READY with 4 professional graphs!")

if __name__ == "__main__":
    main()