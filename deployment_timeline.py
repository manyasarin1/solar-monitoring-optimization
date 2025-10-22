import matplotlib.pyplot as plt
import numpy as np

def chart1_deployment_timeline():
    """Traditional vs AI-Optimized Implementation Timeline"""
    phases = ['Planning', 'Infrastructure', 'Data\nCollection', 'Training', 'Deployment']
    traditional_timeline = [3, 6, 4, 5, 2]  # months
    ai_timeline = [2, 4, 3, 3, 1]  # months
    
    x = np.arange(len(phases))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(x - width/2, traditional_timeline, width, label='Traditional', color='#2E86AB', alpha=0.8)
    bars2 = plt.bar(x + width/2, ai_timeline, width, label='AI-Optimized', color='#A23B72', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.1, 
                    f'{height}', ha='center', va='bottom', fontweight='bold')
    
    plt.ylabel('Time (Months)', fontsize=12, fontweight='bold')
    plt.title('Deployment Timeline Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(x, phases)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 7)
    plt.tight_layout()
    plt.show()
    print("📅 AI reduces deployment time by 40% (20 → 12 months)")

def chart2_resource_utilization():
    """Resource Usage Efficiency Comparison"""
    resources = ['Manpower', 'Energy', 'Bandwidth', 'Storage']
    traditional_usage = [85, 70, 60, 75]  # %
    ai_usage = [45, 35, 30, 40]  # %
    
    x = np.arange(len(resources))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width/2, traditional_usage, width, label='Traditional', color='#2E86AB', alpha=0.8)
    bars2 = plt.bar(x + width/2, ai_usage, width, label='AI-Optimized', color='#A23B72', alpha=0.8)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 1, 
                    f'{height}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.ylabel('Resource Utilization (%)', fontsize=12, fontweight='bold')
    plt.title('Resource Usage Efficiency', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(x, resources)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()
    print("⚡ AI reduces resource usage by 45-50% across all categories")

def chart3_geographical_coverage():
    """Rural Area Coverage Impact"""
    metrics = ['Villages\nCovered', 'Population\nReached', 'Services\nDelivered']
    traditional_coverage = [45, 12000, 8]
    ai_coverage = [78, 25000, 15]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Bars for villages and services
    bars1 = ax1.bar(x - width/2, [traditional_coverage[0], 0, traditional_coverage[2]], 
                    width, label='Traditional', color='#2E86AB', alpha=0.8)
    bars2 = ax1.bar(x + width/2, [ai_coverage[0], 0, ai_coverage[2]], 
                    width, label='AI-Optimized', color='#A23B72', alpha=0.8)
    
    ax1.set_ylabel('Villages & Services', fontsize=12, fontweight='bold')
    ax1.set_title('Geographical Coverage Impact', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend(loc='upper left')
    ax1.grid(axis='y', alpha=0.3)
    
    # Second y-axis for population
    ax2 = ax1.twinx()
    ax2.plot(x, [traditional_coverage[1], 0, 0], 's-', color='#2E86AB', linewidth=3, markersize=8)
    ax2.plot(x, [ai_coverage[1], 0, 0], 's-', color='#A23B72', linewidth=3, markersize=8)
    ax2.set_ylabel('Population Reached', fontsize=12, fontweight='bold')
    
    # Add value labels
    for i, (trad, ai) in enumerate(zip(traditional_coverage, ai_coverage)):
        if i == 1:  # Population data
            ax1.text(i, max(trad, ai)/2, f'Trad: {trad:,}\nAI: {ai:,}', 
                    ha='center', va='center', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    print("🗺️ AI increases coverage: +73% villages, +108% population, +87% services")

def chart4_maintenance_reliability():
    """System Reliability & Maintenance Metrics"""
    reliability_metrics = ['Uptime %', 'Failure Rate', 'Response Time\n(hours)', 'Maintenance Cost\n($K/year)']
    traditional_scores = [85, 15, 48, 80]
    ai_scores = [95, 5, 12, 45]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    colors = ['#2E86AB', '#A23B72']
    labels = ['Traditional', 'AI-Optimized']
    
    for i, (metric, trad, ai) in enumerate(zip(reliability_metrics, traditional_scores, ai_scores)):
        data = [trad, ai]
        bars = axes[i].bar(labels, data, color=colors, alpha=0.8)
        axes[i].set_title(metric, fontweight='bold')
        axes[i].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, data):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(data)*0.05, 
                        f'{value}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('System Reliability & Maintenance Metrics', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()
    print("🔧 AI improves uptime to 95%, reduces failures by 67%, cuts response time by 75%")

def chart5_roi_timeline():
    """Return on Investment Over Time"""
    years = ['Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5']
    traditional_roi = [-20, -5, 10, 25, 40]  # % ROI
    ai_roi = [-15, 5, 30, 60, 95]  # % ROI
    
    plt.figure(figsize=(12, 6))
    plt.plot(years, traditional_roi, 'o-', linewidth=3, markersize=8, label='Traditional', color='#2E86AB')
    plt.plot(years, ai_roi, 's-', linewidth=3, markersize=8, label='AI-Optimized', color='#A23B72')
    
    # Add value labels on points
    for i, (yr, trad, ai) in enumerate(zip(years, traditional_roi, ai_roi)):
        plt.text(i, trad - 3, f'{trad}%', ha='center', va='top', fontweight='bold', color='#2E86AB')
        plt.text(i, ai + 3, f'{ai}%', ha='center', va='bottom', fontweight='bold', color='#A23B72')
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.ylabel('ROI (%)', fontsize=12, fontweight='bold')
    plt.title('Return on Investment Timeline', fontsize=14, fontweight='bold', pad=20)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-25, 110)
    plt.tight_layout()
    plt.show()
    print("💰 AI achieves break-even in Year 2 vs Year 3 for Traditional, 2.4x better ROI by Year 5")

def chart6_environmental_impact():
    """Environmental Sustainability Metrics"""
    environmental_metrics = ['Energy\nConsumption', 'Carbon\nFootprint', 'E-Waste', 'Renewable\nUsage']
    traditional_impact = [100, 100, 100, 30]  # baseline %
    ai_impact = [65, 55, 40, 75]  # baseline %
    
    x = np.arange(len(environmental_metrics))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width/2, traditional_impact, width, label='Traditional (Baseline)', color='#2E86AB', alpha=0.8)
    bars2 = plt.bar(x + width/2, ai_impact, width, label='AI-Optimized', color='#A23B72', alpha=0.8)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 2, 
                    f'{height}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.ylabel('Impact (% of Baseline)', fontsize=12, fontweight='bold')
    plt.title('Environmental Sustainability Impact', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(x, environmental_metrics)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 120)
    plt.tight_layout()
    plt.show()
    print("🌱 AI reduces environmental impact: -35% energy, -45% carbon, -60% e-waste, +150% renewable usage")

def chart7_service_quality():
    """Service Quality Metrics Comparison"""
    quality_metrics = ['Accuracy', 'Speed', 'Availability', 'User Satisfaction']
    traditional_quality = [6.5, 5.8, 7.2, 6.0]  # 0-10 scale
    ai_quality = [8.8, 8.2, 9.1, 8.5]  # 0-10 scale
    
    x = np.arange(len(quality_metrics))
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, traditional_quality, 'o-', linewidth=3, markersize=10, label='Traditional', color='#2E86AB')
    plt.plot(x, ai_quality, 's-', linewidth=3, markersize=10, label='AI-Optimized', color='#A23B72')
    
    # Add value labels
    for i, (trad, ai) in enumerate(zip(traditional_quality, ai_quality)):
        plt.text(i, trad - 0.3, f'{trad}', ha='center', va='top', fontweight='bold', color='#2E86AB')
        plt.text(i, ai + 0.3, f'{ai}', ha='center', va='bottom', fontweight='bold', color='#A23B72')
    
    plt.ylabel('Quality Score (0-10)', fontsize=12, fontweight='bold')
    plt.title('Service Quality Metrics', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(x, quality_metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(4, 10)
    plt.tight_layout()
    plt.show()
    print("⭐ AI improves service quality by 30-40% across all metrics")

def chart8_scalability_analysis():
    """Scaling Capabilities Comparison"""
    scale_factors = ['10 villages', '50 villages', '100 villages', '500 villages']
    traditional_costs = [100, 180, 320, 1200]  # cost units
    ai_costs = [100, 130, 180, 450]  # cost units
    
    plt.figure(figsize=(10, 6))
    plt.plot(scale_factors, traditional_costs, 'o-', linewidth=3, markersize=8, label='Traditional', color='#2E86AB')
    plt.plot(scale_factors, ai_costs, 's-', linewidth=3, markersize=8, label='AI-Optimized', color='#A23B72')
    
    # Add value labels
    for i, (scale, trad, ai) in enumerate(zip(scale_factors, traditional_costs, ai_costs)):
        plt.text(i, trad + 50, f'{trad}', ha='center', va='bottom', fontweight='bold', color='#2E86AB')
        plt.text(i, ai - 40, f'{ai}', ha='center', va='top', fontweight='bold', color='#A23B72')
    
    plt.ylabel('Relative Cost', fontsize=12, fontweight='bold')
    plt.title('Scaling Cost Analysis', fontsize=14, fontweight='bold', pad=20)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1400)
    plt.tight_layout()
    plt.show()
    print("📈 AI shows better scalability: 62% lower costs at 500 villages scale")

def main():
    while True:
        print("\n" + "="*60)
        print("🏞️ RURAL AI OPTIMIZATION - COMPREHENSIVE DASHBOARD")
        print("="*60)
        print("1.  Deployment Timeline Comparison")
        print("2.  Resource Utilization Efficiency") 
        print("3.  Geographical Coverage Impact")
        print("4.  Maintenance & Reliability Metrics")
        print("5.  ROI Timeline Analysis")
        print("6.  Environmental Impact")
        print("7.  Service Quality Metrics")
        print("8.  Scalability Analysis")
        print("9.  Exit")
        print("-"*60)
        
        choice = input("Select chart (1-9): ").strip()
        
        chart_functions = {
            '1': chart1_deployment_timeline,
            '2': chart2_resource_utilization,
            '3': chart3_geographical_coverage,
            '4': chart4_maintenance_reliability,
            '5': chart5_roi_timeline,
            '6': chart6_environmental_impact,
            '7': chart7_service_quality,
            '8': chart8_scalability_analysis
        }
        
        if choice in chart_functions:
            chart_functions[choice]()
        elif choice == '9':
            print("Thank you for using the Rural AI Dashboard! 👋")
            break
        else:
            print("Invalid choice! Please enter 1-9")

if __name__ == "__main__":
    main()