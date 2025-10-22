import matplotlib.pyplot as plt
import numpy as np

def chart1_traditional_vs_ai():
    """Traditional vs AI-Optimized Prediction Error"""
    methods = ['Traditional\n(FDW)', 'AI-Optimized\n(RF)']
    prediction_error = [3.5, 1.0]  # °C
    colors = ['#2E86AB', '#A23B72']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, prediction_error, color=colors, width=0.6)

    for i, (bar, value) in enumerate(zip(bars, prediction_error)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                 f'{value}°C', ha='center', va='bottom', fontweight='bold', fontsize=12)

    plt.ylabel('Prediction Error (°C)', fontsize=12, fontweight='bold')
    plt.title('Traditional vs AI-Optimized Prediction Error', fontsize=14, fontweight='bold', pad=20)
    plt.ylim(0, 4.5)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def chart2_traditional_performance():
    """Traditional Performance: Consistent Gains Year-Round"""
    categories = ['General', 'Data', 'Jupyter', 'Lst']
    efficiency_improvement = [12, 16, 18, 20]  # %
    colors = ['#2E86AB', '#2E86AB', '#2E86AB', '#2E86AB']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, efficiency_improvement, color=colors, width=0.6)

    for bar, value in zip(bars, efficiency_improvement):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 f'{value}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.ylabel('Efficiency Improvement (%)', fontsize=12, fontweight='bold')
    plt.title('Traditional Performance: Consistent Gains Year-Round', fontsize=14, fontweight='bold', pad=20)
    plt.ylim(0, 25)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def chart3_rural_deployment():
    """Rural Deployment Made Affordable"""
    methods = ['Traditional', 'AI-Optimized']
    costs = [180000, 140000]  # USD
    colors = ['#2E86AB', '#A23B72']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, costs, color=colors, width=0.6)

    for bar, value in zip(bars, costs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000, 
                 f'${value:,}', ha='center', va='bottom', fontweight='bold', fontsize=12)

    plt.ylabel('Implementation Cost (USD)', fontsize=12, fontweight='bold')
    plt.title('Rural Deployment Made Affordable', fontsize=14, fontweight='bold', pad=20)
    plt.ylim(0, 200000)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def chart4_overall_impact():
    """Overall Impact: Comprehensive Rural Improvement"""
    metrics = ['Accuracy', 'Cost', 'Efficiency', 'Viability']
    traditional_scores = [6, 8, 7, 6]  # Estimated from radar chart
    ai_scores = [9, 5, 8, 8]  # Estimated from radar chart

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width/2, traditional_scores, width, label='Traditional', color='#2E86AB', alpha=0.8)
    bars2 = plt.bar(x + width/2, ai_scores, width, label='AI-Optimized', color='#A23B72', alpha=0.8)

    plt.ylabel('Performance Score (0-10)', fontsize=12, fontweight='bold')
    plt.title('Overall Impact: Comprehensive Rural Improvement', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(x, metrics)
    plt.legend()
    plt.ylim(0, 10)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    while True:
        print("\n" + "="*50)
        print("📊 RURAL AI OPTIMIZATION DASHBOARD")
        print("="*50)
        print("1. Traditional vs AI-Optimized Prediction Error")
        print("2. Traditional Performance Gains")
        print("3. Rural Deployment Costs")
        print("4. Overall Impact Comparison")
        print("5. Exit")
        print("-"*50)
        
        choice = input("Select chart (1-5): ").strip()
        
        if choice == '1':
            chart1_traditional_vs_ai()
        elif choice == '2':
            chart2_traditional_performance()
        elif choice == '3':
            chart3_rural_deployment()
        elif choice == '4':
            chart4_overall_impact()
        elif choice == '5':
            print("Goodbye! 👋")
            break
        else:
            print("Invalid choice! Please enter 1-5")

if __name__ == "__main__":
    main()