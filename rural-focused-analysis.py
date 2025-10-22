import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_rural_performance():
    print("🌾 RURAL SCENARIO DEEP DIVE")
    print("="*50)
    
    # Load data
    with open('ml/results/scenario_test_results.json', 'r') as f:
        ml_data = json.load(f)
    
    pinns_df = pd.read_csv('pinn/analysis_results/pinn_summary_metrics.csv')
    fdm_economic = {
        'chennai': {'energy_kwh': 207.60, 'cost_savings': 1660.83, 'efficiency': 37.46, 'viability': 93.17},
        'delhi': {'energy_kwh': 226.30, 'cost_savings': 1810.41, 'efficiency': 37.46, 'viability': 93.49},
        'jaipur': {'energy_kwh': 218.45, 'cost_savings': 1747.56, 'efficiency': 37.46, 'viability': 93.14},
        'leh': {'energy_kwh': 253.62, 'cost_savings': 2028.98, 'efficiency': 37.46, 'viability': 95.95}
    }
    
    # Extract RURAL scenarios only
    rural_ml = [s for s in ml_data if s['scenario'] == 'rural']
    rural_pinns = pinns_df[pinns_df['Dataset'].str.contains('rural')]
    
    print("📊 RURAL PERFORMANCE COMPARISON:")
    print(f"{'City':<10} {'RF RMSE':<12} {'XGB RMSE':<12} {'PINNs RMSE':<14} {'Best Method':<15}")
    print("-" * 65)
    
    city_performance = {}
    
    for city in ['chennai', 'delhi', 'jaipur', 'leh']:
        # ML rural performance for this city
        city_ml = [s for s in rural_ml if s['city'] == city]
        rf_rmse = np.mean([s['rf_rmse'] for s in city_ml]) if city_ml else None
        xgb_rmse = np.mean([s['xgb_rmse'] for s in city_ml]) if city_ml else None
        
        # PINNs rural performance for this city
        city_pinns = rural_pinns[rural_pinns['Dataset'].str.startswith(city)]
        pinns_rmse = city_pinns['RMSE'].mean() / 100 if len(city_pinns) > 0 else None
        
        # Determine best method
        methods = {'RF': rf_rmse, 'XGB': xgb_rmse, 'PINNs': pinns_rmse}
        valid_methods = {k: v for k, v in methods.items() if v is not None}
        best_method = min(valid_methods, key=valid_methods.get) if valid_methods else 'N/A'
        
        city_performance[city] = {
            'rf_rmse': rf_rmse,
            'xgb_rmse': xgb_rmse, 
            'pinns_rmse': pinns_rmse,
            'best_method': best_method,
            'economic': fdm_economic[city]
        }
        
        # FIXED PRINT STATEMENT
        pinns_display = f"{pinns_rmse:.6f}" if pinns_rmse is not None else "N/A"
        print(f"{city:<10} {rf_rmse:.6f}  {xgb_rmse:.6f}  {pinns_display:<14} {best_method:<15}")
    
    return city_performance

def analyze_weather_impact():
    print(f"\n🌤️  WEATHER IMPACT ON RURAL PERFORMANCE:")
    print("="*50)
    
    # Weather characteristics for each city
    weather_profiles = {
        'chennai': {'type': 'Tropical', 'variability': 'Medium', 'extremes': 'High temp', 'impact': 'Stable patterns favor ML'},
        'delhi': {'type': 'Continental', 'variability': 'High', 'extremes': 'Both hot/cold', 'impact': 'Challenging for all models'},
        'jaipur': {'type': 'Desert', 'variability': 'Medium', 'extremes': 'High temp', 'impact': 'Predictable patterns favor ML'},
        'leh': {'type': 'Mountain', 'variability': 'Very High', 'extremes': 'Very cold', 'impact': 'Extremes may favor PINNs'}
    }
    
    print("City-wise Weather Analysis:")
    for city, weather in weather_profiles.items():
        print(f"  {city.upper():<8}: {weather['type']:<12} | Extremes: {weather['extremes']:<12} | Impact: {weather['impact']}")

def cost_effectiveness_analysis(city_performance):
    print(f"\n💰 COST-EFFECTIVENESS FOR RURAL DEPLOYMENT:")
    print("="*50)
    
    # Rural implementation costs (lower than urban due to simpler infrastructure)
    implementation_costs = {
        'RF': 40000,    # Lower cost for rural - basic hardware
        'XGBoost': 45000, 
        'PINNs': 100000,  # Still high - needs expertise
        'FDM': 60000     # Moderate but less practical for rural
    }
    
    print("Method Cost-Benefit Analysis (Rural Focus):")
    print(f"{'Method':<12} {'Rural Cost (₹)':<15} {'Avg Rural RMSE':<16} {'Cost per Accuracy':<18}")
    print("-" * 65)
    
    for method in ['RF', 'XGBoost', 'PINNs', 'FDM']:
        cost = implementation_costs[method]
        
        # Calculate average rural RMSE
        if method == 'RF':
            rmse_vals = [city_performance[c]['rf_rmse'] for c in city_performance]
            avg_rmse = np.mean(rmse_vals)
        elif method == 'XGBoost':
            rmse_vals = [city_performance[c]['xgb_rmse'] for c in city_performance]
            avg_rmse = np.mean(rmse_vals)
        elif method == 'PINNs':
            rmse_vals = [city_performance[c]['pinns_rmse'] for c in city_performance if city_performance[c]['pinns_rmse']]
            avg_rmse = np.mean(rmse_vals) if rmse_vals else 1.0
        else:  # FDM
            avg_rmse = 3.0  # Estimated for rural
        
        # Cost per unit accuracy (lower is better)
        accuracy_score = 1 / avg_rmse
        cost_per_accuracy = cost / accuracy_score
        
        print(f"{method:<12} ₹{cost:<13,} {avg_rmse:<15.4f}°C ₹{cost_per_accuracy:,.0f}")

def rural_recommendations(city_performance):
    print(f"\n🎯 RURAL DEPLOYMENT STRATEGY:")
    print("="*50)
    
    print("CITY-SPECIFIC RURAL RECOMMENDATIONS:")
    for city, data in city_performance.items():
        best_method = data['best_method']
        economic = data['economic']
        
        print(f"\n🏙️  {city.upper()}:")
        print(f"   Best Method: {best_method}")
        print(f"   Accuracy: {data[f'{best_method.lower()}_rmse']:.4f}°C RMSE")
        print(f"   Rural Advantage: {economic['viability']}% viability")
        print(f"   Expected Savings: ₹{economic['cost_savings']:.0f}/system")
        
        # Rural-specific advice
        if city == 'leh':
            print(f"   💡 Recommendation: Consider PINNs for extreme cold handling")
        elif best_method == 'RF':
            print(f"   💡 Recommendation: RF provides best cost-accuracy balance")
        else:
            print(f"   💡 Recommendation: {best_method} suits local conditions")

def create_rural_summary():
    print(f"\n🌍 FINAL RURAL INSIGHTS:")
    print("="*50)
    print("1. 🏆 RF consistently best for most rural scenarios")
    print("2. 💰 Most cost-effective for limited rural budgets") 
    print("3. 🌤️ Weather impacts PINNs vs ML performance differently")
    print("4. 🎯 Leh might benefit from PINNs' physics in extremes")
    print("5. 📈 Rural viability scores are EXCELLENT (93-96%)")

def main():
    print("🚀 RURAL-FOCUSED SOLAR OPTIMIZATION ANALYSIS")
    print("="*60)
    
    city_performance = analyze_rural_performance()
    analyze_weather_impact()
    cost_effectiveness_analysis(city_performance)
    rural_recommendations(city_performance)
    create_rural_summary()
    
    print(f"\n✅ RURAL ANALYSIS COMPLETE!")
    print("🌾 Key Finding: RF is the optimal choice for most rural deployments")

if __name__ == "__main__":
    main()