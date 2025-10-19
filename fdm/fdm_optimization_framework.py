# fdm_optimization_framework.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
import os
import yaml

class RuralSolarFDM:
    def __init__(self, config=None):
        """
        FDM solver optimized for rural Indian solar panel monitoring
        """
        if config is None:
            config = {
                'cities': ['jaipur', 'chennai', 'delhi', 'leh'],
                'noise_levels': [0, 5, 10, 20],
                'rural_mounting_type': 'rooftop'
            }
        self.config = config
        self.setup_rural_parameters()
        
    def setup_rural_parameters(self):
        """Set parameters specific to rural Indian solar installations"""
        # Typical rural panel parameters (cost-optimized)
        self.panel_length = 1.6  # meters (standard size)
        self.panel_width = 1.0   # meters
        self.thickness = 0.002   # meters
        
        # Material properties for typical rural installations
        self.k = 160    # Thermal conductivity (W/m·K) - aluminum frame
        self.rho = 2700 # Density (kg/m³)
        self.cp = 900   # Specific heat (J/kg·K)
        self.alpha = self.k / (self.rho * self.cp)  # Thermal diffusivity
        
        # Rural-specific coefficients
        self.absorptivity = 0.85  # Slightly lower for cost-effective coatings
        self.emissivity = 0.88    # Typical for rural-grade panels
        self.sigma = 5.67e-8      # Stefan-Boltzmann constant
        
        # Grid setup - optimized for computational efficiency
        self.nx = 40  # Reduced grid for faster computation (rural constraint)
        self.ny = 25
        self.dx = self.panel_length / (self.nx - 1)
        self.dy = self.panel_width / (self.ny - 1)
        
        # Initialize temperature field
        self.T = np.ones((self.nx, self.ny)) * 300  # Start at 300K (27°C)
        
        # Create grid coordinates
        self.x = np.linspace(0, self.panel_length, self.nx)
        self.y = np.linspace(0, self.panel_width, self.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        
        # Time step (1 hour in seconds)
        self.dt = 3600
        
    def rural_heat_transfer_coeff(self, wind_speed, mounting_type='rooftop'):
        """
        Calculate heat transfer coefficient for rural conditions
        Different mounting types affect cooling efficiency
        """
        if mounting_type == 'ground':
            h = 5.7 + 3.5 * wind_speed  # Better airflow
        elif mounting_type == 'rooftop':
            h = 5.2 + 3.2 * wind_speed  # Reduced airflow
        else:  # elevated
            h = 6.0 + 3.8 * wind_speed  # Best airflow
            
        return max(4.0, h)  # Minimum convection for stagnant air
    
    def rural_efficiency_model(self, temperature, dust_level=0):
        """
        Efficiency model considering rural conditions:
        - Temperature dependence
        - Dust accumulation effects
        - Panel degradation
        """
        # Base efficiency for rural-grade panels (slightly lower than premium)
        eta_ref = 0.16  # 16% nominal efficiency
        
        # Temperature coefficient (higher degradation in hot climates)
        beta = -0.0045  # %/°C (slightly worse than premium panels)
        
        # Dust impact (rural areas have more dust)
        dust_impact = 1.0 - (dust_level * 0.15)  # Up to 15% reduction
        
        # Calculate efficiency
        T_ref = 25  # °C
        T_cell = temperature - 273.15  # Convert to °C
        
        efficiency = eta_ref * (1 + beta * (T_cell - T_ref)) * dust_impact
        
        return max(0.10, efficiency)  # Minimum 10% efficiency
    
    def build_matrices(self, h_conv):
        """Build matrices for Crank-Nicolson scheme with Robin BC"""
        n = self.nx * self.ny
        A = csr_matrix((n, n))
        B = csr_matrix((n, n))
        
        # Thermal diffusivity coefficients
        rx = self.alpha * self.dt / (2 * self.dx**2)
        ry = self.alpha * self.dt / (2 * self.dy**2)
        
        # Build matrices
        for i in range(self.nx):
            for j in range(self.ny):
                idx = i * self.ny + j
                
                # Diagonal element
                A[idx, idx] = 1 + 2*rx + 2*ry
                B[idx, idx] = 1 - 2*rx - 2*ry
                
                # x-direction neighbors
                if i > 0:
                    A[idx, idx - self.ny] = -rx
                    B[idx, idx - self.ny] = rx
                if i < self.nx - 1:
                    A[idx, idx + self.ny] = -rx
                    B[idx, idx + self.ny] = rx
                
                # y-direction neighbors
                if j > 0:
                    A[idx, idx - 1] = -ry
                    B[idx, idx - 1] = ry
                if j < self.ny - 1:
                    A[idx, idx + 1] = -ry
                    B[idx, idx + 1] = ry
        
        # Apply Robin boundary conditions
        self.apply_robin_bc(A, h_conv)
        
        return A, B
    
    def apply_robin_bc(self, A, h_conv):
        """Apply Robin boundary conditions for rural cooling"""
        # Robin BC: -k*dT/dn = h*(T - T_air) + radiation
        coeff = h_conv * self.dx / self.k
        
        # Apply to all boundaries
        for i in [0, self.nx-1]:  # Left and right boundaries
            for j in range(self.ny):
                idx = i * self.ny + j
                A[idx, idx] += coeff
        
        for j in [0, self.ny-1]:  # Top and bottom boundaries
            for i in range(self.nx):
                idx = i * self.ny + j
                A[idx, idx] += coeff
    
    def solve_timestep(self, T_air, GHI, wind_speed, hour, dust_level=0):
        """Solve for one time step using Crank-Nicolson"""
        mounting_type = self.config.get('rural_mounting_type', 'rooftop')
        h_conv = self.rural_heat_transfer_coeff(wind_speed, mounting_type)
        A, B = self.build_matrices(h_conv)
        
        # Current state vector
        T_flat = self.T.flatten()
        
        # Source term (solar heating with dust effect)
        dust_factor = 1.0 - (dust_level * 0.15)
        Q_solar = self.absorptivity * GHI * dust_factor
        source_term = np.ones_like(T_flat) * (Q_solar * self.dt / (self.rho * self.cp))
        
        # Boundary condition contribution
        bc_contrib = np.zeros_like(T_flat)
        robin_coeff = h_conv * self.dx / self.k
        
        # Apply to all boundary points
        for i in [0, self.nx-1]:
            for j in range(self.ny):
                idx = i * self.ny + j
                bc_contrib[idx] += robin_coeff * T_air
        for j in [0, self.ny-1]:
            for i in range(self.nx):
                idx = i * self.ny + j
                bc_contrib[idx] += robin_coeff * T_air
        
        # Right-hand side
        b = B @ T_flat + source_term + bc_contrib
        
        # Solve system
        T_new_flat = spsolve(A, b)
        self.T = T_new_flat.reshape((self.nx, self.ny))
        
        return self.T.copy()
    
    def calculate_kpis(self, GHI, hour, dust_level=0):
        """Calculate Key Performance Indicators for rural analysis"""
        T_max = np.max(self.T)
        T_avg = np.mean(self.T)
        T_min = np.min(self.T)
        
        # Rural efficiency model
        efficiency = self.rural_efficiency_model(T_avg, dust_level)
        
        # Energy calculation (considering panel area)
        panel_area = self.panel_length * self.panel_width  # m²
        daily_energy = efficiency * GHI * panel_area / 1000  # kWh
        
        # Temperature distribution metrics
        temp_gradient = np.max(np.gradient(self.T, self.dx, self.dy))
        
        return {
            'T_max': T_max,
            'T_avg': T_avg,
            'T_min': T_min,
            'efficiency': efficiency,
            'daily_energy': daily_energy,
            'temp_gradient': temp_gradient,
            'hour': hour,
            'temperature_field': self.T.copy()
        }
    
    def run_simulation(self, weather_data, city_name, dust_level=0):
        """Run complete simulation for given weather data"""
        hours = len(weather_data['GHI'])
        results = []
        temperature_fields = []
        
        print(f"Starting FDM simulation for {city_name}...")
        
        # Reset temperature field for each simulation
        self.T = np.ones((self.nx, self.ny)) * weather_data['Ta'][0]
        
        for hour in range(hours):
            T_air = weather_data['Ta'][hour]
            GHI = weather_data['GHI'][hour]
            wind_speed = weather_data['wind_speed'][hour]
            
            # Solve for this time step
            T_field = self.solve_timestep(T_air, GHI, wind_speed, hour, dust_level)
            
            # Calculate KPIs
            kpis = self.calculate_kpis(GHI, hour, dust_level)
            results.append(kpis)
            temperature_fields.append(T_field.copy())
            
            if hour % 6 == 0:  # Print progress every 6 hours
                print(f"  Hour {hour:02d}: T_avg = {kpis['T_avg'] - 273.15:.1f}°C, "
                      f"η = {kpis['efficiency']*100:.1f}%")
        
        return results, np.array(temperature_fields)
    
    def process_noisy_datasets(self, city_name, noise_levels=[0, 5, 10, 20]):
        """
        Process all noise levels for a given city and evaluate FDM performance
        """
        results = {}
        
        for noise_level in noise_levels:
            print(f"Processing {city_name} with {noise_level}% noise...")
            
            try:
                # Load dataset (assuming files are in fdm_ready folder)
                data_file = f"fdm_ready/{city_name}_noise_{noise_level}.npz"
                data = np.load(data_file)
                
                # Convert to weather data format
                weather_data = {
                    'GHI': data['GHI'],
                    'Ta': data['Ta'],
                    'wind_speed': data['wind_speed']
                }
                
                # Run FDM simulation
                clean_results, temp_fields = self.run_simulation(weather_data, city_name)
                
                # Calculate performance metrics
                metrics = self.calculate_rural_metrics(clean_results, weather_data, noise_level)
                
                results[noise_level] = {
                    'temperature_fields': temp_fields,
                    'kpis': clean_results,
                    'metrics': metrics
                }
                
                # Save results
                self.save_optimized_results(city_name, noise_level, results[noise_level])
                
            except FileNotFoundError:
                print(f"  ⚠️  Data file not found for {city_name} {noise_level}% noise")
                continue
        
        return results
    
    def calculate_rural_metrics(self, fdm_results, noisy_data, noise_level):
        """
        Calculate metrics relevant for rural solar optimization
        """
        # Energy production metrics
        total_energy = sum([kpi['daily_energy'] for kpi in fdm_results])
        avg_efficiency = np.mean([kpi['efficiency'] for kpi in fdm_results])
        max_temperature = max([kpi['T_max'] for kpi in fdm_results])
        avg_temperature = np.mean([kpi['T_avg'] for kpi in fdm_results])
        
        # Cost-related metrics (for rural economic analysis)
        energy_cost_savings = total_energy * 8  # ₹8/kWh approximate savings
        panel_degradation = self.estimate_degradation(fdm_results)
        
        # Sensor optimization metrics
        temperature_accuracy = self.estimate_temperature_accuracy(fdm_results, noisy_data)
        required_sensor_density = self.optimize_sensor_density(fdm_results)
        
        # Performance metrics
        performance_penalty = self.calculate_noise_penalty(noise_level, avg_efficiency)
        
        viability_score = self.calculate_viability_score({
            'total_energy': total_energy,
            'avg_efficiency': avg_efficiency,
            'max_temperature': max_temperature,
            'noise_level': noise_level,
            'sensor_density': required_sensor_density
        })
        
        return {
            'total_energy_kwh': total_energy,
            'avg_efficiency_percent': avg_efficiency * 100,
            'max_temperature_c': max_temperature - 273.15,
            'avg_temperature_c': avg_temperature - 273.15,
            'energy_cost_savings_inr': energy_cost_savings,
            'estimated_degradation_per_year': panel_degradation,
            'temperature_accuracy_k': temperature_accuracy,
            'optimal_sensor_density': required_sensor_density,
            'performance_penalty_percent': performance_penalty,
            'viability_score': viability_score,
            'noise_tolerance_level': noise_level
        }
    
    def optimize_sensor_density(self, fdm_results):
        """
        Determine minimum sensor density needed for accurate monitoring
        Based on temperature field complexity
        """
        # Analyze temperature gradients to determine critical monitoring points
        temp_fields = [result.get('temperature_field', None) for result in fdm_results]
        temp_fields = [field for field in temp_fields if field is not None]
        
        if not temp_fields:
            return 4  # Default to 4 sensors
        
        # Calculate spatial temperature variations
        max_gradients = []
        for field in temp_fields:
            grad_x, grad_y = np.gradient(field)
            max_gradient = np.max(np.sqrt(grad_x**2 + grad_y**2))
            max_gradients.append(max_gradient)
        
        avg_gradient = np.mean(max_gradients)
        
        # Determine sensor density based on complexity
        if avg_gradient < 2.0:  # Low variation
            return 2  # 2 sensors sufficient
        elif avg_gradient < 5.0:  # Medium variation
            return 4  # 4 sensors recommended
        else:  # High variation
            return 6  # 6 sensors needed
    
    def estimate_degradation(self, fdm_results):
        """
        Estimate annual panel degradation based on temperature stress
        Higher temperatures accelerate degradation in rural conditions
        """
        avg_temperature = np.mean([kpi['T_avg'] for kpi in fdm_results]) - 273.15
        
        # Degradation model: higher temps → faster degradation
        base_degradation = 0.005  # 0.5% per year base rate
        temperature_acceleration = max(0, (avg_temperature - 35) * 0.0002)
        
        total_degradation = base_degradation + temperature_acceleration
        return min(0.02, total_degradation)  # Cap at 2% per year
    
    def estimate_temperature_accuracy(self, fdm_results, noisy_data):
        """
        Estimate temperature measurement accuracy based on noise level
        """
        # Simple accuracy model - in real implementation, this would compare
        # with ground truth or use statistical methods
        avg_temp = np.mean([kpi['T_avg'] for kpi in fdm_results])
        
        # Basic accuracy estimation (simplified)
        accuracy = 0.5  # Base accuracy in K
        return accuracy
    
    def calculate_noise_penalty(self, noise_level, efficiency):
        """
        Calculate performance penalty due to sensor noise
        """
        # Simplified model - noise affects monitoring accuracy which impacts optimization
        penalty = noise_level * 0.001  # 0.1% penalty per 1% noise
        return penalty * 100  # Convert to percentage
    
    def calculate_viability_score(self, metrics):
        """Calculate overall viability score for rural installation"""
        score = 0
        
        # Energy production (30% weight)
        energy_score = min(100, (metrics['total_energy'] / 3) * 100) * 0.3
        
        # Efficiency (25% weight)
        efficiency_score = metrics['avg_efficiency'] * 100 * 2.5 * 0.25
        
        # Noise tolerance (20% weight) - higher noise tolerance is better
        noise_score = max(0, 100 - metrics['noise_level'] * 3) * 0.2
        
        # Cost savings (15% weight) - fewer sensors = lower cost
        sensor_score = max(0, 100 - (metrics['sensor_density'] - 2) * 20) * 0.15
        
        # Temperature management (10% weight)
        temp_score = max(0, 100 - (metrics['max_temperature'] - 320) * 2) * 0.1
        
        return min(100, energy_score + efficiency_score + noise_score + sensor_score + temp_score)
    
    def generate_rural_recommendations(self, city_name, all_results):
        """
        Generate cost-saving recommendations for rural installations
        """
        recommendations = []
        
        for noise_level, result in all_results.items():
            metrics = result['metrics']
            
            rec = {
                'city': city_name,
                'noise_level': noise_level,
                'sensor_cost_optimization': f"Use {metrics['optimal_sensor_density']} sensors",
                'expected_energy_yield': f"{metrics['total_energy_kwh']:.1f} kWh/day",
                'cost_savings': f"₹{metrics['energy_cost_savings_inr']:.0f}/day",
                'monitoring_accuracy': f"±{metrics['temperature_accuracy_k']:.1f}°C",
                'degradation_warning': f"{metrics['estimated_degradation_per_year']*100:.1f}%/year",
                'performance_penalty': f"{metrics['performance_penalty_percent']:.1f}%",
                'viability_score': metrics['viability_score']
            }
            recommendations.append(rec)
        
        return recommendations
    
    def save_optimized_results(self, city_name, noise_level, results):
        """Save optimized results for comparison with ML/PINNs"""
        output_dir = "fdm_results"
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"{output_dir}/{city_name}_noise_{noise_level}_results.npz"
        
        np.savez(filename,
                 temperature_fields=results['temperature_fields'],
                 kpis=results['kpis'],
                 metrics=results['metrics'],
                 city=city_name,
                 noise_level=noise_level)
        
        print(f"  💾 Saved results to {filename}")
    
    def create_temperature_heatmap(self, city_name, results, hour=12):
        """Create temperature heatmap visualization"""
        for noise_level, result in results.items():
            temp_field = result['temperature_fields'][hour]
            
            plt.figure(figsize=(10, 8))
            plt.imshow(temp_field - 273.15, cmap='hot', origin='lower',
                      extent=[0, self.panel_length, 0, self.panel_width])
            plt.colorbar(label='Temperature (°C)')
            plt.xlabel('Panel Length (m)')
            plt.ylabel('Panel Width (m)')
            plt.title(f'{city_name} - Temperature Distribution at Hour {hour}\n'
                     f'Noise Level: {noise_level}%')
            
            # Save plot
            plt.savefig(f'fdm_results/{city_name}_noise_{noise_level}_heatmap.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()

# Main execution function
def run_complete_fdm_analysis(cities=None, noise_levels=None):
    """
    Complete FDM analysis for all cities and noise levels
    """
    if cities is None:
        cities = ['jaipur', 'chennai', 'delhi', 'leh']
    if noise_levels is None:
        noise_levels = [0, 5, 10, 20]
    
    config = {
        'cities': cities,
        'noise_levels': noise_levels,
        'rural_mounting_type': 'rooftop'  # Most common in rural India
    }
    
    fdm_optimizer = RuralSolarFDM(config)
    
    all_city_results = {}
    rural_recommendations = []
    
    for city in config['cities']:
        print(f"\n{'='*60}")
        print(f"ANALYZING {city.upper()} FOR RURAL OPTIMIZATION")
        print(f"{'='*60}")
        
        # Process all noise levels
        city_results = fdm_optimizer.process_noisy_datasets(city, config['noise_levels'])
        
        if city_results:  # Only if we have results
            all_city_results[city] = city_results
            
            # Generate rural-specific recommendations
            recommendations = fdm_optimizer.generate_rural_recommendations(city, city_results)
            rural_recommendations.extend(recommendations)
            
            # Create optimization plots
            create_rural_optimization_plots(city, city_results)
            
            # Create heatmaps
            fdm_optimizer.create_temperature_heatmap(city, city_results)
    
    # Generate comparative analysis
    if all_city_results:
        generate_comparative_analysis(all_city_results, rural_recommendations)
    
    return all_city_results, rural_recommendations

def create_rural_optimization_plots(city_name, city_results):
    """Create plots specifically for rural optimization analysis"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    noise_levels = sorted(city_results.keys())
    
    # Plot 1: Energy production vs noise
    energy_values = [city_results[noise]['metrics']['total_energy_kwh'] for noise in noise_levels]
    ax1.plot(noise_levels, energy_values, 'go-', linewidth=2, markersize=8)
    ax1.set_xlabel('Noise Level (%)')
    ax1.set_ylabel('Daily Energy (kWh)')
    ax1.set_title(f'{city_name} - Energy Production vs Sensor Noise')
    ax1.grid(True)
    
    # Plot 2: Cost savings vs noise
    cost_values = [city_results[noise]['metrics']['energy_cost_savings_inr'] for noise in noise_levels]
    ax2.plot(noise_levels, cost_values, 'bo-', linewidth=2, markersize=8)
    ax2.set_xlabel('Noise Level (%)')
    ax2.set_ylabel('Daily Savings (₹)')
    ax2.set_title(f'{city_name} - Cost Savings vs Sensor Noise')
    ax2.grid(True)
    
    # Plot 3: Optimal sensor density
    sensor_density = [city_results[noise]['metrics']['optimal_sensor_density'] for noise in noise_levels]
    ax3.bar(noise_levels, sensor_density, color='orange', alpha=0.7)
    ax3.set_xlabel('Noise Level (%)')
    ax3.set_ylabel('Recommended Sensors')
    ax3.set_title(f'{city_name} - Optimal Sensor Density')
    
    # Plot 4: Viability score
    viability_scores = [city_results[noise]['metrics']['viability_score'] for noise in noise_levels]
    ax4.plot(noise_levels, viability_scores, 'ro-', linewidth=2, markersize=8)
    ax4.set_xlabel('Noise Level (%)')
    ax4.set_ylabel('Viability Score')
    ax4.set_title(f'{city_name} - Rural Viability Score')
    ax4.grid(True)
    ax4.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(f'fdm_results/rural_optimization_{city_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_comparative_analysis(all_results, recommendations):
    """Generate final comparative analysis for all cities"""
    
    # Create summary DataFrame
    summary_data = []
    for rec in recommendations:
        energy_yield = float(rec['expected_energy_yield'].split()[0])
        cost_savings = float(rec['cost_savings'][1:-4])
        sensors_needed = int(rec['sensor_cost_optimization'].split()[1])
        
        summary_data.append({
            'City': rec['city'],
            'Noise_Level': rec['noise_level'],
            'Viability_Score': rec['viability_score'],
            'Energy_kWh': energy_yield,
            'Cost_Savings_INR': cost_savings,
            'Sensors_Needed': sensors_needed
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Find best configurations for each city
    best_configs = {}
    for city in all_results.keys():
        city_data = df_summary[df_summary['City'] == city]
        if not city_data.empty:
            best_idx = city_data['Viability_Score'].idxmax()
            best_configs[city] = city_data.loc[best_idx]
    
    print("\n" + "="*80)
    print("RURAL SOLAR OPTIMIZATION RESULTS - FDM ANALYSIS")
    print("="*80)
    
    for city, config in best_configs.items():
        print(f"\n🏆 BEST CONFIGURATION FOR {city.upper()}:")
        print(f"   • Noise Tolerance: {config['Noise_Level']}%")
        print(f"   • Sensors Required: {config['Sensors_Needed']}")
        print(f"   • Daily Energy: {config['Energy_kWh']:.1f} kWh")
        print(f"   • Daily Savings: ₹{config['Cost_Savings_INR']:.0f}")
        print(f"   • Viability Score: {config['Viability_Score']:.1f}/100")
    
    # Save detailed analysis
    df_summary.to_csv('fdm_results/rural_solar_optimization_fdm.csv', index=False)
    print(f"\n📊 Detailed analysis saved to 'fdm_results/rural_solar_optimization_fdm.csv'")

if __name__ == "__main__":
    # Create results directory
    os.makedirs("fdm_results", exist_ok=True)
    
    # Run complete FDM analysis
    print("🚀 Starting FDM Optimization Framework for Rural Solar Monitoring...")
    all_results, recommendations = run_complete_fdm_analysis()
    
    print("\n✅ FDM Analysis Completed!")
    print("📁 Results saved in 'fdm_results/' folder")
    print("📊 Use these results for comparison with ML and PINNs models")