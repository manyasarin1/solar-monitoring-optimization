# data_integration.py (FIXED VERSION)
import pandas as pd
import numpy as np
import os
import glob
import re

class SolarDataIntegrator:
    def __init__(self, data_folder="solar_data"):
        self.data_folder = data_folder
        os.makedirs(data_folder, exist_ok=True)
        
    def discover_datasets(self):
        """Auto-discover all your CSV datasets"""
        csv_files = glob.glob(os.path.join(self.data_folder, "*.csv"))
        print(f"Found {len(csv_files)} CSV files:")
        
        datasets = {}
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            city, noise_level = self.parse_filename(filename)
            if city:
                if city not in datasets:
                    datasets[city] = {}
                datasets[city][noise_level] = file_path
                print(f"  - {city} ({noise_level}% noise): {filename}")
        
        return datasets
    
    def parse_filename(self, filename):
        """Parse your CSV filenames to extract city and noise level"""
        filename_lower = filename.lower()
        
        # Detect city
        city = None
        for city_name in ['jaipur', 'chennai', 'delhi', 'leh']:
            if city_name in filename_lower:
                city = city_name
                break
        
        # Detect noise level
        noise_level = 0
        if 'noise' in filename_lower:
            # Extract number before 'noise' or '%'
            noise_match = re.search(r'(\d+)%?\_?noise', filename_lower)
            if noise_match:
                noise_level = int(noise_match.group(1))
            else:
                # Try to find any number in filename
                numbers = re.findall(r'\d+', filename_lower)
                if numbers:
                    noise_level = int(numbers[0])
        
        return city, noise_level
    
    def load_and_validate_dataset(self, file_path, city_name, noise_level):
        """Load and validate your CSV dataset"""
        try:
            df = pd.read_csv(file_path)
            print(f"\nLoading {city_name} ({noise_level}% noise) - Shape: {df.shape}")
            
            # Display basic info
            print(f"Columns: {df.columns.tolist()}")
            print(f"First 3 rows:")
            print(df.head(3))
            
            # Standardize column names
            df_standard = self.standardize_columns(df, city_name)
            
            # Validate data quality
            self.validate_data_quality(df_standard, city_name, noise_level)
            
            return df_standard
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def standardize_columns(self, df, city_name):
        """Standardize your column names to match FDM requirements"""
        df_standard = df.copy()
        
        # Common column mappings for solar data
        column_mappings = {
            'GHI': ['GHI', 'ghi', 'solar_irradiance', 'irradiance', 'global_irradiance', 'solar'],
            'Ta': ['Ta', 'ta', 'temperature', 'temp', 'air_temperature', 'ambient_temp', 'air_temp'],
            'wind_speed': ['wind_speed', 'ws', 'wind', 'wind_velocity', 'wind_speed_m_s'],
            'timestamp': ['timestamp', 'time', 'date_time', 'datetime', 'hour', 'time_hour'],
            'humidity': ['humidity', 'rh', 'relative_humidity', 'hum']
        }
        
        # Apply mappings
        for standard_col, possible_names in column_mappings.items():
            for possible_name in possible_names:
                if possible_name in df.columns:
                    df_standard[standard_col] = df[possible_name]
                    print(f"   Mapped '{possible_name}' -> '{standard_col}'")
                    break
            else:
                if standard_col in ['GHI', 'Ta', 'wind_speed']:
                    print(f"   Warning: Missing '{standard_col}', will generate synthetic data")
        
        return df_standard
    
    def validate_data_quality(self, df, city_name, noise_level):
        """Validate your dataset quality"""
        print(f"   Data Quality Check for {city_name} ({noise_level}% noise):")
        
        # Check required columns
        required_cols = ['GHI', 'Ta', 'wind_speed']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"   Missing columns: {missing_cols}")
            # Generate missing data
            for col in missing_cols:
                df[col] = self.generate_missing_data(col, len(df), city_name)
                print(f"   Generated synthetic data for {col}")
        else:
            print("   All required columns present")
        
        # Check data ranges
        if 'GHI' in df.columns:
            ghi_range = (df['GHI'].min(), df['GHI'].max())
            print(f"   GHI range: {ghi_range[0]:.1f} to {ghi_range[1]:.1f} W/m2")
        
        if 'Ta' in df.columns:
            ta_range = (df['Ta'].min(), df['Ta'].max())
            print(f"   Temperature range: {ta_range[0]:.1f} to {ta_range[1]:.1f} C")
        
        if 'wind_speed' in df.columns:
            ws_range = (df['wind_speed'].min(), df['wind_speed'].max())
            print(f"   Wind speed range: {ws_range[0]:.1f} to {ws_range[1]:.1f} m/s")
        
        # Check for NaN values
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            print(f"   Found {nan_count} NaN values - filling with mean")
            df.fillna(df.mean(), inplace=True)
        
        print(f"   Data validation completed")
    
    def generate_missing_data(self, data_type, length, city_name):
        """Generate realistic synthetic data if your CSV is missing columns"""
        city_profiles = {
            'jaipur': {'ghi_peak': 950, 'temp_range': (25, 42), 'wind_range': (1, 5)},
            'chennai': {'ghi_peak': 900, 'temp_range': (28, 38), 'wind_range': (2, 6)},
            'delhi': {'ghi_peak': 980, 'temp_range': (20, 38), 'wind_range': (1, 4)},
            'leh': {'ghi_peak': 1100, 'temp_range': (8, 25), 'wind_range': (3, 8)}
        }
        
        profile = city_profiles.get(city_name.lower(), {'ghi_peak': 800, 'temp_range': (20, 35), 'wind_range': (1, 5)})
        
        if data_type == 'GHI':
            return self.generate_ghi_data(length, profile['ghi_peak'])
        elif data_type == 'Ta':
            return self.generate_temperature_data(length, profile['temp_range'])
        elif data_type == 'wind_speed':
            return self.generate_wind_data(length, profile['wind_range'])
        else:
            return np.zeros(length)
    
    def generate_ghi_data(self, length, ghi_peak):
        """Generate realistic GHI data"""
        ghi_data = []
        for i in range(length):
            hour = i % 24
            if 6 <= hour <= 18:
                ghi = ghi_peak * np.sin(np.pi * (hour - 6) / 12) ** 2
                ghi += np.random.normal(0, 30)
                ghi = max(0, ghi)
            else:
                ghi = 0
            ghi_data.append(ghi)
        return ghi_data
    
    def generate_temperature_data(self, length, temp_range):
        """Generate realistic temperature data"""
        temp_data = []
        temp_amplitude = (temp_range[1] - temp_range[0]) / 2
        temp_base = temp_range[0] + temp_amplitude
        
        for i in range(length):
            hour = i % 24
            temp = temp_base + temp_amplitude * np.sin(2 * np.pi * (hour - 14) / 24)
            temp += np.random.normal(0, 1)
            temp_data.append(temp)
        return temp_data
    
    def generate_wind_data(self, length, wind_range):
        """Generate realistic wind speed data"""
        wind_data = []
        wind_base = (wind_range[0] + wind_range[1]) / 2
        
        for i in range(length):
            hour = i % 24
            # Slight diurnal pattern
            diurnal_factor = 0.2 * np.sin(2 * np.pi * (hour - 14) / 24)
            wind = wind_base + diurnal_factor + np.random.normal(0, 0.3)
            wind = max(wind_range[0], min(wind_range[1], wind))
            wind_data.append(wind)
        return wind_data
    
    def save_for_fdm(self, df, city_name, noise_level):
        """Save processed data in FDM-ready format"""
        # Convert temperature to Kelvin
        Ta_K = df['Ta'] + 273.15
        
        # Calculate derived parameters
        h = 5.7 + 3.8 * df['wind_speed']
        q_solar = 0.9 * df['GHI']
        
        # Create output filename
        output_file = f"fdm_ready/{city_name}_noise_{noise_level}.npz"
        os.makedirs("fdm_ready", exist_ok=True)
        
        # Save as NPZ
        np.savez(output_file,
                 GHI=df['GHI'].values,
                 Ta=Ta_K.values,
                 wind_speed=df['wind_speed'].values,
                 h=h.values,
                 q_solar=q_solar.values,
                 time=df['timestamp'].values if 'timestamp' in df.columns else np.arange(len(df)),
                 city_name=city_name,
                 noise_level=noise_level)
        
        print(f"   Saved FDM-ready data: {output_file}")
        return output_file

# UPDATED MAIN EXECUTION WITH YOUR DATASETS
def main_with_your_data():
    """Main function that works with YOUR actual datasets"""
    
    integrator = SolarDataIntegrator()
    
    # Step 1: Discover your datasets
    print("Discovering your datasets...")
    datasets = integrator.discover_datasets()
    
    if not datasets:
        print("\nNo datasets found! Please ensure your CSV files are in the 'solar_data' folder.")
        print("Expected filename format: 'jaipur_noise_5.csv', 'chennai_10_noise.csv', etc.")
        return
    
    # Step 2: Process all datasets
    fdm_ready_files = {}
    
    for city, noise_files in datasets.items():
        fdm_ready_files[city] = {}
        
        for noise_level, file_path in noise_files.items():
            print(f"\nProcessing {city} - {noise_level}% noise")
            
            # Load and validate your dataset
            df_processed = integrator.load_and_validate_dataset(file_path, city, noise_level)
            
            if df_processed is not None:
                # Save in FDM format
                fdm_file = integrator.save_for_fdm(df_processed, city, noise_level)
                fdm_ready_files[city][noise_level] = fdm_file
    
    # Step 3: Run FDM analysis on all processed datasets
    print(f"\nStarting FDM analysis on {sum(len(files) for files in fdm_ready_files.values())} datasets...")
    run_fdm_on_all_datasets(fdm_ready_files)
    
    return fdm_ready_files

def run_fdm_on_all_datasets(fdm_ready_files):
    """Run FDM analysis on all your processed datasets"""
    try:
        from fdm_optimization_framework import RuralSolarFDM
    except ImportError:
        print("FDM optimization framework not found. Please ensure fdm_optimization_framework.py is in the same directory.")
        return
    
    # Modified FDM runner that uses your specific datasets
    config = {
        'cities': list(fdm_ready_files.keys()),
        'noise_levels': [],
        'rural_mounting_type': 'rooftop',
        'custom_datasets': fdm_ready_files  # Your actual datasets
    }
    
    # Extract all unique noise levels
    all_noise_levels = set()
    for city_files in fdm_ready_files.values():
        all_noise_levels.update(city_files.keys())
    config['noise_levels'] = sorted(all_noise_levels)
    
    print(f"\nFDM Analysis Configuration:")
    print(f"   Cities: {config['cities']}")
    print(f"   Noise Levels: {config['noise_levels']}")
    print(f"   Total datasets: {sum(len(files) for files in fdm_ready_files.values())}")
    
    # Initialize FDM optimizer
    fdm_optimizer = RuralSolarFDM(config)
    
    # Process each city and noise level
    all_results = {}
    
    for city, noise_files in fdm_ready_files.items():
        print(f"\n" + "="*50)
        print(f"ANALYZING {city.upper()}")
        print(f"="*50)
        
        city_results = {}
        
        for noise_level, data_file in noise_files.items():
            print(f"\nProcessing {noise_level}% noise...")
            
            # Load the FDM-ready data
            data = np.load(data_file)
            
            # Convert to dictionary for FDM
            weather_data = {
                'GHI': data['GHI'],
                'Ta': data['Ta'],
                'wind_speed': data['wind_speed']
            }
            
            # Run FDM simulation
            results, temp_fields = fdm_optimizer.run_simulation(weather_data, city)
            
            # Calculate metrics
            metrics = fdm_optimizer.calculate_rural_metrics(results, weather_data, noise_level)
            
            city_results[noise_level] = {
                'temperature_fields': temp_fields,
                'kpis': results,
                'metrics': metrics
            }
            
            print(f"   Completed - Viability: {metrics.get('viability_score', 0):.1f}")
        
        all_results[city] = city_results
        
        # Generate recommendations for this city
        recommendations = fdm_optimizer.generate_rural_recommendations(city, city_results)
        
        # Print summary
        print(f"\n{city.upper()} SUMMARY:")
        for rec in recommendations:
            print(f"   {rec['noise_level']}% noise: {rec['viability_score']:.1f} score, "
                  f"{rec['sensor_cost_optimization']}, {rec['expected_energy_yield']}")
    
    # Generate final comparative analysis
    generate_final_comparison(all_results)

def generate_final_comparison(all_results):
    """Generate final comparison across all your datasets"""
    print(f"\n" + "="*80)
    print(f"FINAL COMPARATIVE ANALYSIS - YOUR DATASETS")
    print(f"="*80)
    
    summary_data = []
    
    for city, city_results in all_results.items():
        for noise_level, results in city_results.items():
            metrics = results['metrics']
            
            summary_data.append({
                'City': city,
                'Noise_Level': noise_level,
                'Viability_Score': metrics.get('viability_score', 0),
                'Energy_kWh': metrics.get('total_energy_kwh', 0),
                'Cost_Savings_INR': metrics.get('energy_cost_savings_inr', 0),
                'Sensors_Needed': metrics.get('optimal_sensor_density', 4),
                'Avg_Efficiency_%': metrics.get('avg_efficiency_percent', 0),
                'Max_Temperature_C': metrics.get('max_temperature_c', 0)
            })
    
    # Create summary DataFrame
    df_summary = pd.DataFrame(summary_data)
    
    # Find best configuration for each city
    best_configs = {}
    for city in all_results.keys():
        city_data = df_summary[df_summary['City'] == city]
        if not city_data.empty:
            best_idx = city_data['Viability_Score'].idxmax()
            best_configs[city] = city_data.loc[best_idx]
    
    # Display results
    for city, config in best_configs.items():
        print(f"\n{city.upper()} - OPTIMAL CONFIGURATION:")
        print(f"   * Noise Tolerance: {config['Noise_Level']}%")
        print(f"   * Recommended Sensors: {config['Sensors_Needed']}")
        print(f"   * Daily Energy: {config['Energy_kWh']:.1f} kWh")
        print(f"   * Daily Savings: Rs.{config['Cost_Savings_INR']:.0f}")
        print(f"   * Viability Score: {config['Viability_Score']:.1f}/100")
    
    # Save detailed results
    df_summary.to_csv('YOUR_FDM_ANALYSIS_RESULTS.csv', index=False)
    print(f"\nFull analysis saved to 'YOUR_FDM_ANALYSIS_RESULTS.csv'")

# QUICK START INSTRUCTIONS (FIXED - NO EMOJIS)
def setup_instructions():
    """Print setup instructions"""
    print("""
    HOW TO USE YOUR DATASETS:
    
    1. CREATE A FOLDER called 'solar_data' in the same directory as this script
    
    2. PLACE YOUR CSV FILES in the 'solar_data' folder with names like:
       - jaipur_noise_0.csv
       - jaipur_noise_5.csv  
       - jaipur_noise_10.csv
       - jaipur_noise_20.csv
       - chennai_noise_0.csv
       - ... etc.
    
    3. RUN THIS SCRIPT - it will automatically:
       - Discover all your CSV files
       - Process them for FDM
       - Run complete analysis
       - Generate optimization results
    
    4. YOUR CSV FILES should contain columns like:
       - GHI/solar_irradiance (W/m2)
       - Ta/temperature (C) 
       - wind_speed (m/s)
       - timestamp/hour (optional)
    """)

if __name__ == "__main__":
    setup_instructions()
    main_with_your_data()