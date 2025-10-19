import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib
import os
import matplotlib.pyplot as plt

# Check if XGBoost is available
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost is available")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available. Using only Random Forest.")

def train_baseline_models():
    print("🤖 Training Baseline ML Models...")
    
    # Load merged data - from parent directory
    data_path = "../data_processed/merged_complete_data.csv"
    if not os.path.exists(data_path):
        print("❌ Merged data not found! Run 1_data_merging.py first")
        print("📁 Looking for:", os.path.abspath(data_path))
        return None, None
    
    data = pd.read_csv(data_path)
    
    print(f"📊 Loaded data shape: {data.shape}")
    print("🔍 Available columns:", data.columns.tolist())
    
    # Check for problematic header rows
    print(f"\n🔍 Checking data quality...")
    print(f"First few rows of data:")
    print(data.head(3))
    
    # Check if there are header rows mixed in data (like '-BEGIN HEADER-')
    if '-BEGIN HEADER-' in data.columns:
        print("⚠️  Found header marker in columns. Cleaning data...")
        # Remove rows that contain header markers
        data = data[data['-BEGIN HEADER-'] != '-BEGIN HEADER-']
        data = data.dropna(subset=['T2M', 'ALLSKY_SFC_SW_DWN'])  # Drop rows with missing key features
    
    # Use only CLEAN data for training baseline (if available)
    if 'clean' in data['scenario_type'].unique():
        train_data = data[data['scenario_type'] == 'clean'].copy()  # Use .copy() to avoid warnings
        print("✅ Using CLEAN data for training")
    else:
        train_data = data[data['scenario_type'] == 'noisy'].copy()
        print("⚠️  Using NOISY data for training (clean data not available)")
    
    print(f"Training data shape before cleaning: {train_data.shape}")
    
    # Data Cleaning: Remove rows with missing values in key columns
    train_data = train_data.dropna(subset=['T2M', 'ALLSKY_SFC_SW_DWN', 'WS10M'])
    print(f"Training data shape after cleaning: {train_data.shape}")
    
    if len(train_data) == 0:
        print("❌ No valid training data after cleaning!")
        return None, None
    
    # Based on your column structure from the merge
    feature_columns = [
        'T2M',           # Temperature at 2 meters (°C) - Ambient temperature
        'ALLSKY_SFC_SW_DWN',  # All-sky surface shortwave downward irradiance (W/m²) - Solar irradiance
        'WS10M',         # Wind speed at 10 meters (m/s)
        'HR',            # Hour of day
        'MO',            # Month (seasonal pattern)
    ]
    
    # TARGET - Check for panel temperature column
    print("\n🔍 Checking for panel temperature column...")
    panel_temp_columns = [col for col in data.columns if 'panel' in col.lower() or 'temp' in col.lower() or 'T_' in col]
    
    if panel_temp_columns:
        target_column = panel_temp_columns[0]
        print(f"✅ Found panel temperature column: {target_column}")
    else:
        # Calculate approximate panel temperature using simple model
        print("⚠️  No direct panel temperature column found. Calculating using simple model...")
        # Simple panel temperature model: T_panel = T_ambient + (irradiance * 0.03)
        # Use .loc to avoid SettingWithCopyWarning
        train_data.loc[:, 'panel_temperature'] = train_data['T2M'] + (train_data['ALLSKY_SFC_SW_DWN'] * 0.03)
        target_column = 'panel_temperature'
        print("✅ Created panel_temperature using: T_panel = T_ambient + (irradiance * 0.03)")
    
    # Check if all required columns exist
    missing_features = [col for col in feature_columns if col not in train_data.columns]
    if missing_features:
        print(f"⚠️  Missing features: {missing_features}")
        feature_columns = [col for col in feature_columns if col in train_data.columns]
    
    if target_column not in train_data.columns:
        print(f"❌ Target column '{target_column}' not found in data!")
        print("Available columns:", train_data.columns.tolist())
        return None, None
    
    # Final data cleaning - remove any rows with NaN in target
    train_data = train_data.dropna(subset=[target_column])
    print(f"Final training data shape: {train_data.shape}")
    
    if len(train_data) == 0:
        print("❌ No valid data after final cleaning!")
        return None, None
    
    print(f"\n🎯 Using features: {feature_columns}")
    print(f"🎯 Using target: {target_column}")
    
    # Convert numeric columns to proper data types
    for col in feature_columns + [target_column]:
        train_data[col] = pd.to_numeric(train_data[col], errors='coerce')
    
    # Remove any remaining NaN values
    train_data = train_data.dropna(subset=feature_columns + [target_column])
    print(f"Training data after final numeric conversion: {train_data.shape}")
    
    # One-hot encode city
    cities_encoded = pd.get_dummies(train_data['city'], prefix='city')
    features = pd.concat([train_data[feature_columns], cities_encoded], axis=1)
    
    X = features
    y = train_data[target_column]
    
    print(f"📐 Final feature matrix shape: {X.shape}")
    print(f"📐 Target vector shape: {y.shape}")
    
    # Check for any remaining NaN values
    print(f"🔍 NaN check - Features: {X.isna().sum().sum()}, Target: {y.isna().sum()}")
    
    if len(X) == 0 or len(y) == 0:
        print("❌ No data available for training!")
        return None, None
    
    # Split data (80% train, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"📊 Training set: {X_train.shape[0]} samples")
    print(f"📊 Validation set: {X_val.shape[0]} samples")
    
    if X_train.shape[0] == 0:
        print("❌ No training samples available!")
        return None, None
    
    # Train Random Forest (always available)
    print("\n🚀 Training Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=100,  # Reduced for stability
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42, 
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    # Train XGBoost only if available
    xgb_model = None
    if XGBOOST_AVAILABLE:
        print("🚀 Training XGBoost...")
        xgb_model = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
    else:
        print("⏭️  Skipping XGBoost (not installed)")
    
    # Create models directory if not exists
    os.makedirs('../models', exist_ok=True)
    
    # Save models
    joblib.dump(rf_model, '../models/random_forest_baseline.pkl')
    if xgb_model is not None:
        joblib.dump(xgb_model, '../models/xgboost_baseline.pkl')
        print("✅ Saved Random Forest and XGBoost models")
    else:
        print("✅ Saved Random Forest model only")
    
    # Predictions and metrics for Random Forest
    rf_train_pred = rf_model.predict(X_train)
    rf_val_pred = rf_model.predict(X_val)
    
    rf_train_rmse = mean_squared_error(y_train, rf_train_pred, squared=False)
    rf_val_rmse = mean_squared_error(y_val, rf_val_pred, squared=False)
    rf_train_mae = mean_absolute_error(y_train, rf_train_pred)
    rf_val_mae = mean_absolute_error(y_val, rf_val_pred)
    
    # Predictions and metrics for XGBoost if available
    if xgb_model is not None:
        xgb_train_pred = xgb_model.predict(X_train)
        xgb_val_pred = xgb_model.predict(X_val)
        
        xgb_train_rmse = mean_squared_error(y_train, xgb_train_pred, squared=False)
        xgb_val_rmse = mean_squared_error(y_val, xgb_val_pred, squared=False)
        xgb_train_mae = mean_absolute_error(y_train, xgb_train_pred)
        xgb_val_mae = mean_absolute_error(y_val, xgb_val_pred)
    
    print(f"\n📊 MODEL PERFORMANCE:")
    print("="*50)
    print(f"{'Model':<15} {'Split':<10} {'RMSE':<10} {'MAE':<10}")
    print("-"*50)
    print(f"{'Random Forest':<15} {'Train':<10} {rf_train_rmse:<10.4f} {rf_train_mae:<10.4f}")
    print(f"{'Random Forest':<15} {'Val':<10} {rf_val_rmse:<10.4f} {rf_val_mae:<10.4f}")
    
    if xgb_model is not None:
        print(f"{'XGBoost':<15} {'Train':<10} {xgb_train_rmse:<10.4f} {xgb_train_mae:<10.4f}")
        print(f"{'XGBoost':<15} {'Val':<10} {xgb_val_rmse:<10.4f} {xgb_val_mae:<10.4f}")
    else:
        print(f"{'XGBoost':<15} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
    
    # Feature importance
    print(f"\n🔍 FEATURE IMPORTANCE (Random Forest):")
    print("-"*40)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in feature_importance.head(10).iterrows():
        print(f"{row['feature']:<25} {row['importance']:.4f}")
    
    # Create results directory if not exists
    os.makedirs('../results', exist_ok=True)
    
    # Create feature importance plot
    plt.figure(figsize=(10, 6))
    top_features = feature_importance.head(10)
    plt.barh(top_features['feature'], top_features['importance'])
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Feature Importances - Random Forest')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('../results/feature_importance.png', dpi=300, bbox_inches='tight')
    print("✅ Feature importance plot saved to ../results/feature_importance.png")
    
    # Save training summary
    training_summary = {
        'features_used': feature_columns,
        'target_column': target_column,
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'random_forest_val_rmse': rf_val_rmse,
        'xgboost_available': XGBOOST_AVAILABLE,
        'cities_included': list(train_data['city'].unique()),
        'xgboost_val_rmse': xgb_val_rmse if xgb_model is not None else 'N/A'
    }
    
    import json
    with open('../results/training_summary.json', 'w') as f:
        json.dump(training_summary, f, indent=2)
    
    print(f"\n✅ Models saved to ../models/")
    print("✅ Training summary saved to ../results/training_summary.json")
    
    # Installation instructions if XGBoost is missing
    if not XGBOOST_AVAILABLE:
        print(f"\n💡 INSTALLATION TIP: To use XGBoost, run:")
        print("   pip install xgboost")
        print("   OR")
        print("   conda install -c conda-forge xgboost")
    
    return rf_model, xgb_model

if __name__ == "__main__":
    rf_model, xgb_model = train_baseline_models()
    
    if rf_model is not None:
        print(f"\n🎯 TRAINING COMPLETE!")
        if xgb_model is not None:
            print("✅ Both Random Forest and XGBoost models trained successfully!")
        else:
            print("✅ Random Forest model trained successfully!")
            print("💡 You can still proceed without XGBoost for the competition")
        print("Next step: Run 3_scenario_testing.py to test models on all scenarios")
    else:
        print("❌ Training failed! Check the data quality issues above.")