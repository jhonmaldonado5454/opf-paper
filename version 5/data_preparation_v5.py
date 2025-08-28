"""
Data Preparation Script v5 for Texas Synthetic Power System Load Forecasting
This script generates a complete CSV file with all features needed for the models
Version: 5.0
Output: texas_load_forecasting_data_2018_v5.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

def create_load_forecasting_dataset_v5():
    """
    Create a comprehensive dataset for load forecasting with all features
    Version 5 with enhanced features and proper naming
    """
    
    print("="*80)
    print("GENERATING COMPREHENSIVE LOAD FORECASTING DATASET - VERSION 5")
    print("Texas Synthetic Power System (TX-123BT)")
    print("="*80)
    
    # Create hourly timestamps for one year (2018)
    dates = pd.date_range(start='2018-01-01 00:00:00', 
                         end='2018-12-31 23:00:00', 
                         freq='H')
    
    # Initialize dataframe
    df = pd.DataFrame()
    df['timestamp'] = dates
    df['year'] = dates.year
    df['month'] = dates.month
    df['day'] = dates.day
    df['hour'] = dates.hour
    df['day_of_week'] = dates.dayofweek  # 0=Monday, 6=Sunday
    df['day_of_year'] = dates.dayofyear
    df['week_of_year'] = dates.isocalendar().week
    df['quarter'] = dates.quarter
    df['is_weekend'] = (dates.dayofweek >= 5).astype(int)
    df['is_holiday'] = 0  # You can add US holidays here
    
    # Add cyclic encoding for temporal features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # System parameters (based on TX-123BT)
    total_capacity_MW = 54811.6  # Total generation capacity from the system
    base_load_MW = total_capacity_MW * 0.55  # Average load factor
    
    # Daily load patterns (typical power system profile)
    hourly_factors = np.array([
        0.67, 0.63, 0.60, 0.59, 0.59, 0.60,  # 00:00-05:00 (night)
        0.65, 0.72, 0.85, 0.95, 0.99, 1.00,  # 06:00-11:00 (morning)
        1.00, 0.99, 0.99, 0.98, 0.99, 1.00,  # 12:00-17:00 (afternoon)
        1.00, 0.96, 0.91, 0.83, 0.75, 0.70   # 18:00-23:00 (evening)
    ])
    
    # Seasonal factors by month (Texas climate)
    seasonal_factors = {
        1: 0.95,   # January - heating
        2: 0.93,   # February - heating
        3: 0.88,   # March - mild
        4: 0.85,   # April - mild
        5: 0.92,   # May - cooling starts
        6: 1.05,   # June - cooling
        7: 1.15,   # July - peak cooling
        8: 1.14,   # August - peak cooling
        9: 1.02,   # September - cooling
        10: 0.91,  # October - mild
        11: 0.89,  # November - mild
        12: 0.94   # December - heating
    }
    
    # Generate base load
    df['base_load_MW'] = base_load_MW
    
    # Apply hourly patterns
    for h in range(24):
        mask = df['hour'] == h
        df.loc[mask, 'base_load_MW'] *= hourly_factors[h]
    
    # Apply seasonal patterns
    for month, factor in seasonal_factors.items():
        mask = df['month'] == month
        df.loc[mask, 'base_load_MW'] *= factor
    
    # Weekend adjustment
    df.loc[df['is_weekend'] == 1, 'base_load_MW'] *= 0.85
    
    # Add growth trend (2% annual growth)
    trend = np.linspace(1.0, 1.02, len(df))
    df['base_load_MW'] *= trend
    
    # Add random variation (realistic noise)
    noise = np.random.normal(1.0, 0.015, len(df))  # 1.5% standard deviation
    df['load_MW'] = df['base_load_MW'] * noise
    
    # Weather features for 8 Texas zones
    # Base temperatures by month (Celsius)
    base_temps = {
        1: 10, 2: 12, 3: 16, 4: 20, 5: 24,
        6: 28, 7: 32, 8: 32, 9: 28, 10: 22,
        11: 16, 12: 11
    }
    
    # Generate weather for each zone
    zones = ['COAST', 'EAST', 'FWEST', 'NCENT', 'NORTH', 'SCENT', 'SOUTH', 'WEST']
    zone_temp_offset = {
        'COAST': 2, 'EAST': 0, 'FWEST': -2, 'NCENT': -1,
        'NORTH': -3, 'SCENT': 1, 'SOUTH': 3, 'WEST': -1
    }
    
    for zone in zones:
        # Temperature
        df[f'temp_{zone}_C'] = 0.0
        for month, base_temp in base_temps.items():
            mask = df['month'] == month
            zone_temp = base_temp + zone_temp_offset[zone]
            
            # Add daily variation
            daily_var = 5 * np.sin(2 * np.pi * df.loc[mask, 'hour'] / 24 - np.pi/2)
            
            # Add random variation
            random_var = np.random.normal(0, 2, sum(mask))
            
            df.loc[mask, f'temp_{zone}_C'] = zone_temp + daily_var + random_var
        
        # Humidity (%)
        df[f'humidity_{zone}_%'] = 60 + 20 * np.sin(2 * np.pi * df['day_of_year'] / 365 + np.pi/4)
        df[f'humidity_{zone}_%'] += np.random.normal(0, 5, len(df))
        df[f'humidity_{zone}_%'] = np.clip(df[f'humidity_{zone}_%'], 20, 95)
        
        # Wind speed (m/s)
        df[f'wind_speed_{zone}_ms'] = 3 + 2 * np.sin(2 * np.pi * df['day_of_year'] / 365)
        df[f'wind_speed_{zone}_ms'] += np.random.normal(0, 1, len(df))
        df[f'wind_speed_{zone}_ms'] = np.clip(df[f'wind_speed_{zone}_ms'], 0, 15)
        
        # Solar radiation (W/m²) - only during daylight
        solar_base = 600 * np.sin(np.pi * df['hour'] / 24)
        solar_seasonal = 1 + 0.3 * np.sin(2 * np.pi * df['day_of_year'] / 365)
        df[f'solar_rad_{zone}_Wm2'] = np.maximum(0, solar_base * solar_seasonal)
        df[f'solar_rad_{zone}_Wm2'] += np.random.normal(0, 30, len(df))
        df[f'solar_rad_{zone}_Wm2'] = np.clip(df[f'solar_rad_{zone}_Wm2'], 0, 1000)
    
    # Average weather across all zones
    temp_cols = [f'temp_{zone}_C' for zone in zones]
    humidity_cols = [f'humidity_{zone}_%' for zone in zones]
    wind_cols = [f'wind_speed_{zone}_ms' for zone in zones]
    solar_cols = [f'solar_rad_{zone}_Wm2' for zone in zones]
    
    df['temp_avg_C'] = df[temp_cols].mean(axis=1)
    df['humidity_avg_%'] = df[humidity_cols].mean(axis=1)
    df['wind_speed_avg_ms'] = df[wind_cols].mean(axis=1)
    df['solar_rad_avg_Wm2'] = df[solar_cols].mean(axis=1)
    
    # Zone-specific load distribution
    zone_load_fraction = {
        'COAST': 0.15, 'EAST': 0.20, 'FWEST': 0.08, 'NCENT': 0.18,
        'NORTH': 0.12, 'SCENT': 0.15, 'SOUTH': 0.07, 'WEST': 0.05
    }
    
    for zone, fraction in zone_load_fraction.items():
        # Zone load with correlation to temperature
        temp_effect = (df[f'temp_{zone}_C'] - 20) * 50  # 50 MW per degree from 20°C
        df[f'load_{zone}_MW'] = df['load_MW'] * fraction + temp_effect
        df[f'load_{zone}_MW'] *= np.random.normal(1.0, 0.02, len(df))  # Small variation
    
    # Create lagged features (previous hours)
    lag_hours = [1, 2, 3, 6, 12, 24, 48, 72, 168, 336]  # Up to 2 weeks
    
    for lag in lag_hours:
        df[f'load_lag_{lag}h_MW'] = df['load_MW'].shift(lag)
        df[f'temp_avg_lag_{lag}h_C'] = df['temp_avg_C'].shift(lag)
        
        # Zone-specific lags for major zones
        for zone in ['COAST', 'EAST', 'NCENT']:
            df[f'load_{zone}_lag_{lag}h_MW'] = df[f'load_{zone}_MW'].shift(lag)
    
    # Moving averages and statistics
    windows = [24, 48, 168, 336]  # Daily, 2-day, weekly, 2-weekly
    
    for window in windows:
        # Load statistics
        df[f'load_ma_{window}h_MW'] = df['load_MW'].rolling(window=window, min_periods=1).mean()
        df[f'load_std_{window}h_MW'] = df['load_MW'].rolling(window=window, min_periods=1).std()
        df[f'load_min_{window}h_MW'] = df['load_MW'].rolling(window=window, min_periods=1).min()
        df[f'load_max_{window}h_MW'] = df['load_MW'].rolling(window=window, min_periods=1).max()
        
        # Temperature statistics
        df[f'temp_ma_{window}h_C'] = df['temp_avg_C'].rolling(window=window, min_periods=1).mean()
        df[f'temp_std_{window}h_C'] = df['temp_avg_C'].rolling(window=window, min_periods=1).std()
    
    # Rate of change features
    df['load_change_1h_MW'] = df['load_MW'].diff(1)
    df['load_change_24h_MW'] = df['load_MW'].diff(24)
    df['load_change_168h_MW'] = df['load_MW'].diff(168)
    
    df['temp_change_1h_C'] = df['temp_avg_C'].diff(1)
    df['temp_change_24h_C'] = df['temp_avg_C'].diff(24)
    
    # Interaction features
    df['temp_load_interaction'] = df['temp_avg_C'] * df['load_MW'] / 1000
    df['humidity_load_interaction'] = df['humidity_avg_%'] * df['load_MW'] / 1000
    df['wind_load_interaction'] = df['wind_speed_avg_ms'] * df['load_MW'] / 1000
    
    # Peak indicators
    df['is_peak_hour'] = ((df['hour'] >= 14) & (df['hour'] <= 20)).astype(int)
    df['is_summer'] = ((df['month'] >= 6) & (df['month'] <= 9)).astype(int)
    df['is_winter'] = ((df['month'] <= 2) | (df['month'] >= 11)).astype(int)
    
    # Target variables for different horizons
    df['target_load_1h_MW'] = df['load_MW'].shift(-1)    # 1 hour ahead
    df['target_load_6h_MW'] = df['load_MW'].shift(-6)    # 6 hours ahead
    df['target_load_12h_MW'] = df['load_MW'].shift(-12)  # 12 hours ahead
    df['target_load_24h_MW'] = df['load_MW'].shift(-24)  # 24 hours ahead
    df['target_load_48h_MW'] = df['load_MW'].shift(-48)  # 48 hours ahead
    df['target_load_168h_MW'] = df['load_MW'].shift(-168)  # 1 week ahead
    df['target_load_360h_MW'] = df['load_MW'].shift(-360)  # 15 days ahead
    
    # Drop base_load_MW as it's not a real feature
    df.drop('base_load_MW', axis=1, inplace=True)
    
    # Drop rows with NaN values (from lag and target features)
    initial_rows = len(df)
    df.dropna(inplace=True)
    final_rows = len(df)
    
    # Round all numeric columns to 3 decimal places
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(3)
    
    # Save to CSV with v5 suffix
    output_file = 'texas_load_forecasting_data_2018_v5.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\nDataset Generated Successfully!")
    print(f"Version: 5.0")
    print(f"Initial rows: {initial_rows:,}")
    print(f"Final rows (after removing NaN): {final_rows:,}")
    print(f"Total features: {len(df.columns)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\nLoad Statistics:")
    print(f"  Mean: {df['load_MW'].mean():.2f} MW")
    print(f"  Min:  {df['load_MW'].min():.2f} MW")
    print(f"  Max:  {df['load_MW'].max():.2f} MW")
    print(f"  Std:  {df['load_MW'].std():.2f} MW")
    
    print(f"\nFile saved as: {output_file}")
    print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    
    # Print feature categories
    print("\n" + "="*60)
    print("FEATURE CATEGORIES IN DATASET (VERSION 5):")
    print("="*60)
    
    print("\n1. Temporal Features (19):")
    temporal_features = [col for col in df.columns if any(x in col for x in 
                         ['year', 'month', 'day', 'hour', 'week', 'quarter', 
                          'is_weekend', 'is_holiday', 'sin', 'cos', 'is_peak', 
                          'is_summer', 'is_winter'])]
    print(f"   {', '.join(temporal_features[:5])}...")
    
    print("\n2. Load Features (Current & Historical):")
    load_features = [col for col in df.columns if 'load' in col and 'target' not in col]
    print(f"   Total: {len(load_features)} features")
    print(f"   Including: zone loads, lags, moving averages, statistics")
    
    print("\n3. Weather Features:")
    weather_features = [col for col in df.columns if any(x in col for x in 
                       ['temp', 'humidity', 'wind', 'solar'])]
    print(f"   Total: {len(weather_features)} features")
    print(f"   Including: 8 zones + averages + lags + statistics")
    
    print("\n4. Interaction Features:")
    interaction_features = [col for col in df.columns if 'interaction' in col]
    print(f"   {', '.join(interaction_features)}")
    
    print("\n5. Target Variables (7 horizons):")
    target_features = [col for col in df.columns if 'target' in col]
    for target in target_features:
        print(f"   {target}")
    
    print("\n" + "="*60)
    print("Dataset ready for Julia implementation v5!")
    print("Run: julia load_forecasting_framework_v5.jl")
    print("="*60)
    
    return df

if __name__ == "__main__":
    # Generate the dataset
    print("Data Preparation Script - Version 5")
    print("File name: data_preparation_v5.py")
    df = create_load_forecasting_dataset_v5()
    
    # Display first few rows
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    # Display data types
    print("\nColumn data types:")
    print(df.dtypes.value_counts())