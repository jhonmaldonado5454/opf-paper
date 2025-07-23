import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configuration
start_date = datetime(2023, 1, 1)
end_date = start_date + timedelta(days=365)  # Full year for better patterns
interval = timedelta(minutes=5)

# Lists to store data
rows = []

# Define US holidays for 2023 (based on Hong Tao's thesis)
us_holidays = {
    datetime(2023, 1, 1).date(),    # New Year's Day
    datetime(2023, 1, 16).date(),   # Martin Luther King Jr. Day (3rd Monday in Jan)
    datetime(2023, 2, 20).date(),   # Presidents Day (3rd Monday in Feb)
    datetime(2023, 5, 29).date(),   # Memorial Day (Last Monday in May)
    datetime(2023, 7, 4).date(),    # Independence Day
    datetime(2023, 9, 4).date(),    # Labor Day (1st Monday in Sep)
    datetime(2023, 10, 9).date(),   # Columbus Day (2nd Monday in Oct)
    datetime(2023, 11, 11).date(),  # Veterans Day
    datetime(2023, 11, 23).date(),  # Thanksgiving (4th Thursday in Nov)
    datetime(2023, 12, 25).date(),  # Christmas Day
}

# Temperature parameters for realistic seasonal variation
def generate_temperature(timestamp):
    """Generate realistic temperature based on time of year and day"""
    # Day of year for seasonal component
    day_of_year = timestamp.timetuple().tm_yday
    
    # Seasonal component (warmer in summer, cooler in winter)
    seasonal_temp = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)  # Peak around day 171 (June 20)
    
    # Daily component (warmer in afternoon, cooler at night)
    hour_of_day = timestamp.hour + timestamp.minute / 60
    daily_temp = 5 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)  # Peak at 3 PM
    
    # Random variation
    random_temp = np.random.normal(0, 2)
    
    # Combine components
    temperature = seasonal_temp + daily_temp + random_temp
    
    return temperature

# Generate demand based on multiple factors
def generate_demand(timestamp, temperature, is_holiday, is_weekend):
    """Generate realistic demand based on various factors"""
    hour = timestamp.hour
    minute = timestamp.minute
    day_of_week = timestamp.weekday()
    month = timestamp.month
    
    # Base load pattern
    hour_decimal = hour + minute / 60
    
    # Different patterns for weekday vs weekend
    if is_weekend:
        # Weekend pattern: later morning peak, lower overall
        base_load = 300 + 50 * np.sin(2 * np.pi * (hour_decimal - 8) / 24)
    else:
        # Weekday pattern: morning and evening peaks
        morning_peak = 40 * np.exp(-((hour_decimal - 8)**2) / 8)
        evening_peak = 60 * np.exp(-((hour_decimal - 18)**2) / 12)
        base_load = 350 + morning_peak + evening_peak
    
    # Temperature effect (higher demand for extreme temperatures)
    # Heating load (cold weather) and cooling load (hot weather)
    temp_effect = 0
    if temperature < 10:  # Heating
        temp_effect = 2 * (10 - temperature)
    elif temperature > 25:  # Cooling
        temp_effect = 3 * (temperature - 25)
    
    # Seasonal effect
    seasonal_factor = 1.0
    if month in [12, 1, 2]:  # Winter
        seasonal_factor = 1.15
    elif month in [6, 7, 8]:  # Summer
        seasonal_factor = 1.20
    elif month in [3, 4, 5, 9, 10, 11]:  # Spring/Fall
        seasonal_factor = 1.0
    
    # Holiday effect
    holiday_factor = 0.85 if is_holiday else 1.0
    
    # Industrial/commercial component (lower on weekends/holidays)
    commercial_load = 100 * (0.5 if is_weekend or is_holiday else 1.0)
    
    # Combine all effects
    demand = (base_load + commercial_load + temp_effect) * seasonal_factor * holiday_factor
    
    # Add random noise
    demand += np.random.normal(0, 10)
    
    return max(demand, 50)  # Ensure demand is never negative

# Generate the data
current = start_date
previous_temps = []  # Store recent temperatures for continuity

print(f"Generating data from {start_date} to {end_date}...")
data_points = 0

while current < end_date:
    # Extract time components
    hour = current.hour
    minute = current.minute
    day_of_week = current.weekday()  # 0=Monday, 6=Sunday
    month = current.month
    
    # Determine if holiday or weekend
    is_holiday = int(current.date() in us_holidays)
    is_weekend = int(day_of_week >= 5)  # Saturday=5, Sunday=6
    
    # Generate temperature with some continuity
    if len(previous_temps) > 0:
        # Temperature changes gradually
        last_temp = previous_temps[-1]
        temp_change = np.random.normal(0, 0.5)
        base_temp = generate_temperature(current)
        temperature = 0.7 * last_temp + 0.3 * base_temp + temp_change
    else:
        temperature = generate_temperature(current)
    
    previous_temps.append(temperature)
    if len(previous_temps) > 288:  # Keep only last 24 hours (288 * 5min = 24h)
        previous_temps.pop(0)
    
    # Generate demand
    demand = generate_demand(current, temperature, is_holiday, is_weekend)
    
    # Create row
    rows.append({
        "timestamp": current,
        "hour": hour,
        "minute": minute,
        "day_of_week": day_of_week,
        "month": month,
        "is_holiday": is_holiday,
        "is_weekend": is_weekend,
        "temperature": round(temperature, 2),
        "demand": round(demand, 2)
    })
    
    current += interval
    data_points += 1
    
    # Progress indicator
    if data_points % 10000 == 0:
        print(f"Generated {data_points} data points...")

print(f"Total data points generated: {data_points}")

# Create DataFrame
df = pd.DataFrame(rows)

# Create additional features that might be useful
# Day name for readability
df['day_name'] = df['timestamp'].dt.day_name()

# Week of year
df['week_of_year'] = df['timestamp'].dt.isocalendar().week

# Quarter
df['quarter'] = df['timestamp'].dt.quarter

# Is it a workday? (not weekend and not holiday)
df['is_workday'] = ((df['is_weekend'] == 0) & (df['is_holiday'] == 0)).astype(int)

# Create the target variable: demand 5 minutes later
df["demand_next"] = df["demand"].shift(-1)

# Create lagged features for demand (useful for autoregressive models)
for lag in [1, 2, 3, 6, 12, 24]:  # 5min, 10min, 15min, 30min, 1h, 2h lags
    df[f'demand_lag_{lag}'] = df['demand'].shift(lag)

# Create lagged features for temperature
for lag in [1, 2, 3, 6, 12]:
    df[f'temperature_lag_{lag}'] = df['temperature'].shift(lag)

# Rolling statistics (moving averages)
df['demand_ma_12'] = df['demand'].rolling(window=12, min_periods=1).mean()  # 1-hour moving average
df['demand_ma_288'] = df['demand'].rolling(window=288, min_periods=1).mean()  # 24-hour moving average
df['temp_ma_12'] = df['temperature'].rolling(window=12, min_periods=1).mean()
df['temp_ma_288'] = df['temperature'].rolling(window=288, min_periods=1).mean()

# Remove rows with NaN values (from shifting and lagging)
initial_rows = len(df)
df = df.dropna().reset_index(drop=True)
dropped_rows = initial_rows - len(df)

print(f"Dropped {dropped_rows} rows with NaN values")
print(f"Final dataset size: {len(df)} rows")

# Save to CSV
output_file = "data/synthetic_demand_5min_v2.csv"
df.to_csv(output_file, index=False)
print(f"\nData saved to {output_file}")

# Display summary statistics
print("\n=== Dataset Summary ===")
print(df.info())
print("\n=== Demand Statistics ===")
print(df['demand'].describe())
print("\n=== Temperature Statistics ===")
print(df['temperature'].describe())

# Show sample data
print("\n=== First 10 rows ===")
print(df.head(10))
print("\n=== Sample of different conditions ===")
print("\nWeekday sample:")
print(df[df['is_workday'] == 1].head(3))
print("\nWeekend sample:")
print(df[df['is_weekend'] == 1].head(3))
print("\nHoliday sample:")
print(df[df['is_holiday'] == 1].head(3))

# Create some visualization of patterns
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    # Plot sample week of data
    sample_week = df[df['week_of_year'] == 25].copy()  # Week in June
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Demand over time
    axes[0].plot(sample_week['timestamp'], sample_week['demand'], 'b-', alpha=0.7)
    axes[0].set_xlabel('Date and Time', fontsize=12, labelpad=10)
    axes[0].set_ylabel('Demand (MW)', fontsize=12)
    axes[0].set_title('Sample Week - Demand Pattern', fontsize=14, pad=15)
    axes[0].grid(True, alpha=0.3)
    # Format x-axis to show dates nicely
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes[0].xaxis.set_major_locator(mdates.DayLocator())
    axes[0].tick_params(axis='x', rotation=45, labelsize=10)
    
    # Temperature over time
    axes[1].plot(sample_week['timestamp'], sample_week['temperature'], 'r-', alpha=0.7)
    axes[1].set_xlabel('Date and Time', fontsize=12, labelpad=10)
    axes[1].set_ylabel('Temperature (°C)', fontsize=12)
    axes[1].set_title('Sample Week - Temperature Pattern', fontsize=14, pad=15)
    axes[1].grid(True, alpha=0.3)
    # Format x-axis to show dates nicely
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes[1].xaxis.set_major_locator(mdates.DayLocator())
    axes[1].tick_params(axis='x', rotation=45, labelsize=10)
    
    # Demand vs Temperature scatter
    axes[2].scatter(df['temperature'].sample(5000), df['demand'].sample(5000), alpha=0.5, s=10)
    axes[2].set_xlabel('Temperature (°C)', fontsize=12, labelpad=10)
    axes[2].set_ylabel('Demand (MW)', fontsize=12)
    axes[2].set_title('Demand vs Temperature Relationship', fontsize=14, pad=15)
    axes[2].grid(True, alpha=0.3)
    
    # Adjust layout to prevent overlap
    plt.tight_layout(pad=3.0)
    
    # Add extra space at the bottom for rotated labels
    plt.subplots_adjust(bottom=0.1)
    
    plt.savefig('data/synthetic_demand_patterns_v2.png', dpi=150, bbox_inches='tight')
    print("\nPlots saved to data/synthetic_demand_patterns_v2.png")
    plt.show()
    
except ImportError:
    print("\nMatplotlib not installed. Skipping visualization.")

print("\n=== Data Generation Complete ===")
print("The dataset includes:")
print("- Full year of 5-minute interval data")
print("- Realistic temperature patterns (seasonal + daily)")
print("- Demand patterns with weekday/weekend/holiday effects")
print("- Temperature-dependent demand (heating/cooling)")
print("- All features required for Hong Tao's thesis methods")
print("- Lagged features for time series analysis")
print("- Moving averages for trend analysis")