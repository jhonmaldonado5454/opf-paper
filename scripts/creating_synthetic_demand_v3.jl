# Synthetic Load Data Generator
# Generates realistic hourly load data for forecasting research
# Based on Prof. Gonzalo's requirements

using CSV, DataFrames
using Statistics
using Random
using Printf
using Plots
using Dates

# Set random seed for reproducibility
Random.seed!(42)

"""
Generates realistic synthetic load data with complex patterns
"""
function generate_synthetic_load_data(;
    start_date = DateTime(2022, 1, 1),
    end_date = DateTime(2023, 12, 31),
    interval = Hour(1),  # Hourly resolution
    output_dir = raw"C:\Users\jhonm\Visual SC Projects\opf-paper\data"
)
    
    println("=== Synthetic Load Data Generator ===")
    println("Generating realistic load data...")
    println("Period: $start_date to $end_date")
    
    # Generate timestamps
    timestamps = collect(start_date:interval:end_date)
    n_points = length(timestamps)
    println("Total data points: $n_points")
    
    # Initialize DataFrame
    df = DataFrame()
    df.timestamp = timestamps
    df.hour = hour.(timestamps)
    df.day_of_week = dayofweek.(timestamps)
    df.month = month.(timestamps)
    df.week_of_year = week.(timestamps)
    df.day_of_year = dayofyear.(timestamps)
    
    # Define holidays (major US holidays)
    holidays = [
        # 2022
        Date(2022, 1, 1),   # New Year
        Date(2022, 7, 4),   # Independence Day
        Date(2022, 11, 24), # Thanksgiving
        Date(2022, 12, 25), # Christmas
        # 2023
        Date(2023, 1, 1),   # New Year
        Date(2023, 7, 4),   # Independence Day
        Date(2023, 11, 23), # Thanksgiving
        Date(2023, 12, 25)  # Christmas
    ]
    
    df.is_holiday = [Date(ts) in holidays for ts in timestamps]
    df.is_weekend = df.day_of_week .>= 6
    
    println("Generating temperature patterns...")
    
    # Generate realistic temperature with seasonal and daily patterns
    df.temperature = zeros(n_points)
    base_temp = 15.0  # Base temperature in Celsius
    
    for i in 1:n_points
        # Seasonal component (sine wave over the year)
        seasonal = 10 * sin(2π * (df.day_of_year[i] - 80) / 365)
        
        # Daily component (temperature variation during day)
        daily = 4 * sin(2π * (df.hour[i] - 6) / 24)
        
        # Weather persistence (autocorrelation)
        if i > 1
            persistence = 0.85 * (df.temperature[i-1] - base_temp - seasonal)
        else
            persistence = 0.0
        end
        
        # Random weather variations
        random_variation = randn() * 2.0
        
        # Combine all components
        df.temperature[i] = base_temp + seasonal + daily + persistence + random_variation
    end
    
    # Temperature-derived features (Hong Tao methodology)
    df.temp_squared = df.temperature .^ 2
    df.temp_cubed = df.temperature .^ 3
    df.cooling_degree_days = max.(df.temperature .- 22, 0)  # Cooling threshold at 22°C
    df.heating_degree_days = max.(18 .- df.temperature, 0)  # Heating threshold at 18°C
    
    println("Generating load patterns...")
    
    # Generate sophisticated load patterns
    df.load_base = zeros(n_points)
    
    for i in 1:n_points
        h = df.hour[i]
        dow = df.day_of_week[i]
        temp = df.temperature[i]
        month = df.month[i]
        
        # Base load level (varies by month)
        base_level = 320 + 30 * sin(2π * (month - 1) / 12)  # Seasonal base load
        
        # Hourly patterns
        if df.is_weekend[i]
            # Weekend pattern: later morning peak, lower overall
            morning_peak = 25 * exp(-((h - 10)^2) / 16)  # Later, gentler peak
            evening_peak = 35 * exp(-((h - 19)^2) / 20)  # Evening entertainment
            hourly_pattern = base_level * 0.85 + morning_peak + evening_peak
        else
            # Weekday pattern: sharp morning and evening peaks
            morning_peak = 45 * exp(-((h - 8)^2) / 8)   # Sharp morning peak
            evening_peak = 65 * exp(-((h - 18)^2) / 12) # Sharp evening peak
            lunch_peak = 15 * exp(-((h - 12)^2) / 4)    # Lunch peak
            hourly_pattern = base_level + morning_peak + evening_peak + lunch_peak
        end
        
        # Temperature effects (air conditioning and heating)
        temp_effect = 0.0
        if temp < 15  # Heating load
            temp_effect = 3.0 * (15 - temp)
        elseif temp > 24  # Cooling load
            temp_effect = 4.0 * (temp - 24)
        end
        
        # Holiday effects
        if df.is_holiday[i]
            holiday_factor = 0.75  # Reduced load on holidays
        else
            holiday_factor = 1.0
        end
        
        # Economic growth trend
        growth_trend = 1.0 + 0.05 * (i / n_points)  # 5% growth over period
        
        # Industrial load component (weekday only)
        if !df.is_weekend[i] && h >= 7 && h <= 17
            industrial_load = 40 * (1 + 0.2 * sin(2π * h / 24))
        else
            industrial_load = 10  # Base industrial load
        end
        
        # Combine all components
        df.load_base[i] = (hourly_pattern + temp_effect + industrial_load) * 
                         holiday_factor * growth_trend
    end
    
    # Add realistic noise and volatility
    # Different noise levels for different times
    noise = zeros(n_points)
    for i in 1:n_points
        # Higher volatility during peak hours
        if df.hour[i] in [7, 8, 9, 17, 18, 19, 20]
            noise_std = 8.0  # Higher volatility during peaks
        else
            noise_std = 4.0  # Lower volatility during off-peak
        end
        
        # Add autocorrelated noise
        if i > 1
            noise[i] = 0.3 * noise[i-1] + randn() * noise_std
        else
            noise[i] = randn() * noise_std
        end
    end
    
    # Final load with noise
    df.load = df.load_base + noise
    
    # Ensure no negative loads
    df.load = max.(df.load, 50.0)  # Minimum load of 50 MW
    
    # Create lag features (previous hours)
    df.load_lag_1h = circshift(df.load, 1)
    df.load_lag_24h = circshift(df.load, 24)
    df.load_lag_168h = circshift(df.load, 168)  # Week lag
    
    # Create target variables for different forecasting horizons
    df.load_1h = circshift(df.load, -1)     # 1 hour ahead
    df.load_24h = circshift(df.load, -24)   # 24 hours ahead (1 day)
    df.load_168h = circshift(df.load, -168) # 168 hours ahead (1 week)
    df.load_360h = circshift(df.load, -360) # 360 hours ahead (15 days)
    
    # Remove rows with missing values due to shifting
    valid_range = 360:(n_points-360)
    df = df[valid_range, :]
    
    println("Data generation complete!")
    println("Final dataset size: $(nrow(df)) rows × $(ncol(df)) columns")
    
    # Generate summary statistics
    println("\n=== Load Statistics ===")
    println("Mean load: $(round(mean(df.load), digits=2)) MW")
    println("Min load: $(round(minimum(df.load), digits=2)) MW")
    println("Max load: $(round(maximum(df.load), digits=2)) MW")
    println("Load standard deviation: $(round(std(df.load), digits=2)) MW")
    
    println("\n=== Temperature Statistics ===")
    println("Mean temperature: $(round(mean(df.temperature), digits=2)) °C")
    println("Min temperature: $(round(minimum(df.temperature), digits=2)) °C")
    println("Max temperature: $(round(maximum(df.temperature), digits=2)) °C")
    
    # Save data
    output_file = joinpath(output_dir, "synthetic_load_data_v3.csv")
    CSV.write(output_file, df)
    println("\nData saved to: $output_file")
    
    return df
end

"""
Generates visualization of the synthetic data for quality check
"""
function visualize_synthetic_data(df, output_dir = raw"C:\Users\jhonm\Visual SC Projects\opf-paper\results")
    println("\n=== Generating Data Visualization ===")
    
    # Sample first month for detailed view
    n_month = min(24*30, nrow(df))  # 30 days
    
    # 1. Load and temperature time series
    p1 = plot(df.timestamp[1:n_month], df.load[1:n_month], 
        lw=2, color=:blue, label="Load (MW)",
        title="Load and Temperature - First Month")
    
    # Add temperature on secondary axis (scaled)
    temp_scaled = df.temperature[1:n_month] .* 10 .+ 200  # Scale for visibility
    plot!(p1, df.timestamp[1:n_month], temp_scaled, 
        lw=2, color=:red, label="Temperature (scaled)", alpha=0.7)
    
    xlabel!(p1, "Date")
    ylabel!(p1, "Load (MW)")
    
    # 2. Daily load patterns
    hours = 0:23
    weekday_pattern = [mean(df.load[df.hour .== h .&& df.day_of_week .<= 5]) for h in hours]
    weekend_pattern = [mean(df.load[df.hour .== h .&& df.day_of_week .> 5]) for h in hours]
    
    p2 = plot(hours, weekday_pattern, lw=3, color=:blue, marker=:circle,
        label="Weekday", title="Average Daily Load Patterns")
    plot!(p2, hours, weekend_pattern, lw=3, color=:red, marker=:square,
        label="Weekend")
    xlabel!(p2, "Hour of Day")
    ylabel!(p2, "Average Load (MW)")
    
    # 3. Monthly patterns
    months = 1:12
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_avg = [mean(df.load[df.month .== m]) for m in months]
    
    p3 = plot(months, monthly_avg, lw=3, color=:green, marker=:circle,
        title="Monthly Load Patterns", label="Average Load")
    xlabel!(p3, "Month")
    ylabel!(p3, "Average Load (MW)")
    xticks!(p3, months, month_names)
    
    # 4. Load vs Temperature scatter
    p4 = scatter(df.temperature[1:min(5000, end)], df.load[1:min(5000, end)], 
        alpha=0.4, color=:purple, markersize=2,
        title="Load vs Temperature Relationship", label="Data Points")
    xlabel!(p4, "Temperature (°C)")
    ylabel!(p4, "Load (MW)")
    
    # Combine plots
    combined_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1600, 1200),
        plot_title="Synthetic Load Data Quality Check",
        plot_titlefont=font(16, "bold"))
    
    # Save visualization
    output_file = joinpath(output_dir, "synthetic_data_visualization_v3.png")
    savefig(combined_plot, output_file)
    println("Visualization saved to: $output_file")
    
    display(combined_plot)
    
    return combined_plot
end

"""
Main function to generate and visualize synthetic data
"""
function main()
    println("Starting synthetic load data generation...")
    
    # Generate data
    df = generate_synthetic_load_data()
    
    # Create visualization
    plot_result = visualize_synthetic_data(df)
    
    println("\n=== Generation Complete ===")
    println("Files created:")
    println("- synthetic_load_data_v3.csv (in data folder)")
    println("- synthetic_data_visualization_v3.png (in results folder)")
    
    return df
end

# Execute the generation
df = main()