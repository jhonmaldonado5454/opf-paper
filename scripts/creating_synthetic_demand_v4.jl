# Synthetic Load Data Generator V4 - FINAL WORKING VERSION
# Generates realistic hourly load data for comprehensive forecasting research
# Author: Advanced Load Forecasting System V4 - Final

using CSV, DataFrames
using Statistics
using Random
using Printf
using Plots
using Dates
using LinearAlgebra

# Set random seed for reproducibility
Random.seed!(42)

"""
Enhanced synthetic load data generator with improved features for statistical analysis
"""
function generate_enhanced_synthetic_data(;
    start_date = DateTime(2022, 1, 1),
    end_date = DateTime(2023, 12, 31),
    interval = Hour(1),
    output_dir = "data"  # Simplified path
)
    
    println("=" ^ 60)
    println("ENHANCED SYNTHETIC LOAD DATA GENERATOR V4 - FINAL")
    println("=" ^ 60)
    println("Generating comprehensive load data for advanced analysis...")
    println("Period: $start_date to $end_date")
    
    # Create output directory if it doesn't exist
    if !isdir(output_dir)
        mkpath(output_dir)
        println("Created output directory: $output_dir")
    end
    
    # Generate timestamps
    timestamps = collect(start_date:interval:end_date)
    n_points = length(timestamps)
    println("Total data points: $n_points")
    
    # Initialize enhanced DataFrame with proper column names
    df = DataFrame()
    df.timestamp = timestamps
    df.hour = hour.(timestamps)
    df.day_of_week = dayofweek.(timestamps)
    df.month = month.(timestamps)
    df.week_of_year = week.(timestamps)
    df.day_of_year = dayofyear.(timestamps)
    df.quarter = Dates.quarterofyear.(timestamps)
    
    # Enhanced holiday calendar
    holidays = [
        # 2022
        Date(2022, 1, 1),   # New Year
        Date(2022, 2, 21),  # Presidents Day
        Date(2022, 5, 30),  # Memorial Day
        Date(2022, 7, 4),   # Independence Day
        Date(2022, 9, 5),   # Labor Day
        Date(2022, 11, 24), # Thanksgiving
        Date(2022, 12, 25), # Christmas
        Date(2022, 12, 26), # Christmas observed
        # 2023
        Date(2023, 1, 1),   # New Year
        Date(2023, 2, 20),  # Presidents Day
        Date(2023, 5, 29),  # Memorial Day
        Date(2023, 7, 4),   # Independence Day
        Date(2023, 9, 4),   # Labor Day
        Date(2023, 11, 23), # Thanksgiving
        Date(2023, 12, 25), # Christmas
    ]
    
    df.is_holiday = [Date(ts) in holidays for ts in timestamps]
    df.is_weekend = df.day_of_week .>= 6
    df.is_business_day = (.!df.is_weekend) .* (.!df.is_holiday)
    
    println("Generating enhanced temperature patterns...")
    
    # Enhanced temperature model with multiple components
    df.temperature = zeros(n_points)
    df.temperature_trend = zeros(n_points)
    df.temperature_seasonal = zeros(n_points)
    df.temperature_daily = zeros(n_points)
    df.temperature_noise = zeros(n_points)
    
    base_temp = 15.0
    
    for i in 1:n_points
        # Long-term trend (climate change effect)
        trend = 0.01 * (i / n_points) * 365  # 0.01°C per year
        df.temperature_trend[i] = trend
        
        # Seasonal component (enhanced)
        day_of_year = df.day_of_year[i]
        seasonal = 12 * sin(2π * (day_of_year - 80) / 365) + 
                  3 * sin(4π * (day_of_year - 80) / 365)  # Harmonics
        df.temperature_seasonal[i] = seasonal
        
        # Daily component (enhanced)
        hour_of_day = df.hour[i]
        daily = 6 * sin(2π * (hour_of_day - 6) / 24) + 
               1.5 * sin(4π * (hour_of_day - 6) / 24)  # Harmonics
        df.temperature_daily[i] = daily
        
        # Weather persistence with seasonal variance
        seasonal_noise_factor = 1.0 + 0.3 * abs(sin(2π * day_of_year / 365))
        if i > 1
            persistence = 0.85 * (df.temperature[i-1] - base_temp - trend - seasonal)
            noise = randn() * 2.5 * seasonal_noise_factor
        else
            persistence = 0.0
            noise = randn() * 2.5
        end
        df.temperature_noise[i] = persistence + noise
        
        # Combine all components
        df.temperature[i] = base_temp + trend + seasonal + daily + df.temperature_noise[i]
    end
    
    # Enhanced temperature-derived features
    df.temp_squared = df.temperature .^ 2
    df.temp_cubed = df.temperature .^ 3
    df.temp_log = log.(max.(df.temperature .+ 50, 1))  # Avoid log of negative
    df.cooling_degree_days = max.(df.temperature .- 22, 0)
    df.heating_degree_days = max.(18 .- df.temperature, 0)
    df.temperature_volatility = [i > 24 ? std(df.temperature[max(1,i-23):i]) : 0.0 for i in 1:n_points]
    
    println("Generating enhanced load patterns...")
    
    # Enhanced load model with multiple components
    df.load_base = zeros(n_points)
    df.load_seasonal = zeros(n_points)
    df.load_daily = zeros(n_points)
    df.load_weather = zeros(n_points)
    df.load_economic = zeros(n_points)
    df.load_noise = zeros(n_points)
    
    for i in 1:n_points
        h = df.hour[i]
        dow = df.day_of_week[i]
        temp = df.temperature[i]
        month = df.month[i]
        quarter = df.quarter[i]
        
        # Base load with seasonal variation
        base_seasonal = 320 + 25 * sin(2π * (month - 1) / 12) + 
                       10 * sin(4π * (month - 1) / 12)
        df.load_seasonal[i] = base_seasonal
        
        # Enhanced daily patterns
        if df.is_weekend[i]
            # Weekend pattern
            morning_peak = 20 * exp(-((h - 10)^2) / 20)
            evening_peak = 30 * exp(-((h - 19)^2) / 25)
            daily_pattern = base_seasonal * 0.8 + morning_peak + evening_peak
        elseif df.is_holiday[i]
            # Holiday pattern (similar to weekend but lower)
            morning_peak = 15 * exp(-((h - 11)^2) / 25)
            evening_peak = 25 * exp(-((h - 18)^2) / 20)
            daily_pattern = base_seasonal * 0.7 + morning_peak + evening_peak
        else
            # Enhanced weekday pattern
            morning_peak = 50 * exp(-((h - 8)^2) / 10)
            lunch_peak = 20 * exp(-((h - 12)^2) / 6)
            evening_peak = 70 * exp(-((h - 18)^2) / 15)
            late_evening = 10 * exp(-((h - 22)^2) / 8)
            daily_pattern = base_seasonal + morning_peak + lunch_peak + evening_peak + late_evening
        end
        df.load_daily[i] = daily_pattern - base_seasonal
        
        # Enhanced weather effects
        if temp < 12  # Heating
            heating_effect = 4.0 * (12 - temp) * (1 + 0.1 * sin(2π * h / 24))
            cooling_effect = 0.0
        elseif temp > 26  # Cooling
            heating_effect = 0.0
            cooling_effect = 5.0 * (temp - 26) * (1 + 0.15 * sin(2π * (h - 12) / 24))
        else
            heating_effect = 0.0
            cooling_effect = 0.0
        end
        df.load_weather[i] = heating_effect + cooling_effect
        
        # Economic and growth factors
        growth_trend = 0.08 * (i / n_points)  # 8% growth over period
        quarterly_cycle = 0.02 * sin(2π * quarter / 4)  # Quarterly business cycle
        df.load_economic[i] = (daily_pattern + df.load_weather[i]) * (growth_trend + quarterly_cycle)
        
        # Combine base components
        df.load_base[i] = daily_pattern + df.load_weather[i] + df.load_economic[i]
    end
    
    # Enhanced noise model with heteroskedasticity
    for i in 1:n_points
        # Time-varying noise (higher during peaks)
        if df.hour[i] in [7, 8, 9, 17, 18, 19, 20]
            base_noise = 12.0  # Higher volatility during peaks
        elseif df.hour[i] in [0, 1, 2, 3, 4, 5]
            base_noise = 4.0   # Lower volatility at night
        else
            base_noise = 8.0   # Medium volatility
        end
        
        # Add day-of-week effect on noise
        if df.is_weekend[i]
            noise_multiplier = 0.8
        elseif df.is_holiday[i]
            noise_multiplier = 0.6
        else
            noise_multiplier = 1.0
        end
        
        # Autoregressive noise
        if i > 1
            ar_component = 0.4 * df.load_noise[i-1]
        else
            ar_component = 0.0
        end
        
        df.load_noise[i] = ar_component + randn() * base_noise * noise_multiplier
    end
    
    # Final load construction
    df.load = df.load_base + df.load_noise
    
    # Ensure realistic load bounds
    df.load = max.(df.load, 80.0)   # Minimum load
    df.load = min.(df.load, 800.0)  # Maximum load
    
    # Enhanced lag features (corrected with proper bounds checking)
    df.load_lag_1h = [i > 1 ? df.load[i-1] : df.load[i] for i in 1:n_points]
    df.load_lag_2h = [i > 2 ? df.load[i-2] : df.load[i] for i in 1:n_points]
    df.load_lag_24h = [i > 24 ? df.load[i-24] : df.load[i] for i in 1:n_points]
    df.load_lag_48h = [i > 48 ? df.load[i-48] : df.load[i] for i in 1:n_points]
    df.load_lag_168h = [i > 168 ? df.load[i-168] : df.load[i] for i in 1:n_points]
    df.load_lag_336h = [i > 336 ? df.load[i-336] : df.load[i] for i in 1:n_points]
    
    # Rolling statistics
    df.load_ma_24h = [i >= 24 ? mean(df.load[max(1,i-23):i]) : df.load[i] for i in 1:n_points]
    df.load_std_24h = [i >= 24 ? std(df.load[max(1,i-23):i]) : 0.0 for i in 1:n_points]
    df.load_ma_168h = [i >= 168 ? mean(df.load[max(1,i-167):i]) : df.load[i] for i in 1:n_points]
    
    # Target variables for different horizons (corrected with proper bounds checking)
    df.load_1h = [i < n_points ? df.load[i+1] : df.load[i] for i in 1:n_points]
    df.load_6h = [i <= n_points-6 ? df.load[i+6] : df.load[i] for i in 1:n_points]
    df.load_12h = [i <= n_points-12 ? df.load[i+12] : df.load[i] for i in 1:n_points]
    df.load_24h = [i <= n_points-24 ? df.load[i+24] : df.load[i] for i in 1:n_points]
    df.load_48h = [i <= n_points-48 ? df.load[i+48] : df.load[i] for i in 1:n_points]
    df.load_168h = [i <= n_points-168 ? df.load[i+168] : df.load[i] for i in 1:n_points]
    df.load_360h = [i <= n_points-360 ? df.load[i+360] : df.load[i] for i in 1:n_points]
    
    # Remove rows with problematic values (keep more data)
    valid_range = 169:(n_points-169)  # Keep more data by reducing the buffer
    df = df[valid_range, :]
    
    println("Enhanced data generation complete!")
    println("Final dataset size: $(nrow(df)) rows × $(ncol(df)) columns")
    
    # Enhanced summary statistics
    println("\n" * "=" ^ 50)
    println("ENHANCED LOAD STATISTICS")
    println("=" ^ 50)
    println("Mean load: $(round(mean(df.load), digits=2)) MW")
    println("Median load: $(round(median(df.load), digits=2)) MW")
    println("Min load: $(round(minimum(df.load), digits=2)) MW")
    println("Max load: $(round(maximum(df.load), digits=2)) MW")
    println("Load standard deviation: $(round(std(df.load), digits=2)) MW")
    println("Load coefficient of variation: $(round(std(df.load)/mean(df.load), digits=3))")
    
    println("\n" * "=" ^ 50)
    println("ENHANCED TEMPERATURE STATISTICS")
    println("=" ^ 50)
    println("Mean temperature: $(round(mean(df.temperature), digits=2)) °C")
    println("Median temperature: $(round(median(df.temperature), digits=2)) °C")
    println("Min temperature: $(round(minimum(df.temperature), digits=2)) °C")
    println("Max temperature: $(round(maximum(df.temperature), digits=2)) °C")
    println("Temperature std: $(round(std(df.temperature), digits=2)) °C")
    
    # Save enhanced data
    output_file = joinpath(output_dir, "synthetic_load_data_v4.csv")
    CSV.write(output_file, df)
    println("\nEnhanced data saved to: $output_file")
    
    return df
end

"""
Enhanced data visualization with clear and professional charts
"""
function create_enhanced_data_visualization(df, output_dir = "results")
    println("\n" * "=" ^ 50)
    println("GENERATING ENHANCED DATA VISUALIZATIONS")
    println("=" ^ 50)
    
    # Create results directory if it doesn't exist
    if !isdir(output_dir)
        mkpath(output_dir)
        println("Created results directory: $output_dir")
    end
    
    # Extended sample for visualization
    n_sample = min(24*45, nrow(df))  # 45 days
    
    # 1. Load time series (clean and professional)
    p1 = plot(df.timestamp[1:n_sample], df.load[1:n_sample], 
        lw=2, color=:blue, label="Load Demand",
        title="Electrical Load Demand Pattern (45 Days)")
    xlabel!(p1, "Date")
    ylabel!(p1, "Load (MW)")
    
    # 2. Temperature time series (separate and clear)
    p2 = plot(df.timestamp[1:n_sample], df.temperature[1:n_sample], 
        lw=2, color=:red, label="Temperature",
        title="Temperature Pattern (45 Days)")
    xlabel!(p2, "Date")
    ylabel!(p2, "Temperature (°C)")
    
    # Add reference lines for temperature
    hline!(p2, [0], color=:gray, ls=:dash, alpha=0.5, label="Freezing Point")
    hline!(p2, [25], color=:orange, ls=:dash, alpha=0.5, label="Comfort Zone")
    
    # 3. Enhanced daily patterns comparison
    hours = 0:23
    
    # Business day patterns - Fixed boolean indexing
    business_pattern = [mean(df.load[(df.hour .== h) .* df.is_business_day]) for h in hours]
    weekend_pattern = [mean(df.load[(df.hour .== h) .* df.is_weekend]) for h in hours]
    holiday_pattern = [mean(df.load[(df.hour .== h) .* df.is_holiday]) for h in hours]
    
    p3 = plot(hours, business_pattern, lw=3, color=:blue, marker=:circle,
        label="Business Days", title="Average Daily Load Patterns by Day Type")
    plot!(p3, hours, weekend_pattern, lw=3, color=:red, marker=:square,
        label="Weekends")
    plot!(p3, hours, holiday_pattern, lw=3, color=:green, marker=:diamond,
        label="Holidays")
    xlabel!(p3, "Hour of Day")
    ylabel!(p3, "Average Load (MW)")
    
    # Add peak hour indicators
    vline!(p3, [8], color=:yellow, alpha=0.3, lw=8, label="Morning Peak")
    vline!(p3, [18], color=:orange, alpha=0.3, lw=8, label="Evening Peak")
    
    # 4. Load vs Temperature scatter plot (professional analysis)
    sample_size = min(5000, nrow(df))
    p4 = scatter(df.temperature[1:sample_size], df.load[1:sample_size], 
        alpha=0.4, color=:purple, markersize=2,
        title="Load-Temperature Relationship Analysis", label="Data Points")
    
    # Add trend lines for different temperature regions - Fixed comparison syntax
    temp_low = df.temperature .< 15
    temp_medium = (df.temperature .>= 15) .* (df.temperature .< 25)
    temp_high = df.temperature .>= 25
    
    if sum(temp_low) > 0
        scatter!(p4, df.temperature[temp_low][1:min(1000, sum(temp_low))], 
                df.load[temp_low][1:min(1000, sum(temp_low))],
                alpha=0.5, color=:blue, markersize=2, label="Cold (<15°C)")
    end
    if sum(temp_medium) > 0
        scatter!(p4, df.temperature[temp_medium][1:min(1000, sum(temp_medium))], 
                df.load[temp_medium][1:min(1000, sum(temp_medium))],
                alpha=0.5, color=:green, markersize=2, label="Moderate (15-25°C)")
    end
    if sum(temp_high) > 0
        scatter!(p4, df.temperature[temp_high][1:min(1000, sum(temp_high))], 
                df.load[temp_high][1:min(1000, sum(temp_high))],
                alpha=0.5, color=:red, markersize=2, label="Hot (>25°C)")
    end
    
    xlabel!(p4, "Temperature (°C)")
    ylabel!(p4, "Load (MW)")
    
    # Combine plots with professional layout
    combined_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1800, 1400),
        plot_title="Enhanced Synthetic Load Data V4 - Professional Analysis Dashboard")
    
    # Save enhanced visualization
    output_file = joinpath(output_dir, "enhanced_data_visualization_v4.png")
    savefig(combined_plot, output_file)
    println("Enhanced professional visualization saved to: $output_file")
    
    display(combined_plot)
    
    return combined_plot
end

"""
Main execution function for enhanced data generation
"""
function main_enhanced()
    println("Starting Enhanced Synthetic Load Data Generation V4...")
    
    # Generate enhanced data
    df = generate_enhanced_synthetic_data()
    
    # Create enhanced visualizations
    viz_result = create_enhanced_data_visualization(df)
    
    println("\n" * "=" ^ 60)
    println("ENHANCED DATA GENERATION V4 COMPLETE")
    println("=" ^ 60)
    println("Files created:")
    println("- synthetic_load_data_v4.csv (in data folder)")
    println("- enhanced_data_visualization_v4.png (in results folder)")
    println("\nReady for advanced modeling and analysis!")
    
    return df
end

# Execute enhanced generation
println("🚀 Running Final Data Generator...")
df_enhanced = main_enhanced()