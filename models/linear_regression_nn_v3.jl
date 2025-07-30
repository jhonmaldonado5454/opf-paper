# Decision-Oriented Load Forecasting System
# Based on Prof. Gonzalo's requirements
# Objective: 24h model that mimics decisions from 15-day model

using CSV, DataFrames
using Flux
using Statistics
using Random
using Printf
using Plots
using Dates
using LinearAlgebra

# ========== PART 1: ENHANCED SYNTHETIC DATA GENERATION ==========

"""
Generates realistic synthetic data for load forecasting with operational patterns
"""
function generate_synthetic_operational_data(;
    start_date = DateTime(2022, 1, 1),
    end_date = DateTime(2023, 12, 31),
    interval = Hour(1),  # Hourly resolution as requested
    include_battery_operations = true
)
    
    println("Generating synthetic operational data...")
    
    # Generate timestamps
    timestamps = collect(start_date:interval:end_date)
    n_points = length(timestamps)
    
    # Initialize DataFrame
    df = DataFrame()
    df.timestamp = timestamps
    df.hour = hour.(timestamps)
    df.day_of_week = dayofweek.(timestamps)
    df.month = month.(timestamps)
    df.week_of_year = week.(timestamps)
    
    # Major holidays
    holidays = [
        Date(2022, 1, 1), Date(2022, 7, 4), Date(2022, 12, 25),
        Date(2023, 1, 1), Date(2023, 7, 4), Date(2023, 12, 25)
    ]
    df.is_holiday = [Date(ts) in holidays for ts in timestamps]
    df.is_weekend = df.day_of_week .>= 6
    
    # Generate temperature with realistic patterns
    df.temperature = zeros(n_points)
    for i in 1:n_points
        # Seasonal component
        day_of_year = dayofyear(timestamps[i])
        seasonal = 15 + 10 * sin(2π * (day_of_year - 80) / 365)
        
        # Daily component
        hour_of_day = hour(timestamps[i])
        daily = 5 * sin(2π * (hour_of_day - 6) / 24)
        
        # Random component with autocorrelation
        random = i > 1 ? 0.8 * df.temperature[i-1] + 0.2 * randn() : randn()
        
        df.temperature[i] = seasonal + daily + random
    end
    
    # Generate base demand with complex patterns
    df.demand_base = zeros(n_points)
    
    for i in 1:n_points
        h = df.hour[i]
        dow = df.day_of_week[i]
        temp = df.temperature[i]
        
        # Base hourly pattern (double peak)
        if df.is_weekend[i]
            # Weekend pattern
            hourly_pattern = 300 + 50 * sin(2π * (h - 10) / 24)
        else
            # Weekday pattern with morning and evening peaks
            morning_peak = 40 * exp(-((h - 8)^2) / 8)
            evening_peak = 60 * exp(-((h - 18)^2) / 12)
            hourly_pattern = 350 + morning_peak + evening_peak
        end
        
        # Temperature effect (heating and cooling)
        temp_effect = 0
        if temp < 10
            temp_effect = 2 * (10 - temp)  # Heating
        elseif temp > 25
            temp_effect = 3 * (temp - 25)  # Cooling
        end
        
        # Seasonal effect
        seasonal_factor = 1.0 + 0.1 * sin(2π * df.week_of_year[i] / 52)
        
        # Holiday effect
        holiday_factor = df.is_holiday[i] ? 0.85 : 1.0
        
        # Growth trend
        growth_trend = 0.1 * (i / n_points)
        
        # Combine effects
        df.demand_base[i] = (hourly_pattern + temp_effect) * seasonal_factor * holiday_factor * (1 + growth_trend)
    end
    
    # Add realistic noise
    df.demand = df.demand_base + randn(n_points) * 5
    
    # Temperature variables (Hong Tao features)
    df.temp_squared = df.temperature .^ 2
    df.temp_cubed = df.temperature .^ 3
    
    # If including battery operations
    if include_battery_operations
        # Simulate battery decisions based on price/demand
        df.battery_charge = zeros(n_points)
        df.battery_discharge = zeros(n_points)
        df.battery_soc = zeros(n_points)  # State of Charge
        
        battery_capacity = 100.0  # MWh
        max_power = 20.0  # MW
        soc = battery_capacity * 0.5  # Start at 50%
        
        for i in 1:n_points
            # Simple logic: charge during off-peak, discharge during peaks
            if df.hour[i] in [2, 3, 4, 5]  # Off-peak hours
                charge = min(max_power, battery_capacity - soc)
                df.battery_charge[i] = charge
                soc += charge
            elseif df.hour[i] in [18, 19, 20] && !df.is_weekend[i]  # Evening peaks
                discharge = min(max_power, soc)
                df.battery_discharge[i] = discharge
                soc -= discharge
            end
            
            df.battery_soc[i] = soc
        end
        
        # Net demand after battery
        df.net_demand = df.demand + df.battery_charge - df.battery_discharge
    else
        df.net_demand = df.demand
    end
    
    # Create target variables for different horizons
    # 24 hours ahead
    df.demand_24h = circshift(df.demand, -24)
    df.net_demand_24h = circshift(df.net_demand, -24)
    
    # 15 days (360 hours) ahead
    df.demand_15d = circshift(df.demand, -360)
    df.net_demand_15d = circshift(df.net_demand, -360)
    
    # Remove rows with missing values
    df = df[1:end-360, :]
    
    return df
end

# ========== PART 2: FORECASTING MODELS ==========

"""
Model A: 15-day horizon (reference)
"""
function train_model_15days(df_train, features)
    println("\n=== Training Model A (15 days) ===")
    
    # Prepare data
    X = Matrix{Float32}(df_train[!, features])
    y = Float32.(df_train.demand_15d)
    
    # Normalize
    mean_X = mean(X, dims=1)
    std_X = std(X, dims=1)
    std_X[std_X .== 0] .= 1
    X_norm = (X .- mean_X) ./ std_X
    
    # More complex model for long horizon
    model = Chain(
        Dense(size(X, 2), 128, relu),
        Dropout(0.3),
        Dense(128, 64, relu),
        Dropout(0.2),
        Dense(64, 32, relu),
        Dense(32, 1)
    )
    
    # Train
    opt = ADAM(0.001)
    state = Flux.setup(opt, model)
    
    losses = Float32[]
    for epoch in 1:300
        grads = gradient(m -> mean((m(X_norm') .- y').^2), model)
        Flux.update!(state, model, grads[1])
        
        if epoch % 50 == 0
            loss = mean((model(X_norm') .- y').^2)
            push!(losses, loss)
            println("Epoch $epoch - Loss: $loss")
        end
    end
    
    return model, mean_X, std_X, losses
end

"""
Model B: 24-hour horizon (target)
"""
function train_model_24hours(df_train, features, model_15d=nothing, decisions_15d=nothing)
    println("\n=== Training Model B (24 hours) ===")
    
    # Prepare data
    X = Matrix{Float32}(df_train[!, features])
    y = Float32.(df_train.demand_24h)
    
    # Normalize
    mean_X = mean(X, dims=1)
    std_X = std(X, dims=1)
    std_X[std_X .== 0] .= 1
    X_norm = (X .- mean_X) ./ std_X
    
    # Simpler model for short horizon
    model = Chain(
        Dense(size(X, 2), 64, relu),
        Dropout(0.2),
        Dense(64, 32, relu),
        Dense(32, 1)
    )
    
    # Custom loss function
    function custom_loss(m, x, y_true)
        y_pred = m(x)
        
        # Standard prediction loss
        pred_loss = mean((y_pred .- y_true).^2)
        
        # If we have decisions from 15-day model, add imitation loss
        if !isnothing(decisions_15d)
            # Here we would implement logic to penalize decision differences
            decision_loss = 0.0  # Placeholder
            return pred_loss + 0.1 * decision_loss
        else
            return pred_loss
        end
    end
    
    # Train
    opt = ADAM(0.01)  # Higher learning rate for simpler model
    state = Flux.setup(opt, model)
    
    losses = Float32[]
    for epoch in 1:200
        grads = gradient(m -> custom_loss(m, X_norm', y'), model)
        Flux.update!(state, model, grads[1])
        
        if epoch % 20 == 0
            loss = custom_loss(model, X_norm', y')
            push!(losses, loss)
            println("Epoch $epoch - Loss: $loss")
        end
    end
    
    return model, mean_X, std_X, losses
end

# ========== PART 3: OPERATIONAL DECISION SIMULATION ==========

"""
Simulates battery decisions based on forecasts
"""
function simulate_battery_decisions(predictions, battery_capacity=100.0, max_power=20.0)
    n = length(predictions)
    decisions = DataFrame(
        charge = zeros(n),
        discharge = zeros(n),
        soc = zeros(n),
        revenue = zeros(n)
    )
    
    soc = battery_capacity * 0.5  # Start at 50%
    
    # Synthetic prices (correlated with demand)
    prices = 30 .+ 0.1 * (predictions .- mean(predictions))
    
    for i in 1:n
        # Simple strategy: buy low, sell high
        price_percentile = (prices[i] - minimum(prices)) / (maximum(prices) - minimum(prices))
        
        if price_percentile < 0.3 && soc < battery_capacity
            # Charge when price is low
            charge = min(max_power, battery_capacity - soc)
            decisions.charge[i] = charge
            decisions.revenue[i] = -charge * prices[i]  # Cost
            soc += charge
        elseif price_percentile > 0.7 && soc > 0
            # Discharge when price is high
            discharge = min(max_power, soc)
            decisions.discharge[i] = discharge
            decisions.revenue[i] = discharge * prices[i]  # Revenue
            soc -= discharge
        end
        
        decisions.soc[i] = soc
    end
    
    return decisions
end

# ========== PART 4: EVALUATION AND COMPARISON ==========

"""
Evaluates prediction and decision performance
"""
function evaluate_models(model_15d, model_24h, df_test, features, norm_params_15d, norm_params_24h)
    println("\n=== Model Evaluation ===")
    
    # Prepare test data
    X_test = Matrix{Float32}(df_test[!, features])
    
    # Normalize with training parameters
    X_test_15d = (X_test .- norm_params_15d.mean) ./ norm_params_15d.std
    X_test_24h = (X_test .- norm_params_24h.mean) ./ norm_params_24h.std
    
    # Predictions
    pred_15d = vec(model_15d(X_test_15d'))
    pred_24h = vec(model_24h(X_test_24h'))
    
    # Prediction metrics
    actual_15d = df_test.demand_15d
    actual_24h = df_test.demand_24h
    
    metrics = DataFrame(
        Model = ["15 days", "24 hours"],
        RMSE = [
            sqrt(mean((pred_15d .- actual_15d).^2)),
            sqrt(mean((pred_24h .- actual_24h).^2))
        ],
        MAE = [
            mean(abs.(pred_15d .- actual_15d)),
            mean(abs.(pred_24h .- actual_24h))
        ],
        MAPE = [
            mean(abs.((pred_15d .- actual_15d) ./ actual_15d)) * 100,
            mean(abs.((pred_24h .- actual_24h) ./ actual_24h)) * 100
        ]
    )
    
    # Simulate battery decisions
    decisions_15d = simulate_battery_decisions(pred_15d)
    decisions_24h = simulate_battery_decisions(pred_24h)
    
    # Compare decisions
    decision_metrics = DataFrame(
        Model = ["15 days", "24 hours"],
        Total_Charge = [sum(decisions_15d.charge), sum(decisions_24h.charge)],
        Total_Discharge = [sum(decisions_15d.discharge), sum(decisions_24h.discharge)],
        Total_Revenue = [sum(decisions_15d.revenue), sum(decisions_24h.revenue)],
        Avg_SOC = [mean(decisions_15d.soc), mean(decisions_24h.soc)]
    )
    
    # Decision correlation
    charge_corr = cor(decisions_15d.charge, decisions_24h.charge)
    discharge_corr = cor(decisions_15d.discharge, decisions_24h.discharge)
    
    println("\n--- Prediction Metrics ---")
    println(metrics)
    
    println("\n--- Decision Metrics ---")
    println(decision_metrics)
    
    println("\n--- Decision Correlation ---")
    println("Charge correlation: ", round(charge_corr, digits=3))
    println("Discharge correlation: ", round(discharge_corr, digits=3))
    
    return metrics, decision_metrics, decisions_15d, decisions_24h
end

# ========== PART 5: VISUALIZATION ==========

"""
Generates comparative visualizations
"""
function plot_comparative_results(df_test, pred_15d, pred_24h, decisions_15d, decisions_24h)
    # Create figure with multiple subplots
    p1 = plot(df_test.timestamp[1:168], df_test.demand[1:168], 
        label="Actual Demand", lw=2, color=:black,
        title="Forecast Comparison - First Week")
    plot!(p1, df_test.timestamp[1:168], pred_15d[1:168], 
        label="15-day Model", lw=2, color=:blue, alpha=0.7)
    plot!(p1, df_test.timestamp[1:168], pred_24h[1:168], 
        label="24-hour Model", lw=2, color=:red, alpha=0.7)
    xlabel!(p1, "Date")
    ylabel!(p1, "Demand (MW)")
    
    # Battery decisions
    p2 = plot(df_test.timestamp[1:168], decisions_15d.charge[1:168], 
        label="Charge 15d", lw=2, color=:green,
        title="Battery Decisions")
    plot!(p2, df_test.timestamp[1:168], -decisions_15d.discharge[1:168], 
        label="Discharge 15d", lw=2, color=:green, ls=:dash)
    plot!(p2, df_test.timestamp[1:168], decisions_24h.charge[1:168], 
        label="Charge 24h", lw=2, color=:orange)
    plot!(p2, df_test.timestamp[1:168], -decisions_24h.discharge[1:168], 
        label="Discharge 24h", lw=2, color=:orange, ls=:dash)
    xlabel!(p2, "Date")
    ylabel!(p2, "Power (MW)")
    hline!(p2, [0], color=:gray, ls=:dot, label="")
    
    # State of charge
    p3 = plot(df_test.timestamp[1:168], decisions_15d.soc[1:168], 
        label="SOC 15d", lw=2, color=:blue,
        title="Battery State of Charge")
    plot!(p3, df_test.timestamp[1:168], decisions_24h.soc[1:168], 
        label="SOC 24h", lw=2, color=:red)
    xlabel!(p3, "Date")
    ylabel!(p3, "SOC (MWh)")
    
    # Cumulative revenue
    p4 = plot(df_test.timestamp[1:168], cumsum(decisions_15d.revenue[1:168]), 
        label="Revenue 15d", lw=2, color=:blue,
        title="Cumulative Revenue")
    plot!(p4, df_test.timestamp[1:168], cumsum(decisions_24h.revenue[1:168]), 
        label="Revenue 24h", lw=2, color=:red)
    xlabel!(p4, "Date")
    ylabel!(p4, "Cumulative Revenue (\$)")
    
    # Combine plots
    plot(p1, p2, p3, p4, layout=(2,2), size=(1400, 1000))
end

# ========== PART 6: MAIN WORKFLOW ==========

function main()
    println("=== Decision-Oriented Operational Load Forecasting System ===")
    
    # 1. Generate synthetic data
    df = generate_synthetic_operational_data()
    println("Data generated: ", size(df))
    
    # 2. Data split
    train_size = Int(floor(0.7 * nrow(df)))
    val_size = Int(floor(0.15 * nrow(df)))
    
    df_train = df[1:train_size, :]
    df_val = df[train_size+1:train_size+val_size, :]
    df_test = df[train_size+val_size+1:end, :]
    
    # 3. Define features
    features = [
        :hour, :day_of_week, :month, :is_holiday, :is_weekend,
        :temperature, :temp_squared, :temp_cubed,
        :demand  # Current demand as feature
    ]
    
    # 4. Train models
    model_15d, mean_15d, std_15d, losses_15d = train_model_15days(df_train, features)
    model_24h, mean_24h, std_24h, losses_24h = train_model_24hours(df_train, features)
    
    # 5. Evaluate models
    norm_params_15d = (mean=mean_15d, std=std_15d)
    norm_params_24h = (mean=mean_24h, std=std_24h)
    
    pred_metrics, decision_metrics, decisions_15d, decisions_24h = evaluate_models(
        model_15d, model_24h, df_test, features, norm_params_15d, norm_params_24h
    )
    
    # 6. Computational efficiency analysis
    println("\n=== Computational Efficiency Analysis ===")
    
    # Measure inference time
    X_sample = Matrix{Float32}(df_test[1:24, features])  # 24 hours of data
    
    # 15-day model time
    X_norm_15d = (X_sample .- mean_15d) ./ std_15d
    t_15d = @elapsed for i in 1:100
        _ = model_15d(X_norm_15d')
    end
    
    # 24-hour model time
    X_norm_24h = (X_sample .- mean_24h) ./ std_24h
    t_24h = @elapsed for i in 1:100
        _ = model_24h(X_norm_24h')
    end
    
    println("Average inference time:")
    println("15-day model: ", round(t_15d/100*1000, digits=2), " ms")
    println("24-hour model: ", round(t_24h/100*1000, digits=2), " ms")
    println("Speedup: ", round(t_15d/t_24h, digits=2), "x")
    
    # 7. Generate visualizations
    # Prepare predictions for visualization
    X_test = Matrix{Float32}(df_test[!, features])
    X_test_15d_norm = (X_test .- mean_15d) ./ std_15d
    X_test_24h_norm = (X_test .- mean_24h) ./ std_24h
    
    pred_15d_vis = vec(model_15d(X_test_15d_norm'))
    pred_24h_vis = vec(model_24h(X_test_24h_norm'))
    
    plot_results = plot_comparative_results(
        df_test, pred_15d_vis, pred_24h_vis, 
        decisions_15d, decisions_24h
    )
    
    display(plot_results)
    savefig(plot_results, joinpath(raw"C:\Users\jhonm\Visual SC Projects\opf-paper\data", "operational_forecasting_comparison_v3.png"))
    
    # 8. Save results
    data_dir = raw"C:\Users\jhonm\Visual SC Projects\opf-paper\data"
    CSV.write(joinpath(data_dir, "synthetic_operational_data_v3.csv"), df)
    CSV.write(joinpath(data_dir, "prediction_metrics_v3.csv"), pred_metrics)
    CSV.write(joinpath(data_dir, "decision_metrics_v3.csv"), decision_metrics)
    CSV.write(joinpath(data_dir, "battery_decisions_15d_v3.csv"), decisions_15d)
    CSV.write(joinpath(data_dir, "battery_decisions_24h_v3.csv"), decisions_24h)
    
    println("\n=== Analysis Complete ===")
    println("Files saved in: C:\\Users\\jhonm\\Visual SC Projects\\opf-paper\\data")
    println("- synthetic_operational_data_v3.csv")
    println("- prediction_metrics_v3.csv")
    println("- decision_metrics_v3.csv")
    println("- battery_decisions_15d_v3.csv")
    println("- battery_decisions_24h_v3.csv")
    println("- operational_forecasting_comparison_v3.png")
    
    return df, model_15d, model_24h, pred_metrics, decision_metrics
end

# Execute the system
df, model_15d, model_24h, pred_metrics, decision_metrics = main()