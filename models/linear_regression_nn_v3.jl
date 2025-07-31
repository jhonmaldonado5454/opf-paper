# Load Forecasting Models Comparison
# Compares 15-day vs 24-hour forecasting models
# Based on Prof. Gonzalo's requirements

using CSV, DataFrames
using Flux
using Statistics
using Random
using Printf
using Plots
using Dates
using LinearAlgebra

# Set random seed for reproducibility
Random.seed!(42)

"""
Loads the synthetic data generated previously
"""
function load_synthetic_data(data_dir = raw"C:\Users\jhonm\Visual SC Projects\opf-paper\data")
    data_file = joinpath(data_dir, "synthetic_load_data_v3.csv")
    
    if !isfile(data_file)
        error("Synthetic data file not found at: $data_file")
        println("Please run the synthetic data generator first.")
        return nothing
    end
    
    println("Loading synthetic data from: $data_file")
    df = CSV.read(data_file, DataFrame)
    
    # Convert timestamp column back to DateTime
    df.timestamp = DateTime.(df.timestamp)
    
    println("Data loaded successfully: $(nrow(df)) rows × $(ncol(df)) columns")
    return df
end

"""
Prepares data for training by splitting and selecting features
"""
function prepare_data(df; train_ratio=0.7, val_ratio=0.15)
    println("\n=== Data Preparation ===")
    
    n_total = nrow(df)
    n_train = Int(floor(train_ratio * n_total))
    n_val = Int(floor(val_ratio * n_total))
    
    # Split data chronologically
    df_train = df[1:n_train, :]
    df_val = df[n_train+1:n_train+n_val, :]
    df_test = df[n_train+n_val+1:end, :]
    
    println("Training set: $(nrow(df_train)) samples")
    println("Validation set: $(nrow(df_val)) samples")
    println("Test set: $(nrow(df_test)) samples")
    
    # Define feature sets
    temporal_features = [:hour, :day_of_week, :month, :day_of_year, :week_of_year]
    calendar_features = [:is_holiday, :is_weekend]
    weather_features = [:temperature, :temp_squared, :temp_cubed, 
                       :cooling_degree_days, :heating_degree_days]
    lag_features = [:load_lag_1h, :load_lag_24h, :load_lag_168h]
    
    # Combine all features
    all_features = vcat(temporal_features, calendar_features, weather_features, lag_features)
    
    println("Selected features: $(length(all_features))")
    for feat in all_features
        println("  - $feat")
    end
    
    return df_train, df_val, df_test, all_features
end

"""
Creates and trains a neural network model
"""
function create_and_train_model(X_train, y_train, X_val, y_val; 
                               model_type="24h", 
                               epochs=200, 
                               learning_rate=0.01,
                               batch_size=256)
    
    println("\n=== Training $(model_type) Model ===")
    
    # Normalize features
    mean_X = mean(X_train, dims=1)
    std_X = std(X_train, dims=1)
    std_X[std_X .== 0] .= 1  # Avoid division by zero
    
    X_train_norm = (X_train .- mean_X) ./ std_X
    X_val_norm = (X_val .- mean_X) ./ std_X
    
    # Model architecture based on complexity needed
    if model_type == "15d" || model_type == "360h"
        # More complex model for longer horizon
        model = Chain(
            Dense(size(X_train, 2), 128, relu),
            Dropout(0.3),
            Dense(128, 64, relu),
            Dropout(0.2),
            Dense(64, 32, relu),
            Dense(32, 1)
        )
    else
        # Simpler model for shorter horizon
        model = Chain(
            Dense(size(X_train, 2), 64, relu),
            Dropout(0.2),
            Dense(64, 32, relu),
            Dense(32, 1)
        )
    end
    
    # Training setup
    opt = ADAM(learning_rate)
    state = Flux.setup(opt, model)
    
    # Training history
    train_losses = Float32[]
    val_losses = Float32[]
    
    println("Starting training...")
    println("Epochs: $epochs, Learning rate: $learning_rate, Batch size: $batch_size")
    
    # Training loop
    for epoch in 1:epochs
        # Batch training
        n_batches = div(size(X_train_norm, 1), batch_size)
        epoch_loss = 0.0
        
        for batch in 1:n_batches
            start_idx = (batch - 1) * batch_size + 1
            end_idx = min(batch * batch_size, size(X_train_norm, 1))
            
            X_batch = X_train_norm[start_idx:end_idx, :]
            y_batch = y_train[start_idx:end_idx]
            
            # Forward pass and gradient computation
            grads = gradient(m -> mean((m(X_batch') .- y_batch').^2), model)
            Flux.update!(state, model, grads[1])
            
            # Accumulate loss
            batch_loss = mean((model(X_batch') .- y_batch').^2)
            epoch_loss += batch_loss
        end
        
        # Average epoch loss
        epoch_loss /= n_batches
        push!(train_losses, epoch_loss)
        
        # Validation loss
        val_loss = mean((model(X_val_norm') .- y_val').^2)
        push!(val_losses, val_loss)
        
        # Print progress
        if epoch % 20 == 0 || epoch == 1
            println("Epoch $epoch - Train Loss: $(round(epoch_loss, digits=4)), Val Loss: $(round(val_loss, digits=4))")
        end
    end
    
    println("Training completed!")
    final_val_loss = val_losses[end]
    println("Final validation loss: $(round(final_val_loss, digits=4))")
    
    return model, mean_X, std_X, train_losses, val_losses
end

"""
Evaluates model performance with comprehensive metrics
"""
function evaluate_model(model, X_test, y_test, mean_X, std_X, model_name="Model")
    println("\n=== Evaluating $model_name ===")
    
    # Normalize test data
    X_test_norm = (X_test .- mean_X) ./ std_X
    
    # Make predictions
    predictions = vec(model(X_test_norm'))
    
    # Calculate metrics
    mse = mean((predictions .- y_test).^2)
    rmse = sqrt(mse)
    mae = mean(abs.(predictions .- y_test))
    mape = mean(abs.((predictions .- y_test) ./ y_test)) * 100
    
    # Additional metrics
    r2 = 1 - sum((y_test .- predictions).^2) / sum((y_test .- mean(y_test)).^2)
    max_error = maximum(abs.(predictions .- y_test))
    
    # Peak hour performance (hours 7-9 and 17-20)
    peak_mask = [h in [7, 8, 9, 17, 18, 19, 20] for h in repeat(0:23, outer=div(length(y_test), 24) + 1)[1:length(y_test)]]
    if sum(peak_mask) > 0
        peak_mae = mean(abs.(predictions[peak_mask] .- y_test[peak_mask]))
    else
        peak_mae = NaN
    end
    
    # Create results DataFrame
    results = DataFrame(
        Metric = ["RMSE", "MAE", "MAPE", "R²", "Max Error", "Peak MAE"],
        Value = [rmse, mae, mape, r2, max_error, peak_mae],
        Unit = ["MW", "MW", "%", "-", "MW", "MW"]
    )
    
    println("Performance Metrics:")
    println(results)
    
    return predictions, results
end

"""
Compares two models and generates comprehensive analysis
"""
function compare_models(df_test, pred_24h, pred_15d, features)
    println("\n=== Model Comparison Analysis ===")
    
    # Use appropriate targets for each model
    actual_24h = df_test.load_24h[1:length(pred_24h)]  # Ensure same length
    actual_15d = df_test.load_24h[1:length(pred_15d)]  # Using 24h target for fair comparison
    
    # Basic comparison metrics
    rmse_24h = sqrt(mean((pred_24h .- actual_24h).^2))
    rmse_15d = sqrt(mean((pred_15d .- actual_15d).^2))
    mae_24h = mean(abs.(pred_24h .- actual_24h))
    mae_15d = mean(abs.(pred_15d .- actual_15d))
    
    # Model comparison summary
    comparison = DataFrame(
        Model = ["24-hour", "15-day"],
        RMSE = [rmse_24h, rmse_15d],
        MAE = [mae_24h, mae_15d],
        MAPE = [mean(abs.((pred_24h .- actual_24h) ./ actual_24h)) * 100,
                mean(abs.((pred_15d .- actual_15d) ./ actual_15d)) * 100]
    )
    
    println("Model Comparison:")
    println(comparison)
    
    # Performance difference
    rmse_improvement = (rmse_15d - rmse_24h) / rmse_15d * 100
    mae_improvement = (mae_15d - mae_24h) / mae_15d * 100
    
    println("\nPerformance Analysis:")
    println("RMSE improvement (24h vs 15d): $(round(rmse_improvement, digits=2))%")
    println("MAE improvement (24h vs 15d): $(round(mae_improvement, digits=2))%")
    
    if rmse_24h < rmse_15d
        println("✅ 24-hour model performs better")
    else
        println("❌ 15-day model performs better")
    end
    
    return comparison
end

"""
Creates comprehensive visualizations for model analysis
"""
function create_model_visualizations(df_test, pred_24h, pred_15d, 
                                   results_dir = raw"C:\Users\jhonm\Visual SC Projects\opf-paper\results")
    println("\n=== Creating Visualizations ===")
    
    # Ensure all arrays have the same length
    min_length = min(length(df_test.load_24h), length(pred_24h), length(pred_15d))
    actual = df_test.load_24h[1:min_length]
    pred_24h_viz = pred_24h[1:min_length]
    pred_15d_viz = pred_15d[1:min_length]
    timestamps = df_test.timestamp[1:min_length]
    
    n_display = min(168, min_length)  # First week
    
    # 1. Time series comparison
    p1 = plot(timestamps[1:n_display], actual[1:n_display],
        lw=3, color=:black, label="Actual Load",
        title="Load Forecast Comparison - First Week")
    plot!(p1, timestamps[1:n_display], pred_24h_viz[1:n_display],
        lw=2, color=:red, label="24-hour Model", alpha=0.8)
    plot!(p1, timestamps[1:n_display], pred_15d_viz[1:n_display],
        lw=2, color=:blue, label="15-day Model", alpha=0.8, ls=:dash)
    xlabel!(p1, "Date")
    ylabel!(p1, "Load (MW)")
    
    # 2. Error analysis
    error_24h = abs.(pred_24h_viz .- actual)
    error_15d = abs.(pred_15d_viz .- actual)
    
    p2 = plot(timestamps[1:n_display], error_24h[1:n_display],
        lw=2, color=:red, label="24-hour Model", alpha=0.7,
        title="Absolute Prediction Error")
    plot!(p2, timestamps[1:n_display], error_15d[1:n_display],
        lw=2, color=:blue, label="15-day Model", alpha=0.7)
    xlabel!(p2, "Date")
    ylabel!(p2, "Absolute Error (MW)")
    
    # 3. Scatter plot comparison
    sample_size = min(1000, min_length)
    p3 = scatter(actual[1:sample_size], pred_24h_viz[1:sample_size],
        alpha=0.6, color=:red, label="24-hour Model",
        title="Prediction Accuracy")
    scatter!(p3, actual[1:sample_size], pred_15d_viz[1:sample_size],
        alpha=0.6, color=:blue, label="15-day Model")
    plot!(p3, [minimum(actual), maximum(actual)], [minimum(actual), maximum(actual)],
        color=:black, lw=2, ls=:dash, label="Perfect Prediction")
    xlabel!(p3, "Actual Load (MW)")
    ylabel!(p3, "Predicted Load (MW)")
    
    # 4. Error distribution
    p4 = histogram(pred_24h_viz .- actual, bins=50, alpha=0.6, color=:red,
        normalize=:probability, label="24-hour Model",
        title="Prediction Error Distribution")
    histogram!(p4, pred_15d_viz .- actual, bins=50, alpha=0.6, color=:blue,
        normalize=:probability, label="15-day Model")
    vline!(p4, [0], color=:black, lw=2, ls=:dash, label="Perfect Prediction")
    xlabel!(p4, "Prediction Error (MW)")
    ylabel!(p4, "Probability Density")
    
    # Combine plots
    combined_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1600, 1200),
        plot_title="Load Forecasting Model Comparison",
        plot_titlefont=font(16, "bold"))
    
    # Save visualization
    output_file = joinpath(results_dir, "model_comparison_analysis_v3.png")
    savefig(combined_plot, output_file)
    println("Visualization saved to: $output_file")
    
    display(combined_plot)
    
    return combined_plot
end

"""
Main function to run the complete forecasting analysis
"""
function main()
    println("=== Load Forecasting Model Comparison ===")
    
    # 1. Load data
    df = load_synthetic_data()
    if isnothing(df)
        return nothing
    end
    
    # 2. Prepare data
    df_train, df_val, df_test, features = prepare_data(df)
    
    # 3. Prepare training matrices
    X_train = Matrix{Float32}(df_train[!, features])
    X_val = Matrix{Float32}(df_val[!, features])
    X_test = Matrix{Float32}(df_test[!, features])
    
    # 4. Train 24-hour model
    y_train_24h = Float32.(df_train.load_24h)
    y_val_24h = Float32.(df_val.load_24h)
    y_test_24h = Float32.(df_test.load_24h)
    
    model_24h, mean_24h, std_24h, losses_24h_train, losses_24h_val = create_and_train_model(
        X_train, y_train_24h, X_val, y_val_24h,
        model_type="24h", epochs=200, learning_rate=0.01
    )
    
    # 5. Train 15-day model  
    y_train_15d = Float32.(df_train.load_360h)  # 15 days = ~360 hours
    y_val_15d = Float32.(df_val.load_360h)
    y_test_15d = Float32.(df_test.load_360h)
    
    model_15d, mean_15d, std_15d, losses_15d_train, losses_15d_val = create_and_train_model(
        X_train, y_train_15d, X_val, y_val_15d,
        model_type="15d", epochs=300, learning_rate=0.001  # More epochs, lower LR for complex model
    )
    
    # 6. Evaluate models
    pred_24h, results_24h = evaluate_model(model_24h, X_test, y_test_24h, mean_24h, std_24h, "24-hour Model")
    pred_15d, results_15d = evaluate_model(model_15d, X_test, y_test_15d, mean_15d, std_15d, "15-day Model")
    
    # 7. Compare models (using 24h target for fair comparison)
    # For comparison, predict 24h ahead with both models
    X_test_15d_norm = (X_test .- mean_15d) ./ std_15d
    pred_15d_24h = vec(model_15d(X_test_15d_norm'))  # Note the transpose here
    comparison_results = compare_models(df_test, pred_24h, pred_15d_24h, features)
    
    # 8. Create visualizations
    viz_plot = create_model_visualizations(df_test, pred_24h, pred_15d_24h)
    
    # 9. Save results
    data_dir = raw"C:\Users\jhonm\Visual SC Projects\opf-paper\data"
    CSV.write(joinpath(data_dir, "model_comparison_results_v3.csv"), comparison_results)
    CSV.write(joinpath(data_dir, "results_24h_model_v3.csv"), results_24h)
    CSV.write(joinpath(data_dir, "results_15d_model_v3.csv"), results_15d)
    
    # 10. Performance summary
    println("\n=== Final Summary ===")
    println("Analysis complete! Files saved:")
    println("- model_comparison_results_v3.csv")
    println("- results_24h_model_v3.csv") 
    println("- results_15d_model_v3.csv")
    println("- model_comparison_analysis_v3.png")
    
    return Dict(
        "models" => (model_24h, model_15d),
        "predictions" => (pred_24h, pred_15d_24h),
        "results" => (results_24h, results_15d),
        "comparison" => comparison_results,
        "data" => (df_train, df_val, df_test)
    )
end

# Execute the analysis
results = main()