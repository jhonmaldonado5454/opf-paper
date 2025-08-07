# COMPREHENSIVE ANALYSIS: NEURAL NETWORK vs LINEAR REGRESSION - FINAL VERSION
# Temporal horizon comparison (24h vs 15d) with CDFs

using CSV, DataFrames
using Flux
using Statistics
using Random
using Plots
using LinearAlgebra
using GLM

Random.seed!(42)
gr()

function complete_advanced_analysis()
    println("🚀 COMPREHENSIVE ANALYSIS: NEURAL NETWORK vs LINEAR REGRESSION")
    println("="^70)
    
    try
        # ========== DATA LOADING ==========
        data_file = "data/synthetic_load_data_v4.csv"
        
        if !isfile(data_file)
            println("❌ ERROR: File not found: $data_file")
            return
        end
        
        println("📂 Loading data from: $data_file")
        df = CSV.read(data_file, DataFrame)
        println("✅ Loaded $(nrow(df)) rows × $(ncol(df)) columns")
        
        # ========== FEATURE DETECTION ==========
        println("\n🔍 DETECTING AVAILABLE FEATURES...")
        
        all_candidates = [
            :hour, :day_of_week, :month, :week_of_year, :day_of_year, :quarter,
            :is_holiday, :is_weekend, :is_business_day,
            :temperature, :temperature_trend, :temperature_seasonal, :temperature_daily, 
            :temperature_noise, :temp_squared, :temp_cubed, :temp_log, 
            :cooling_degree_days, :heating_degree_days, :temperature_volatility,
            :load, :load_base, :load_seasonal, :load_daily, :load_weather, 
            :load_economic, :load_noise, :load_lag_1h, :load_lag_2h, :load_lag_24h, 
            :load_lag_48h, :load_lag_168h, :load_lag_336h, :load_ma_24h, 
            :load_std_24h, :load_ma_168h
        ]
        
        features = Symbol[]
        for col in all_candidates
            if col in names(df) && eltype(df[!, col]) <: Union{Number, Missing}
                push!(features, col)
                println("  ✅ $col (auto-detected)")
            end
        end
        
        # Search for more features if needed
        if length(features) < 10
            println("\n🔍 SEARCHING FOR ADDITIONAL NUMERIC COLUMNS...")
            for col_name in names(df)
                col_sym = Symbol(col_name)
                if col_sym ∉ features && eltype(df[!, col_sym]) <: Union{Number, Missing}
                    if !occursin("24h", string(col_sym)) && !occursin("168h", string(col_sym))
                        push!(features, col_sym)
                        println("  ✅ $col_sym (auto-detected)")
                    end
                end
            end
        end
        
        println("📊 Features found: $(length(features))")
        
        # Show available load columns
        println("\n🔍 'LOAD' RELATED COLUMNS AVAILABLE:")
        load_columns = filter(col -> occursin("load", lowercase(string(col))), names(df))
        for (i, col) in enumerate(load_columns)
            col_type = eltype(df[!, col])
            println("  $i. $col :: $col_type")
        end
        
        # ========== TARGET DETECTION ==========
        println("\n🎯 SEARCHING FOR TARGET VARIABLES...")
        target_24h = nothing
        target_15d = nothing
        
        # Get column names as strings for robust comparison
        column_names = names(df)
        column_strings = string.(column_names)
        
        # Target 24h - search directly in existing columns
        for col_name in column_names
            col_str = string(col_name)
            if col_str == "load_24h"
                target_24h = col_name
                println("  ✅ 24h target found: $target_24h")
                break
            elseif col_str == "load_lag_24h"
                target_24h = col_name
                println("  ✅ 24h target alternative: $target_24h")
                break
            elseif col_str == "load_1h"
                target_24h = col_name
                println("  ✅ 24h target (using 1h): $target_24h")
                break
            end
        end
        
        # Target 15d - search directly in existing columns
        for col_name in column_names
            col_str = string(col_name)
            if col_str == "load_360h"
                target_15d = col_name
                println("  ✅ 15-day target found: $target_15d")
                break
            elseif col_str == "load_168h"
                target_15d = col_name
                println("  ✅ 15-day target (using 168h): $target_15d")
                break
            elseif col_str == "load_lag_168h"
                target_15d = col_name
                println("  ✅ 15-day target alternative: $target_15d")
                break
            end
        end
        
        # Verify we have both targets
        if target_24h === nothing || target_15d === nothing
            println("\n❌ ERROR: Could not determine valid targets")
            println("  target_24h: $target_24h")
            println("  target_15d: $target_15d")
            return
        end
        
        # ========== CYCLICAL FEATURES ==========
        println("\n🔄 CREATING CYCLICAL FEATURES...")
        original_features = copy(features)
        
        if :hour in names(df)
            df.hour_sin = sin.(2π * df.hour / 24)
            df.hour_cos = cos.(2π * df.hour / 24)
            push!(features, :hour_sin, :hour_cos)
            println("  ✅ Hour features: hour_sin, hour_cos")
        end
        
        if :day_of_week in names(df)
            df.day_sin = sin.(2π * df.day_of_week / 7)
            df.day_cos = cos.(2π * df.day_of_week / 7)
            push!(features, :day_sin, :day_cos)
            println("  ✅ Day features: day_sin, day_cos")
        end
        
        if :month in names(df)
            df.month_sin = sin.(2π * df.month / 12)
            df.month_cos = cos.(2π * df.month / 12)
            push!(features, :month_sin, :month_cos)
            println("  ✅ Month features: month_sin, month_cos")
        end
        
        println("📊 Total final features: $(length(features))")
        
        # ========== DATA CLEANING ==========
        println("\n🧹 CLEANING DATA...")
        original_rows = nrow(df)
        
        # Remove missing values
        df = df[.!ismissing.(df[!, target_24h]), :]
        df = df[.!ismissing.(df[!, target_15d]), :]
        df = dropmissing(df, features)
        
        println("  📊 Clean data: $(nrow(df))/$(original_rows) rows")
        
        # ========== DATA SPLITTING ==========
        println("\n🔧 SPLITTING DATA...")
        n_total = nrow(df)
        n_train = Int(floor(0.7 * n_total))
        n_val = Int(floor(0.2 * n_total))
        n_test = n_total - n_train - n_val
        
        train_idx = 1:n_train
        val_idx = (n_train+1):(n_train+n_val)
        test_idx = (n_train+n_val+1):n_total
        
        println("  🏋️  Training: $n_train samples (70%)")
        println("  ✅ Validation: $(length(val_idx)) samples (20%)")
        println("  🧪 Test: $n_test samples (10%)")
        
        # ========== MATRIX PREPARATION ==========
        println("\n📊 CREATING MATRICES...")
        X_train = Matrix{Float64}(df[train_idx, features])
        y_train_24h = Float64.(df[train_idx, target_24h])
        y_train_15d = Float64.(df[train_idx, target_15d])
        
        X_val = Matrix{Float64}(df[val_idx, features])
        y_val_24h = Float64.(df[val_idx, target_24h])
        y_val_15d = Float64.(df[val_idx, target_15d])
        
        X_test = Matrix{Float64}(df[test_idx, features])
        y_test_24h = Float64.(df[test_idx, target_24h])
        y_test_15d = Float64.(df[test_idx, target_15d])
        
        println("  ✅ Matrices created successfully")
        println("  📐 Training dimensions: $(size(X_train))")
        
        # ========== NORMALIZATION ==========
        println("\n📏 NORMALIZING DATA...")
        X_mean = mean(X_train, dims=1)
        X_std = std(X_train, dims=1)
        X_std[X_std .< 1e-8] .= 1.0
        
        X_train_norm = Float32.((X_train .- X_mean) ./ X_std)
        X_val_norm = Float32.((X_val .- X_mean) ./ X_std)
        X_test_norm = Float32.((X_test .- X_mean) ./ X_std)
        
        println("  ✅ Normalization completed")
        
        # ========== NEURAL NETWORK TRAINING 24H ==========
        println("\n🧠 TRAINING NEURAL NETWORK - 24H...")
        n_features = size(X_train, 2)
        
        model_24h = Chain(
            Dense(n_features, 128, relu),
            Dense(128, 64, relu),
            Dense(64, 32, relu),
            Dense(32, 1)
        )
        
        opt = ADAM(0.001)
        opt_state_24h = Flux.setup(opt, model_24h)
        best_loss_24h = Inf
        best_model_24h = deepcopy(model_24h)
        
        train_input = permutedims(X_train_norm)
        val_input = permutedims(X_val_norm)
        y_train_24h_f32 = Float32.(y_train_24h)
        y_val_24h_f32 = Float32.(y_val_24h)
        
        train_losses_24h = Float64[]
        val_losses_24h = Float64[]
        
        println("  🚀 Starting 24h training...")
        for epoch in 1:100
            loss_fn(m) = mean((vec(m(train_input)) .- y_train_24h_f32).^2)
            
            grads = gradient(loss_fn, model_24h)
            Flux.update!(opt_state_24h, model_24h, grads[1])
            
            train_loss = loss_fn(model_24h)
            val_pred = vec(model_24h(val_input))
            val_loss = mean((val_pred .- y_val_24h_f32).^2)
            
            push!(train_losses_24h, train_loss)
            push!(val_losses_24h, val_loss)
            
            if val_loss < best_loss_24h
                best_loss_24h = val_loss
                best_model_24h = deepcopy(model_24h)
            end
            
            if epoch % 25 == 0
                println("    📈 Epoch $epoch - RMSE: $(round(sqrt(val_loss), digits=4))")
            end
        end
        
        println("  ✅ NN 24h trained - RMSE: $(round(sqrt(best_loss_24h), digits=4))")
        
        # ========== NEURAL NETWORK TRAINING 15D ==========
        println("\n🧠 TRAINING NEURAL NETWORK - 15D...")
        
        model_15d = Chain(
            Dense(n_features, 128, relu),
            Dense(128, 64, relu),
            Dense(64, 32, relu),
            Dense(32, 1)
        )
        
        opt_state_15d = Flux.setup(ADAM(0.001), model_15d)
        best_loss_15d = Inf
        best_model_15d = deepcopy(model_15d)
        
        y_train_15d_f32 = Float32.(y_train_15d)
        y_val_15d_f32 = Float32.(y_val_15d)
        
        train_losses_15d = Float64[]
        val_losses_15d = Float64[]
        
        println("  🚀 Starting 15d training...")
        for epoch in 1:100
            loss_fn(m) = mean((vec(m(train_input)) .- y_train_15d_f32).^2)
            
            grads = gradient(loss_fn, model_15d)
            Flux.update!(opt_state_15d, model_15d, grads[1])
            
            train_loss = loss_fn(model_15d)
            val_pred = vec(model_15d(val_input))
            val_loss = mean((val_pred .- y_val_15d_f32).^2)
            
            push!(train_losses_15d, train_loss)
            push!(val_losses_15d, val_loss)
            
            if val_loss < best_loss_15d
                best_loss_15d = val_loss
                best_model_15d = deepcopy(model_15d)
            end
            
            if epoch % 25 == 0
                println("    📈 Epoch $epoch - RMSE: $(round(sqrt(val_loss), digits=4))")
            end
        end
        
        println("  ✅ NN 15d trained - RMSE: $(round(sqrt(best_loss_15d), digits=4))")
        
        # ========== LINEAR REGRESSION TRAINING ==========
        println("\n📈 TRAINING LINEAR REGRESSIONS...")
        
        # 24h
        df_train = DataFrame(X_train, features)
        df_train.target_24h = y_train_24h
        formula_24h = Meta.parse("@formula(target_24h ~ " * join(string.(features), " + ") * ")")
        linear_model_24h = lm(eval(formula_24h), df_train)
        
        # 15d
        df_train.target_15d = y_train_15d
        formula_15d = Meta.parse("@formula(target_15d ~ " * join(string.(features), " + ") * ")")
        linear_model_15d = lm(eval(formula_15d), df_train)
        
        println("  ✅ Both regressions trained")
        
        # ========== TEST EVALUATION ==========
        println("\n🧪 EVALUATING ALL MODELS...")
        
        test_input = permutedims(X_test_norm)
        df_test = DataFrame(X_test, features)
        
        # Predictions
        nn_pred_24h = vec(best_model_24h(test_input))
        nn_pred_15d = vec(best_model_15d(test_input))
        linear_pred_24h = predict(linear_model_24h, df_test)
        linear_pred_15d = predict(linear_model_15d, df_test)
        
        # Metrics
        function compute_metrics(y_true, y_pred)
            rmse = sqrt(mean((y_true .- y_pred).^2))
            mae = mean(abs.(y_true .- y_pred))
            r2 = 1 - sum((y_true .- y_pred).^2) / sum((y_true .- mean(y_true)).^2)
            return rmse, mae, r2
        end
        
        nn_rmse_24h, nn_mae_24h, nn_r2_24h = compute_metrics(y_test_24h, nn_pred_24h)
        nn_rmse_15d, nn_mae_15d, nn_r2_15d = compute_metrics(y_test_15d, nn_pred_15d)
        linear_rmse_24h, linear_mae_24h, linear_r2_24h = compute_metrics(y_test_24h, linear_pred_24h)
        linear_rmse_15d, linear_mae_15d, linear_r2_15d = compute_metrics(y_test_15d, linear_pred_15d)
        
        # ========== DISPLAY RESULTS ==========
        println("\n" * "="^70)
        println("📋 COMPREHENSIVE COMPARATIVE RESULTS")
        println("="^70)
        
        println("\n🧠 NEURAL NETWORK - 24 HOURS:")
        println("  🎯 RMSE: $(round(nn_rmse_24h, digits=4))")
        println("  📊 MAE:  $(round(nn_mae_24h, digits=4))")
        println("  📈 R²:   $(round(nn_r2_24h, digits=4))")
        
        println("\n🧠 NEURAL NETWORK - 15 DAYS:")
        println("  🎯 RMSE: $(round(nn_rmse_15d, digits=4))")
        println("  📊 MAE:  $(round(nn_mae_15d, digits=4))")
        println("  📈 R²:   $(round(nn_r2_15d, digits=4))")
        
        println("\n📈 LINEAR REGRESSION - 24 HOURS:")
        println("  🎯 RMSE: $(round(linear_rmse_24h, digits=4))")
        println("  📊 MAE:  $(round(linear_mae_24h, digits=4))")
        println("  📈 R²:   $(round(linear_r2_24h, digits=4))")
        
        println("\n📈 LINEAR REGRESSION - 15 DAYS:")
        println("  🎯 RMSE: $(round(linear_rmse_15d, digits=4))")
        println("  📊 MAE:  $(round(linear_mae_15d, digits=4))")
        println("  📈 R²:   $(round(linear_r2_15d, digits=4))")
        
        # ========== CREATE VISUALIZATIONS ==========
        println("\n📊 CREATING ADVANCED VISUALIZATIONS...")
        
        results_path = "C:\\Users\\jhonm\\Visual SC Projects\\opf-paper\\results"
        if !isdir(results_path)
            mkpath(results_path)
            println("  📁 Directory created: $results_path")
        end
        
        # CDF function
        function compute_cdf(residuals)
            sorted_residuals = sort(residuals)
            n = length(sorted_residuals)
            cdf_values = (1:n) / n
            return sorted_residuals, cdf_values
        end
        
        # Calculate residuals
        nn_residuals_24h = y_test_24h .- nn_pred_24h
        nn_residuals_15d = y_test_15d .- nn_pred_15d
        lr_residuals_24h = y_test_24h .- linear_pred_24h
        lr_residuals_15d = y_test_15d .- linear_pred_15d
        
        # ========== PLOT 1: CDFs ==========
        nn_24h_x, nn_24h_cdf = compute_cdf(abs.(nn_residuals_24h))
        nn_15d_x, nn_15d_cdf = compute_cdf(abs.(nn_residuals_15d))
        lr_24h_x, lr_24h_cdf = compute_cdf(abs.(lr_residuals_24h))
        lr_15d_x, lr_15d_cdf = compute_cdf(abs.(lr_residuals_15d))
        
        p_cdf = plot(title="Cumulative Distribution Functions - Absolute Errors",
                    xlabel="Absolute Error", ylabel="Cumulative Probability", 
                    grid=true, legend=:bottomright, size=(900, 600))
        
        plot!(p_cdf, nn_24h_x, nn_24h_cdf, linewidth=3, color=:blue, 
              label="Neural Network 24h", linestyle=:solid)
        plot!(p_cdf, nn_15d_x, nn_15d_cdf, linewidth=3, color=:lightblue, 
              label="Neural Network 15d", linestyle=:dash)
        plot!(p_cdf, lr_24h_x, lr_24h_cdf, linewidth=3, color=:green, 
              label="Linear Regression 24h", linestyle=:solid)
        plot!(p_cdf, lr_15d_x, lr_15d_cdf, linewidth=3, color=:lightgreen, 
              label="Linear Regression 15d", linestyle=:dash)
        
        savefig(p_cdf, joinpath(results_path, "cdf_all_models_v4.png"))
        println("  ✅ CDF plot created")
        
        # ========== PLOT 2: SCATTER COMPARISON ==========
        min_val = minimum([minimum(y_test_24h), minimum(y_test_15d)])
        max_val = maximum([maximum(y_test_24h), maximum(y_test_15d)])
        
        p_scatter = plot(layout=(2, 2), size=(1200, 1000))
        
        # NN 24h
        scatter!(p_scatter[1], y_test_24h, nn_pred_24h, alpha=0.7, color=:blue, markersize=3,
                title="Neural Network - 24h", xlabel="Actual", ylabel="Predicted", grid=true,
                label="R² = $(round(nn_r2_24h, digits=3))")
        plot!(p_scatter[1], [min_val, max_val], [min_val, max_val], color=:red, linewidth=2, 
              linestyle=:dash, label="Perfect")
        
        # NN 15d
        scatter!(p_scatter[2], y_test_15d, nn_pred_15d, alpha=0.7, color=:lightblue, markersize=3,
                title="Neural Network - 15d", xlabel="Actual", ylabel="Predicted", grid=true,
                label="R² = $(round(nn_r2_15d, digits=3))")
        plot!(p_scatter[2], [min_val, max_val], [min_val, max_val], color=:red, linewidth=2, 
              linestyle=:dash, label="Perfect")
        
        # LR 24h
        scatter!(p_scatter[3], y_test_24h, linear_pred_24h, alpha=0.7, color=:green, markersize=3,
                title="Linear Regression - 24h", xlabel="Actual", ylabel="Predicted", grid=true,
                label="R² = $(round(linear_r2_24h, digits=3))")
        plot!(p_scatter[3], [min_val, max_val], [min_val, max_val], color=:red, linewidth=2, 
              linestyle=:dash, label="Perfect")
        
        # LR 15d
        scatter!(p_scatter[4], y_test_15d, linear_pred_15d, alpha=0.7, color=:lightgreen, markersize=3,
                title="Linear Regression - 15d", xlabel="Actual", ylabel="Predicted", grid=true,
                label="R² = $(round(linear_r2_15d, digits=3))")
        plot!(p_scatter[4], [min_val, max_val], [min_val, max_val], color=:red, linewidth=2, 
              linestyle=:dash, label="Perfect")
        
        savefig(p_scatter, joinpath(results_path, "scatter_all_models_v4.png"))
        println("  ✅ Scatter plots created")
        
        # ========== PLOT 3: COMPARATIVE METRICS ==========
        rmse_data = [nn_rmse_24h, nn_rmse_15d, linear_rmse_24h, linear_rmse_15d]
        model_names = ["NN 24h", "NN 15d", "LR 24h", "LR 15d"]
        colors = [:blue, :lightblue, :green, :lightgreen]
        
        p_metrics = bar(1:4, rmse_data, color=colors, alpha=0.7,
                       title="Comparative RMSE by Model and Horizon",
                       xlabel="Models", ylabel="RMSE",
                       xticks=(1:4, model_names), grid=true, legend=false)
        
        savefig(p_metrics, joinpath(results_path, "metrics_comparison_v4.png"))
        println("  ✅ Comparative metrics created")
        
        # ========== PLOT 4: TRAINING AND PERCENTILES ==========
        p_training = plot(layout=(1, 2), size=(1200, 500))
        
        plot!(p_training[1], 1:length(train_losses_24h), sqrt.(train_losses_24h), 
              label="Train 24h", linewidth=2, color=:blue, title="NN Training")
        plot!(p_training[1], 1:length(val_losses_24h), sqrt.(val_losses_24h), 
              label="Val 24h", linewidth=2, color=:blue, linestyle=:dash)
        plot!(p_training[1], 1:length(train_losses_15d), sqrt.(train_losses_15d), 
              label="Train 15d", linewidth=2, color=:lightblue)
        plot!(p_training[1], 1:length(val_losses_15d), sqrt.(val_losses_15d), 
              label="Val 15d", linewidth=2, color=:lightblue, linestyle=:dash)
        
        # Percentiles
        percentiles = [50, 75, 90, 95, 99]
        nn_24h_perc = [quantile(abs.(nn_residuals_24h), p/100) for p in percentiles]
        lr_24h_perc = [quantile(abs.(lr_residuals_24h), p/100) for p in percentiles]
        
        plot!(p_training[2], percentiles, nn_24h_perc, linewidth=3, marker=:circle,
              label="NN 24h", color=:blue, title="Error Percentiles")
        plot!(p_training[2], percentiles, lr_24h_perc, linewidth=3, marker=:square,
              label="LR 24h", color=:green)
        
        savefig(p_training, joinpath(results_path, "training_and_percentiles_v4.png"))
        println("  ✅ Training and percentiles created")
        
        # ========== COMBINED PANEL ==========
        combined = plot(p_cdf, p_scatter, layout=(2, 1), size=(1200, 1400),
                       plot_title="Comprehensive Analysis: CDFs and Horizon Comparisons")
        
        savefig(combined, joinpath(results_path, "combined_advanced_analysis_v4.png"))
        println("  ✅ Combined panel created")
        
        # ========== FINAL SUMMARY ==========
        println("\n📋 FILES CREATED:")
        created_files = [
            "cdf_all_models_v4.png",
            "scatter_all_models_v4.png",
            "metrics_comparison_v4.png",
            "training_and_percentiles_v4.png",
            "combined_advanced_analysis_v4.png"
        ]
        
        for (i, file) in enumerate(created_files)
            println("  $i. $file")
        end
        
        # Risk analysis
        println("\n📊 RISK ANALYSIS (percentiles):")
        p95_nn_24h = quantile(abs.(nn_residuals_24h), 0.95)
        p95_lr_24h = quantile(abs.(lr_residuals_24h), 0.95)
        
        println("  🎯 NN 24h - P95: $(round(p95_nn_24h, digits=2))")
        println("  🎯 LR 24h - P95: $(round(p95_lr_24h, digits=2))")
        
        display(combined)
        
        println("\n" * "="^70)
        println("✅ ADVANCED ANALYSIS COMPLETED SUCCESSFULLY")
        println("📁 Results saved to: $results_path")
        println("="^70)
        
    catch e
        println("\n❌ ERROR:")
        println("$e")
        showerror(stdout, e, catch_backtrace())
    end
end

println("🚀 Running advanced analysis...")
complete_advanced_analysis()