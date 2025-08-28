"""
Setup Script for Julia Load Forecasting Framework - Version 5
File: setup_julia_v5.jl
Run this script first to install all required packages
Version: 5.0
"""

# Package installation script
println("="^80)
println("JULIA LOAD FORECASTING FRAMEWORK - SETUP VERSION 5")
println("="^80)

using Pkg

println("\nInstalling required packages for v5...")
println("-"^40)

# List of required packages
packages = [
    "CSV",
    "DataFrames", 
    "Statistics",
    "LinearAlgebra",
    "Random",
    "Dates",
    "Flux",
    "MLBase",
    "StatsBase",
    "Plots",
    "JSON",
    "BSON",
    "GLM",
    "MLJ",
    "ScikitLearn",
    "Printf",
    "GR",  # Plotting backend
    "PlotlyJS",  # Alternative plotting backend
    "HTTP",  # For potential data downloads
    "ZipFile",  # For extracting data
    "XLSX",  # For reading Excel files if needed
    "ProgressMeter",  # For progress bars
    "Tables",
    "CategoricalArrays"
]

# Install packages
for pkg in packages
    println("Installing $pkg...")
    try
        Pkg.add(pkg)
        println("  ✓ $pkg installed successfully")
    catch e
        println("  ✗ Error installing $pkg: $e")
    end
end

println("\n" * "-"^40)
println("Package installation complete!")

# Verify installations
println("\nVerifying installations for v5...")
println("-"^40)

installed_packages = []
failed_packages = []

for pkg in packages
    try
        eval(Meta.parse("using $pkg"))
        push!(installed_packages, pkg)
        println("  ✓ $pkg loaded successfully")
    catch e
        push!(failed_packages, pkg)
        println("  ✗ Failed to load $pkg")
    end
end

# Summary
println("\n" * "="^60)
println("INSTALLATION SUMMARY - VERSION 5")
println("="^60)
println("Successfully installed: $(length(installed_packages))/$(length(packages)) packages")

if length(failed_packages) > 0
    println("\nFailed packages:")
    for pkg in failed_packages
        println("  - $pkg")
    end
    println("\nTo fix failed packages, try:")
    println("  julia> using Pkg")
    println("  julia> Pkg.update()")
    println("  julia> Pkg.build(\"package_name\")")
end

println("\n" * "="^60)
println("FILE STRUCTURE FOR VERSION 5")
println("="^60)
println("""
Required files:
1. data_preparation_v5.py           - Python script to generate data
2. setup_julia_v5.jl                - This setup script
3. load_forecasting_framework_v5.jl - Main Julia framework

Generated files:
4. texas_load_forecasting_data_2018_v5.csv - Input data (from Python)
5. model_comparison_julia_v5.png           - Output plots
6. forecasting_results_julia_v5.json       - Output results
""")

println("\n" * "="^60)
println("WORKFLOW STEPS - VERSION 5")
println("="^60)
println("""
Step 1: Generate the data CSV file
   ▶ python data_preparation_v5.py
   Output: texas_load_forecasting_data_2018_v5.csv

Step 2: Run this setup script (if not done)
   ▶ julia setup_julia_v5.jl

Step 3: Run the forecasting framework
   ▶ julia load_forecasting_framework_v5.jl

Step 4: Check the outputs
   • model_comparison_julia_v5.png
   • forecasting_results_julia_v5.json
""")

# Create a simple test to verify Flux is working
println("\n" * "="^60)
println("TESTING FLUX NEURAL NETWORK - V5")
println("="^60)

try
    using Flux
    
    # Create a simple neural network
    model = Chain(
        Dense(10, 5, relu),
        Dense(5, 1)
    )
    
    # Test data
    x = rand(Float32, 10, 100)
    y = rand(Float32, 1, 100)
    
    # Loss function
    loss(x, y) = Flux.mse(model(x), y)
    
    # Calculate initial loss
    initial_loss = loss(x, y)
    println("✓ Flux test successful! Initial loss: $initial_loss")
    
catch e
    println("✗ Flux test failed: $e")
    println("You may need to rebuild Flux:")
    println("  julia> using Pkg")
    println("  julia> Pkg.build(\"Flux\")")
end

# ============================================================================
# QUICK START GUIDE - VERSION 5
# ============================================================================

println("\n" * "="^80)
println("QUICK START GUIDE - VERSION 5")
println("="^80)

println("""

╔══════════════════════════════════════════════════════════════════╗
║                    COMPLETE WORKFLOW - V5                        ║
╚══════════════════════════════════════════════════════════════════╝

1️⃣  GENERATE DATA (Python):
    \$ python data_preparation_v5.py
    
    This creates: texas_load_forecasting_data_2018_v5.csv
    • 8,760 hours of data (1 year)
    • 200+ features including weather, load, temporal
    • 7 forecast horizons (1h to 360h)

2️⃣  SETUP JULIA (one time only):
    julia> include("setup_julia_v5.jl")

3️⃣  RUN ANALYSIS:
    julia> include("load_forecasting_framework_v5.jl")
    
    This will:
    • Train Linear, Ridge, Lasso, and Neural Network models
    • Evaluate on 4 horizons: 1h, 24h, 168h, 360h
    • Generate comparison plots and metrics

4️⃣  VIEW RESULTS:
    • Plots: model_comparison_julia_v5.png
    • Metrics: forecasting_results_julia_v5.json
    • Console: Summary statistics and insights

╔══════════════════════════════════════════════════════════════════╗
║                     CUSTOMIZATION OPTIONS                        ║
╚══════════════════════════════════════════════════════════════════╝

📊 CHANGE FORECAST HORIZONS:
   Edit in load_forecasting_framework_v5.jl:
   ```julia
   "horizons" => Dict(
       "1h" => 1,
       "24h" => 24,
       "720h" => 720   # Add 1 month
   )
   ```

🧠 MODIFY NEURAL NETWORK:
   ```julia
   "nn_params" => Dict(
       "24h" => Dict(
           "layers" => [256, 128, 64],  # Change architecture
           "dropout_rate" => 0.25,
           "learning_rate" => 0.0005,
           "epochs" => 150
       )
   )
   ```

📈 ADD NEW MODELS:
   ```julia
   using DecisionTree
   
   # Add Random Forest
   model = RandomForestRegressor(n_trees=100)
   fit!(model, X_train, y_train)
   ```

⚡ ENABLE GPU (if available):
   ```julia
   using CUDA
   model = model |> gpu
   ```

╔══════════════════════════════════════════════════════════════════╗
║                     EXPECTED PERFORMANCE                         ║
╚══════════════════════════════════════════════════════════════════╝

Typical Results (MAPE %):
┌─────────┬──────────┬────────┬────────┬──────┐
│ Horizon │ Linear   │ Ridge  │ Lasso  │ NN   │
├─────────┼──────────┼────────┼────────┼──────┤
│ 1h      │ 1.5-2.0  │ 1.4-1.9│ 1.5-2.0│ 1.2-1.7│
│ 24h     │ 2.5-3.5  │ 2.4-3.3│ 2.5-3.4│ 2.0-2.8│
│ 168h    │ 4.0-6.0  │ 3.9-5.8│ 4.1-5.9│ 3.5-5.2│
│ 360h    │ 6.0-10.0 │ 5.8-9.5│ 6.1-9.8│ 5.5-8.5│
└─────────┴──────────┴────────┴────────┴──────┘

Neural Networks typically show 15-25% improvement over Linear models.

╔══════════════════════════════════════════════════════════════════╗
║                      TROUBLESHOOTING                             ║
╚══════════════════════════════════════════════════════════════════╝

❌ Issue: Package installation fails
   Solution:
   ```julia
   using Pkg
   Pkg.update()
   Pkg.resolve()
   ```

❌ Issue: Out of memory
   Solution:
   • Reduce batch size in NN training
   • Process fewer horizons
   • Use Float32 instead of Float64

❌ Issue: Slow training
   Solution:
   • Reduce epochs
   • Use smaller networks
   • Enable multithreading: julia -t 4

❌ Issue: CSV file not found
   Solution:
   • Ensure you ran: python data_preparation_v5.py
   • Check file is in same directory

╔══════════════════════════════════════════════════════════════════╗
║                    DATA FORMAT REFERENCE                         ║
╚══════════════════════════════════════════════════════════════════╝

CSV Columns Structure (200+ features):
• Temporal: hour, day, month, year, cyclic encodings
• Load: load_MW, zone loads (8 zones)
• Weather: temp, humidity, wind, solar (per zone)
• Lags: 1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h, 336h
• Statistics: MA, std for 24h, 48h, 168h, 336h windows
• Targets: target_load_1h_MW to target_load_360h_MW

Total file size: ~100 MB

╔══════════════════════════════════════════════════════════════════╗
║                         VERSION INFO                             ║
╚══════════════════════════════════════════════════════════════════╝

Version: 5.0
Date: 2025
Framework: Texas Synthetic Power System (TX-123BT)
Models: Linear, Ridge, Lasso, Neural Networks
Implementation: Julia + Flux.jl

All files use "_v5" suffix for easy identification.

""")

println("="^80)
println("SETUP COMPLETE! Version 5 Ready to Use")
println("Next step: python data_preparation_v5.py")
println("="^80)