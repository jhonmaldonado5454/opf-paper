# Load required packages
using CSV, DataFrames            # For data handling
using Flux                       # For neural network modeling
using Statistics                 # For computing loss and metrics
using Random                     # For reproducible data splitting
using Printf                     # For formatted output
using Plots                      # For plotting predictions

# Load the dataset
df = DataFrame(CSV.File("data/synthetic_demand_5min.csv"))

# Select features and target
X = [df.hour df.minute df.day_of_week df.is_holiday df.demand]
X = convert(Matrix{Float32}, X)                  # Convert to Float32 matrix
y = convert(Vector{Float32}, df.demand_next)     # Target vector

# Split into training and test sets (80/20)
Random.seed!(42)
N = size(X, 1)
test_size = Int(floor(0.2 * N))
perm = randperm(N)
test_idx = perm[1:test_size]
train_idx = perm[test_size+1:end]

X_train = X[train_idx, :]
y_train = y[train_idx]
X_test  = X[test_idx, :]
y_test  = y[test_idx]

# Transpose for Flux (inputs as columns)
X_train_t = X_train'
X_test_t  = X_test'
y_train_t = reshape(y_train, 1, :)

# Define a linear regression model (one Dense layer with bias)
model = Chain(Dense(5, 1))  # 5 input features → 1 output

# Define loss function and optimizer
loss(m, x, y) = mean((m(x) .- y).^2)   # Mean Squared Error
opt = ADAM(0.01)
state = Flux.setup(opt, model)        # Optimizer state setup

# Training loop
epochs = 200
for epoch in 1:epochs
    grads = gradient(m -> loss(m, X_train_t, y_train_t), model)
    Flux.update!(state, model, grads)

    if epoch % 20 == 0
        current_loss = loss(model, X_train_t, y_train_t)
        @printf("Epoch %d, Loss: %.4f\n", epoch, current_loss)
    end
end

# Predict on the test set
y_pred_test_matrix = model(X_test_t)
y_pred_test = vec(y_pred_test_matrix)  # Convert from 1×N to Vector

# Compute evaluation metrics
mae  = mean(abs.(y_test .- y_pred_test))
rmse = sqrt(mean((y_test .- y_pred_test).^2))
@printf("\nTest MAE: %.2f\n", mae)
@printf("Test RMSE: %.2f\n", rmse)

# Plot actual vs. predicted demand
plot(y_test, label="Actual", alpha=0.7)
plot!(y_pred_test, label="Predicted", alpha=0.7)
xlabel!("Sample")
ylabel!("Demand (MW)")
title!("Actual vs. Predicted Demand (Test Set)")
