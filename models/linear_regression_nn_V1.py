import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r"C:\Users\jhonm\Visual SC Projects\opf-paper\data\synthetic_demand_5min_v1.csv")

# Select features and target
X = df[["hour", "minute", "day_of_week", "is_holiday", "demand"]].values
y = df["demand_next"].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define a simple linear model (1-layer network)
class LinearDemandPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearDemandPredictor(input_dim=5)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 200
losses = []

for epoch in range(epochs):
    model.train()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test)

mae = mean_absolute_error(y_test.numpy(), y_pred_test.numpy())
rmse = root_mean_squared_error(y_test.numpy(), y_pred_test.numpy())


print(f"\nTest MAE: {mae:.2f}")
print(f"Test RMSE: {rmse:.2f}")


# Plot real vs. predicted demand
plt.figure(figsize=(10, 5))
plt.plot(y_test.numpy(), label="Actual", alpha=0.7)
plt.plot(y_pred_test.numpy(), label="Predicted", alpha=0.7)
plt.legend()
plt.title("Demand Prediction: Real vs Predicted (Test Set)")
plt.xlabel("Sample")
plt.ylabel("Demand (MW)")
plt.tight_layout()
plt.show()
