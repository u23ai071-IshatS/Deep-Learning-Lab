import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

torch.manual_seed(0)

# ------------------------------
# Load CSV
# ------------------------------
data = pd.read_csv("Lab1/house_data.csv")

X = data[["bedrooms", "sqft_living"]].values
Y = data[["price"]].values

# Normalize
X = (X - X.mean(axis=0)) / X.std(axis=0)
Y = Y / Y.max()

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)

# ------------------------------
# 2-2-1 Neural Network
# ------------------------------
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(2, 2)
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x

model = SimpleNN()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

# ------------------------------
# Training
# ------------------------------
losses = []

for epoch in range(300):
    pred = model(X)
    loss = criterion(pred, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss = {loss.item():.6f}")

# ------------------------------
# Predictions
# ------------------------------
with torch.no_grad():
    predicted = model(X).numpy() * data["price"].max()

actual = data["price"].values

# ------------------------------
# Plot: Actual vs Predicted
# ------------------------------
plt.figure()
plt.plot(actual, label="Actual Prices")
plt.plot(predicted, label="Predicted Prices")
plt.xlabel("House Index")
plt.ylabel("Price")
plt.title("House Price Prediction using Neural Network")
plt.legend()
plt.show()

# ------------------------------
# Plot: Training Loss
# ------------------------------
plt.figure()
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()
