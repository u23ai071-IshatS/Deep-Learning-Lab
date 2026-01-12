import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ------------------------------
# Load CSV
# ------------------------------
data = pd.read_csv("Lab1/house_data.csv")

X = data[["bedrooms", "sqft_living"]].values
Y = data["price"].values

# Normalize
X = (X - X.mean(axis=0)) / X.std(axis=0)
Y = Y / Y.max()

# ------------------------------
# Keras Sequential Model (2-2-1)
# ------------------------------
model = Sequential([
    Dense(2, input_dim=2, activation='relu'),   # Hidden layer (2 neurons)
    Dense(1)                                    # Output layer
])

model.compile(
    optimizer='adam',
    loss='mse'
)

# ------------------------------
# Training
# ------------------------------
history = model.fit(X, Y, epochs=500, verbose=0)

# ------------------------------
# Prediction
# ------------------------------
predicted = model.predict(X) * data["price"].max()
actual = data["price"].values

# ------------------------------
# Plot: Actual vs Predicted
# ------------------------------
plt.figure()
plt.plot(actual, label="Actual Prices")
plt.plot(predicted, label="Predicted Prices")
plt.xlabel("House Index")
plt.ylabel("Price")
plt.title("House Price Prediction (Keras)")
plt.legend()
plt.show()

# ------------------------------
# Plot: Training Loss
# ------------------------------
plt.figure()
plt.plot(history.history['loss'])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve (Keras)")
plt.show()
