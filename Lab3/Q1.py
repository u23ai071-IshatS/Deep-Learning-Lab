import math
import random

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

data = [
    (0, 0, 0),
    (0, 1, 0),
    (1, 0, 0),
    (1, 1, 1)
]

w1 = random.uniform(-1, 1)
w2 = random.uniform(-1, 1)
b = random.uniform(-1, 1)

lr = 0.5
epochs = 10000

for epoch in range(epochs):
    total_loss = 0

    for x1, x2, y in data:
        z = w1*x1 + w2*x2 + b
        y_hat = sigmoid(z)

        loss = 0.5 * (y_hat - y)**2
        total_loss += loss 

        dz = (y_hat - y) * y_hat * (1 - y_hat) 

        w1 -= lr * dz * x1
        w2 -= lr * dz * x2
        b  -= lr * dz

    if epoch % 1000 == 0:
        print("Epoch", epoch, "Loss", total_loss)

print("\nLearned parameters:")
print("w1 =", w1)
print("w2 =", w2)
print("b =", b)

print("\nPredictions:")
for x1, x2, y in data:
    y_hat = sigmoid(w1*x1 + w2*x2 + b)
    print((x1, x2), "→", round(y_hat, 4))
