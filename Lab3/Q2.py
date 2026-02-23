import math
import random

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

data = []
for x1 in [0,1]:
    for x2 in [0,1]:
        for x3 in [0,1]:
            y = 1 if (x1 or x2 or x3) else 0
            data.append((x1, x2, x3, y))

w1 = random.uniform(-1, 1)
w2 = random.uniform(-1, 1)
w3 = random.uniform(-1, 1)
b = random.uniform(-1, 1)

lr = 0.5
epochs = 15000

for epoch in range(epochs):
    total_loss = 0

    for x1, x2, x3, y in data:
        z = w1*x1 + w2*x2 + w3*x3 + b
        y_hat = sigmoid(z)

        loss = 0.5 * (y_hat - y)**2
        total_loss += loss

        dz = (y_hat - y) * y_hat * (1 - y_hat)

        w1 -= lr * dz * x1
        w2 -= lr * dz * x2
        w3 -= lr * dz * x3
        b  -= lr * dz

    if epoch % 2000 == 0:
        print("Epoch", epoch, "Loss", total_loss)

print("\nLearned parameters:")
print("w1 =", w1)
print("w2 =", w2)
print("w3 =", w3)
print("b =", b)

correct = 0
print("\nPredictions:")
for x1, x2, x3, y in data:
    y_hat = sigmoid(w1*x1 + w2*x2 + w3*x3 + b)
    pred = 1 if y_hat >= 0.5 else 0
    if pred == y:
        correct += 1
    print((x1,x2,x3), "→", round(y_hat,4), "class:", pred)

accuracy = correct / len(data)
print("\nAccuracy =", accuracy)
