import math
import random

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

data = [
    (0,0,0),
    (0,1,1),
    (1,0,1),
    (1,1,0)
]

def rand():
    return random.uniform(-1,1)

w11, w12, b1 = rand(), rand(), rand()
w21, w22, b2 = rand(), rand(), rand()
v1, v2, b3 = rand(), rand(), rand()

lr = 0.5 
epochs = 20000

for epoch in range(epochs):
    total_loss = 0

    for x1, x2, y in data:

        z1 = w11*x1 + w12*x2 + b1
        h1 = sigmoid(z1) 

        z2 = w21*x1 + w22*x2 + b2
        h2 = sigmoid(z2) 

        z3 = v1*h1 + v2*h2 + b3
        y_hat = sigmoid(z3)

        loss = 0.5 * (y_hat - y)**2
        total_loss += loss

        dz3 = (y_hat - y) * y_hat * (1 - y_hat)

        dv1 = dz3 * h1 
        dv2 = dz3 * h2
        db3 = dz3 

        dz1 = dz3 * v1 * h1 * (1 - h1)
        dz2 = dz3 * v2 * h2 * (1 - h2) 

        dw11 = dz1 * x1
        dw12 = dz1 * x2
        db1  = dz1

        dw21 = dz2 * x1
        dw22 = dz2 * x2
        db2  = dz2

        v1 -= lr * dv1
        v2 -= lr * dv2
        b3 -= lr * db3

        w11 -= lr * dw11
        w12 -= lr * dw12
        b1  -= lr * db1

        w21 -= lr * dw21
        w22 -= lr * dw22
        b2  -= lr * db2

    if epoch % 2000 == 0:
        print("Epoch", epoch, "Loss", total_loss)

print("\nPredictions:")
for x1, x2, y in data:
    h1 = sigmoid(w11*x1 + w12*x2 + b1)
    h2 = sigmoid(w21*x1 + w22*x2 + b2)
    y_hat = sigmoid(v1*h1 + v2*h2 + b3)
    print((x1,x2), "→", round(y_hat,4))

