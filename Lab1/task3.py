import numpy as np

class Perceptron:
    def __init__(self, lr=0.1, epochs=10):
        self.lr = lr
        self.epochs = epochs
        self.weights = np.zeros(3)  # w1, w2, bias

    def activation(self, x):
        return 1 if x >= 0 else 0

    def train(self, X, y):
        for _ in range(self.epochs):
            for i in range(len(X)):
                input_with_bias = np.insert(X[i], 0, 1)
                z = np.dot(self.weights, input_with_bias)
                y_pred = self.activation(z)
                error = y[i] - y_pred
                self.weights += self.lr * error * input_with_bias

    def predict(self, X):
        results = []
        for x in X:
            x = np.insert(x, 0, 1)
            results.append(self.activation(np.dot(self.weights, x)))
        return results

# Input
X = np.array([[0,0],[0,1],[1,0],[1,1]])

# AND Gate
y_and = np.array([0,0,0,1])
p_and = Perceptron()
p_and.train(X, y_and)
print("AND Gate Output:", p_and.predict(X))

# OR Gate
y_or = np.array([0,1,1,1])
p_or = Perceptron()
p_or.train(X, y_or)
print("OR Gate Output:", p_or.predict(X))
