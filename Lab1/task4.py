import torch
import torch.nn as nn
import torch.optim as optim

# XOR Data
X = torch.tensor([[0.,0.],
                  [0.,1.],
                  [1.,0.],
                  [1.,1.]])

Y = torch.tensor([[0.],
                  [1.],
                  [1.],
                  [0.]])

# Neural Network
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

model = XORNet()

# Correct loss for binary classification
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Training
for epoch in range(10000):
    y_pred = model(X)
    loss = criterion(y_pred, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 2000 == 0:
        print(f"Epoch {epoch}, Loss = {loss.item():.9f}")

# Test
print("\nInput:")
print(X)

print("\nFinal Output:")
with torch.no_grad():
    output = model(X)
    print(output.round())
