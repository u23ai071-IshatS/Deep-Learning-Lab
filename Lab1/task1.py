import torch

print("----- PyTorch Tensor Basics -----")

# Tensor Initialization
a = torch.tensor([1, 2, 3])
b = torch.zeros((2, 3))
c = torch.ones((3, 2))
d = torch.rand((2, 2))

print("Tensor a:", a)
print("Zeros:\n", b)
print("Ones:\n", c)
print("Random:\n", d)

# Data types
x = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float32)
y = torch.tensor([1, 2, 3], dtype=torch.int32)
print("Float Tensor:", x)
print("Int Tensor:", y)

# Arithmetic Operations
t1 = torch.tensor([1, 2, 3])
t2 = torch.tensor([4, 5, 6])

print("Addition:", t1 + t2)
print("Subtraction:", t1 - t2)
print("Multiplication:", t1 * t2)
print("Division:", t1 / t2)

# Broadcasting
m = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
n = torch.tensor([1, 2, 3])
print("Broadcasting:\n", m + n)

# Indexing
print("First element:", t1[0])
print("Slice:", t1[1:])

# Reshaping
r = torch.arange(12)
r2 = r.reshape(3, 4)
print("Reshaped Tensor:\n", r2)

# Autograd
x = torch.tensor(2.0, requires_grad=True)
y = x**3 + 2*x
y.backward()

print("Gradient dy/dx:", x.grad)
