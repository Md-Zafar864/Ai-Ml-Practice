import torch

X = torch.tensor([
    [1.0, 0.0, 1.0],
    [0.0, 2.0, 0.0],
    [1.0, 1.0, 0.0]
])

WQ = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0]
])

Q = X @ WQ

print("X:")
print(X)

print("\nQ:")
print(Q)

print("\nQ shape:")
print(Q.shape)