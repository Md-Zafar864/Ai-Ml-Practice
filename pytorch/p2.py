import torch
import torch.nn as nn

X = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],   # cat
    [0.0, 1.0, 0.0, 1.0],   # eats
    [1.0, 1.0, 0.0, 0.0]    # food
])
WQ=nn.Linear(4,2)
WK=nn.Linear(4,2)
WV=nn.Linear(4,2)


Q = WQ(X)
K = WK(X)
V = WV(X)

print("Q:")
print(Q)

print("\nK:")
print(K)

print("\nV:")
print(V)

print("\nShapes:")
print("Q:", Q.shape)
print("K:", K.shape)
print("V:", V.shape)