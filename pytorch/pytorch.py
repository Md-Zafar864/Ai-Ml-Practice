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
WK = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0]
])

WV = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0]
])

Q = X @ WQ
K= X @ WK
V= X @ WV

score=K @ Q.T

d_k=K.shape[1]

score = score / torch.sqrt(
    torch.tensor(float(d_k))
)

att_weight=torch.softmax(score,dim=1)
print(att_weight)

att_weight= att_weight@V