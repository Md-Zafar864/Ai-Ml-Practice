import numpy as np

X = np.array([
    [1.0, 0.0, 1.0],
    [0.0, 2.0, 0.0],
    [1.0, 1.0, 0.0]
])

print(X.shape)

WQ=np.array([
    [1,0],
    [0,1],
    [1,0]
])

WK=np.array([
    [1,0],
    [0,1],
    [1,0]
])
WV=np.array([
    [1,0],
    [0,1],
    [1,0]
])
Q = X @ WQ
K = X @ WK
V = X @ WV
# print("Q-",Q)
# print("K-",K)
# print("V-",V)



score=Q@K.T
# print("Score: ",score)
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
d_k=K.shape[1]
# print(d_k)
scaled_score=score/np.sqrt(d_k)
# print(scaled_score)

mask=np.triu(np.ones((3,3)),k=1)

masked=np.where(
    mask==1,
    -np.inf,
    scaled_score
)


att_weight=softmax(masked)

print("Att weight",att_weight)

finl=att_weight@V

print(finl)