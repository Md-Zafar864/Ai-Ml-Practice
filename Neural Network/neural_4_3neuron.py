import pandas as pd
import numpy as np

df=pd.read_csv(r"C:\Users\MD. ZAFAR\OneDrive\Desktop\Ai_Practice\Neural Network\student_performance.csv")
print(df.columns)
X=df[["Study_Hours", "Attendance_%", "Previous_Exam_Score" ]]
print(X.head())
Y=df[["Passed"]]
print(Y.head())

X=X.to_numpy()
Y=Y.to_numpy()
Xmin=X.min(axis=0)
Xmax=X.max(axis=0)

X_scaled=(X-Xmin)/(Xmax-Xmin)
print("Scales")
# print(X_scaled)

W = np.array([
    [0.2, 0.5, 0.1],
    [0.4, 0.1, 0.7],
    [0.3, 0.6, 0.2]
])

b=np.array([0.10, 0.05, 0.02])

z=X_scaled@ W + b

# print("Z is :", z)
A=np.maximum(z,0)
# print("Z shape :", z.shape)
# print("A shape :", A.shape)
# print(A)

w2=np.array([[0.2],[0.4],[0.5]])
print("W2 shape", w2.shape)
b2=0.1
z2=A@w2 + b2

# print("Z2 shape", z2.shape)
# print("Z2", z2)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

A2=sigmoid(z2)
print("Minimum A2" ,A2.min())
print("A2 shape",A2.shape)
prediction = (A2 >= 0.5).astype(int)

# print(prediction)
print("Pred",prediction.shape)
accuracy=np.mean(prediction==Y)
print(accuracy)

epsilon = 1e-15

A2_clipped = np.clip(A2, epsilon, 1 - epsilon)

loss = -np.mean(
    Y * np.log(A2_clipped) +
    (1 - Y) * np.log(1 - A2_clipped)
)

print("Loss:", loss)