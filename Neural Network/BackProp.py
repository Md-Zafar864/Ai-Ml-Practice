import pandas as pd
import numpy as np


# 1. Load data
df = pd.read_csv(
    r"C:\Users\MD. ZAFAR\OneDrive\Desktop\Ai_Practice\Neural Network\student_performance.csv"
)

print(df.columns)


# 2. Select features and target
X = df[["Study_Hours", "Attendance_%", "Previous_Exam_Score"]]
Y = df[["Passed"]]

print(X.head())
print(Y.head())


# 3. Convert DataFrame to NumPy
X = X.to_numpy()
Y = Y.to_numpy()


# 4. Min-Max scaling
Xmin = X.min(axis=0)
Xmax = X.max(axis=0)
X_scaled = (X - Xmin) / (Xmax - Xmin)

print("Scaling completed")


# 5. Hidden layer: 3 inputs → 3 neurons
W = np.array([
    [0.2, 0.5, 0.1],
    [0.4, 0.1, 0.7],
    [0.3, 0.6, 0.2]
])

b = np.array([0.10, 0.05, 0.02])


# 6. Output layer: 3 neurons → 1 neuron
w2 = np.array([
    [0.2],
    [0.4],
    [0.5]
])

b2 = 0.1


# 7. Learning rate
lr = 0.2


# 8. Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 9. Training loop
for epoch in range(1000):

    # ---------- Forward Propagation ----------

    # Hidden layer
    z = X_scaled @ W + b
    A = np.maximum(z, 0)       # ReLU

    # Output layer
    z2 = A @ w2 + b2
    A2 = sigmoid(z2)           # Sigmoid


    # ---------- Loss ----------

    epsilon = 1e-15
    A2_clipped = np.clip(A2, epsilon, 1 - epsilon)

    loss = -np.mean(
        Y * np.log(A2_clipped) +
        (1 - Y) * np.log(1 - A2_clipped)
    )


    # ---------- Backpropagation ----------

    # Output layer gradients
    error = A2 - Y
    dW2 = A.T @ error / 200
    db2 = np.mean(error)

    # Hidden layer gradients
    dA = error @ w2.T
    relu_derivative = (z > 0).astype(int)
    dZ = dA * relu_derivative

    dW = X_scaled.T @ dZ / 200
    db = np.mean(dZ, axis=0)


    # ---------- Update Parameters ----------

    w2 = w2 - lr * dW2
    b2 = b2 - lr * db2

    W = W - lr * dW
    b = b - lr * db


    # ---------- Check Progress ----------

    if (epoch + 1) % 100 == 0:

        prediction = (A2 >= 0.5).astype(int)
        accuracy = np.mean(prediction == Y)

        print(
            "Epoch:", epoch + 1,
            "Loss:", loss,
            "Accuracy:", accuracy
        )