import pandas as pd
import numpy as np


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(
    r"C:\Users\MD. ZAFAR\OneDrive\Desktop\Ai_Practice\Neural Network\student_performance.csv"
)

print(df.columns)


# ============================================================
# 2. SELECT INPUT FEATURES (X) AND TARGET (Y)
# ============================================================

X = df[
    [
        "Study_Hours",
        "Attendance_%",
        "Previous_Exam_Score"
    ]
]

Y = df[["Passed"]]

print(X.head())
print(Y.head())


# ============================================================
# 3. CONVERT DATAFRAME TO NUMPY ARRAY
# ============================================================

X = X.to_numpy()
Y = Y.to_numpy()


# ============================================================
# 4. MIN-MAX SCALING
# ============================================================

Xmin = X.min(axis=0)
Xmax = X.max(axis=0)

X_scaled = (X - Xmin) / (Xmax - Xmin)

print("Scaling completed")


# ============================================================
# 5. INITIALIZE HIDDEN LAYER PARAMETERS
#
# 3 inputs → 3 hidden neurons
#
# W shape = (3, 3)
# b shape = (3,)
# ============================================================

W = np.array([
    [0.2, 0.5, 0.1],
    [0.4, 0.1, 0.7],
    [0.3, 0.6, 0.2]
])

b = np.array([
    0.10,
    0.05,
    0.02
])


# ============================================================
# 6. INITIALIZE OUTPUT LAYER PARAMETERS
#
# 3 hidden neurons → 1 output neuron
#
# W2 shape = (3, 1)
# b2 shape = scalar
# ============================================================

w2 = np.array([
    [0.2],
    [0.4],
    [0.5]
])

b2 = 0.1


# ============================================================
# 7. LEARNING RATE
# ============================================================

lr = 0.2


# ============================================================
# 8. SIGMOID FUNCTION
# ============================================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ============================================================
# 9. TRAINING LOOP
# ============================================================

for epoch in range(1000):


    # ========================================================
    # FORWARD PROPAGATION
    # ========================================================

    # -------- Hidden Layer --------

    # X_scaled shape = (200, 3)
    # W shape        = (3, 3)
    # b shape        = (3,)
    #
    # Result:
    # z shape = (200, 3)

    z = X_scaled @ W + b


    # ReLU activation
    #
    # A shape = (200, 3)

    A = np.maximum(z, 0)


    # -------- Output Layer --------

    # A shape  = (200, 3)
    # w2 shape = (3, 1)
    #
    # z2 shape = (200, 1)

    z2 = A @ w2 + b2


    # Sigmoid activation
    #
    # A2 shape = (200, 1)

    A2 = sigmoid(z2)


    # ========================================================
    # LOSS
    # ========================================================

    epsilon = 1e-15

    A2_clipped = np.clip(
        A2,
        epsilon,
        1 - epsilon
    )

    loss = -np.mean(
        Y * np.log(A2_clipped)
        +
        (1 - Y) * np.log(1 - A2_clipped)
    )


    # ========================================================
    # BACKPROPAGATION
    # ========================================================

    # --------------------------------------------------------
    # OUTPUT LAYER
    # --------------------------------------------------------

    # Error of output layer
    #
    # dZ2 = A2 - Y

    error = A2 - Y


    # Gradient of output weights
    #
    # A.T shape      = (3, 200)
    # error shape    = (200, 1)
    # dW2 shape      = (3, 1)

    dW2 = A.T @ error / 200


    # Gradient of output bias
    #
    # db2 is a single value

    db2 = np.mean(error)


    # --------------------------------------------------------
    # HIDDEN LAYER
    # --------------------------------------------------------

    # Send the error backward through W2

    # error shape = (200, 1)
    # w2.T shape = (1, 3)
    # dA shape = (200, 3)

    dA = error @ w2.T
    # Derivative of ReLU
    # If z > 0 → 1
    # If z <= 0 → 0

    relu_derivative = (z > 0).astype(int)


    # Gradient with respect to hidden layer Z

    dZ = dA * relu_derivative


    # Gradient of hidden layer weights
    #
    # X_scaled.T shape = (3, 200)
    # dZ shape        = (200, 3)
    #
    # dW shape = (3, 3)

    dW = X_scaled.T @ dZ / 200


    # Gradient of hidden layer bias
    #
    # db shape = (3,)

    db = np.mean(dZ, axis=0)


    # ========================================================
    # UPDATE PARAMETERS
    # ========================================================

    # -------- Output Layer --------

    w2 = w2 - lr * dW2

    b2 = b2 - lr * db2


    # -------- Hidden Layer --------

    W = W - lr * dW

    b = b - lr * db


    # ========================================================
    # PRINT PROGRESS
    # ========================================================

    if (epoch + 1) % 100 == 0:

        prediction = (A2 >= 0.5).astype(int)

        accuracy = np.mean(prediction == Y)

        print(
            "Epoch:", epoch + 1,
            "Loss:", loss,
            "Accuracy:", accuracy
        )