import numpy as np
import pandas as pd


# =========================================================
# 1. Load Dataset
# =========================================================

df = pd.read_csv(
    r"C:\Users\MD. ZAFAR\OneDrive\Desktop\Ai_Practice\Neural Network\student_performance.csv"
)


# =========================================================
# 2. Activation Function
# =========================================================

def step_function(z):
    if z >= 0:
        return 1
    else:
        return 0


# =========================================================
# 3. Separate Features and Target
# =========================================================

X = df[[
    "Study_Hours",
    "Attendance_%",
    "Previous_Exam_Score"
]]

Y = df[["Passed"]]


# =========================================================
# 4. Convert to NumPy
# =========================================================

X = X.to_numpy()
Y = Y.to_numpy()


print("X shape:", X.shape)
print("Y shape:", Y.shape)

print("First student:", X[0])
print("First target:", Y[0])


# =========================================================
# 5. Feature Scaling (Min-Max Scaling)
# =========================================================

X_min = X.min(axis=0)
X_max = X.max(axis=0)

X_scaled = (X - X_min) / (X_max - X_min)

print("Minimum values:", X_min)
print("Maximum values:", X_max)
print("First scaled student:", X_scaled[0])


# =========================================================
# 6. Initialize Parameters
# =========================================================

w = np.array([0.1, 0.2, 0.3])
b = 0.0

learning_rate = 0.01
epochs = 10


# =========================================================
# 7. Training
# =========================================================

for epoch in range(epochs):

    mistakes = 0

    for i in range(len(X)):

        # Weighted sum
        z = np.dot(X_scaled[i], w) + b

        # Prediction
        prediction = step_function(z)

        # Actual answer
        actual = Y[i][0]

        # Error
        error = actual - prediction

        # Update if prediction is wrong
        if error != 0:

            mistakes += 1

            w = w + learning_rate * error * X_scaled[i]

            b = b + learning_rate * error

    print("Epoch:", epoch + 1,"Mistakes:", mistakes )


# =========================================================
# 8. Final Parameters
# =========================================================

print("\nFinal weights:", w)
print("Final bias:", b)


# =========================================================
# 9. Calculate Accuracy
# =========================================================

correct = 0

for i in range(len(X)):

    z = np.dot(X_scaled[i], w) + b

    prediction = step_function(z)

    actual = Y[i][0]

    if actual == prediction:
        correct += 1


accuracy = correct / len(X)

print("Accuracy:", accuracy)
print("Correct:", correct)
print("Total:", len(X))