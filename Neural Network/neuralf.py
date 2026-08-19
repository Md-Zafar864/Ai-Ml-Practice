import numpy as np 
import pandas as pd 
 
# Read the student performance CSV file
df = pd.read_csv(r"C:\Users\MD. ZAFAR\OneDrive\Desktop\Ai_Practice\Neural Network\student_performance.csv") 
 
# Check the first few rows of the dataset
# print(df.head()) 

# Check number of rows and columns
# print(df.shape) 

# Check column names
# print(df.columns) 

# Get information about data types and missing values
# print(df.info())


# Step Function:
# If weighted sum is >= 0, output 1
# If weighted sum is < 0, output 0
def step_function(z): 
    if(z >= 0): 
        return 1 
    else: 
        return 0 


# Select the input features
# Each student has 3 features:
# 1. Study Hours
# 2. Attendance %
# 3. Previous Exam Score
X = df[["Study_Hours", "Attendance_%", "Previous_Exam_Score"]] 

# Select the target/output
# 0 = Fail
# 1 = Pass
Y = df[["Passed"]] 
 
# Check the selected data
# print(X.head()) 


# Convert Pandas DataFrame into NumPy array
# This makes mathematical operations easier
X = X.to_numpy() 
Y = Y.to_numpy() 

# Check the data types
# print(type(X)) 
# print(type(Y)) 
 
# Check dimensions
# X → 200 students, 3 features
# Y → 200 students, 1 target
print(X.shape) 
print(Y.shape) 
 
# Display the first student's features
print(X[0]) 

# Display the first student's actual answer
print(Y[0]) 
 
# Check data types
# print(X.dtype) 
# print(Y.dtype) 


# Create weights for ONE neuron
# Since we have 3 input features, we need 3 weights
w = np.array([0.1, 0.2, 0.3]) 

# Bias of the neuron
b = 0.0 
 
# Check initial weights and bias
# print(w) 
# print(b) 


# Manually calculate weighted sum
# z = x1*w1 + x2*w2 + x3*w3 + b
# We are using only the first student: X[0]
# z = X[0][0]*w[0] + X[0][1]*w[1] + X[0][2]*w[2] + b 
# print("Z is :", z) 


# Calculate weighted sum using NumPy
# np.dot(X[0], w) performs:
# x1*w1 + x2*w2 + x3*w3
z2 = np.dot(X[0], w) + b 


# Pass the weighted sum through the Step Function
# This converts z into either 0 or 1
Prediction = step_function(z2) 

print("Prediction is : ", Prediction) 


# Get the actual answer for Student 1
Actual = Y[0][0] 

print("Actual is : ", Y[0][0]) 


# Calculate prediction error
# Error = Actual - Prediction
error = Actual - Prediction 

print("Error is : ", error) 


# Learning rate controls how much we change the weights
lr = 0.01 


# Update the bias using the perceptron learning rule
# new bias = old bias + learning_rate * error
b_new = b + lr * error 


# Update ALL three weights at once using NumPy
# new weights = old weights + learning_rate * error * input
#
# This is equivalent to:
# w[0] = w[0] + lr * error * X[0][0]
# w[1] = w[1] + lr * error * X[0][1]
# w[2] = w[2] + lr * error * X[0][2]
w = w + lr * error * X[0] 


# Replace the old bias with the updated bias
b = b_new 

print("New bias is: ", b_new) 


# Calculate the weighted sum AGAIN
# This time we use the UPDATED weights and bias
z3 = np.dot(X[0], w) + b 


# Make a new prediction using the updated weights
pred2 = step_function(z3) 

print("Pred is : ", pred2)