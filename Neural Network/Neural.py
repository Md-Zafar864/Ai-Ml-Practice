import numpy as np
import pandas as pd

df=pd.read_csv(r"C:\Users\MD. ZAFAR\OneDrive\Desktop\Ai_Practice\Neural Network\student_performance.csv")

# print(df.head())
# print(df.shape)
# print(df.columns)
# print(df.info())
def step_function(z):
    if(z>=0):
        return 1
    else:
        return 0
X= df[["Study_Hours", "Attendance_%", "Previous_Exam_Score" ]]
Y=df[["Passed"]]

# print(X.head())
X=X.to_numpy()
Y=Y.to_numpy()
# print(type(X))
# print(type(Y))

print(X.shape)
print(Y.shape)

print(X[0])
print(Y[0])

# print(X.dtype)
# print(Y.dtype)
w=np.array([0.1,0.2,0.3])
b=0.0
# print(w)
# print(b)

# z=X[0][0]*w[0]+ X[0][1]*w[1] + X[0][2]*w[2] + b
# print("Z is :",z)

z2=np.dot(X[0],w)+b
Prediction=step_function(z2)
print("Prediction is : ", Prediction)
Actual=Y[0][0]
print("Actual is : ", Y[0][0])

error= Actual-Prediction
print("Error is : ", error)

lr=0.01
w1_new= w[0]+ lr*error*X[0][0]
print("w[0] is: ",w[0])
print("Updated weight 1 is : ", w1_new)

w2_new=w[1]+ lr*error*X[0][1]
print("w[1] is: ",w[1])
print("Updated weight 2 is : ", w2_new)

print("w3 is : ", w[2])

w3_new=w[2] + lr*error*X[0][2]
print("Updated weight 3 is : ", w3_new)

b_new=b+ lr*error
w[0] = w1_new
w[1] = w2_new
w[2] = w3_new

b = b_new

print("New bias is: ", b_new)
z3=np.dot(X[0], w)+b

pred2=step_function(z3)

print("Pred is : ", pred2)