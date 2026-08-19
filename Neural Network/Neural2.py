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
lr=0.01
epochs = 10

for epoch in range(epochs):
    mistake=0
    for i in range(len(X)):
    
        z2=np.dot(X[i],w)+b
        Prediction=step_function(z2)
        # print("Prediction is : ", Prediction)
        Actual=Y[i][0]
        # print("Actual is : ", Actual)
        error = Y[i][0] - Prediction

        if error != 0:
            mistake=mistake+1
            # print("Wrong prediction - needs update")
            w=w+lr*error*X[i]
            b=b+lr*error
        
    print("Epoch:", epoch + 1, "Mistakes:", mistake)

print("Final weights:", w)
print("Final bias:", b)
cor=0
for i in range(len(X)):
    z=np.dot(w,X[i])+b
    pred=step_function(z)
    actual2=Y[i][0]
    if(actual2==pred):
        cor=cor+1
accuracy = cor / len(X)
print(accuracy)
print(len(X))
