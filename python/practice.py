import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score 
df=pd.read_csv(r'C:\Users\MD. ZAFAR\OneDrive\Desktop\Ai_Practice\python\food.csv');


df = df.dropna()



# print(df[df["Proteins_g"].str.contains(",", na=False)])
df["Proteins_g"] = df["Proteins_g"].str.replace(",", ".", regex=False)
df["Proteins_g"] = pd.to_numeric(
    df["Proteins_g"].str.replace(",", ".", regex=False),
    errors="coerce"
)
df = df.dropna()
print(df.columns)
X= df[["Serving_Size_g", "Carbohydrates_g", "Proteins_g", "Sub_Category", "Category" ]]
Y=df["Calories_kcal"]
X = pd.get_dummies(
    X,
    columns=[ "Category", "Sub_Category"],
    drop_first=True
)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)


from sklearn.linear_model import ElasticNet

model = ElasticNet(alpha=0.01, l1_ratio=0.8)
model.fit(X_train, Y_train)
scores = cross_val_score(model, X_train, Y_train, cv=5)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
print("Cross-Validation Scores:", scores)
print("Mean CV Score:", scores.mean())
Y_pred = model.predict(X_test)
comparison = pd.DataFrame({
    "Actual": Y_test.values,
    "Predicted": Y_pred
})

print(comparison.head(10))
score = r2_score(Y_test, Y_pred)

print("R² Score:", score)
print("Training R²:", model.score(X_train, Y_train))
print("Testing R² :", model.score(X_test, Y_test))