import pandas as pd
import numpy as np

df=pd.read_csv(r'C:\Users\MD. ZAFAR\OneDrive\Desktop\Ai_Practice\python\food.csv');

# print(df.shape)
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())
# df["Proteins_g"].unique()
# print(df.info())
# print(df["Proteins_g"].unique())
df = df.dropna()
# print(df.isnull().sum())

print(df["Proteins_g"].head(20))
print(df["Proteins_g"].unique())
print(df[df["Proteins_g"].str.contains(",", na=False)])
df["Proteins_g"] = df["Proteins_g"].str.replace(",", ".", regex=False)
df["Proteins_g"] = pd.to_numeric(
    df["Proteins_g"].str.replace(",", ".", regex=False),
    errors="coerce"
)
print(df.dtypes)