
import pandas as pd
import numpy as np
from sklearn import linear_model as lm
import math


df = pd.read_csv("homeprices.csv")

median_bedrooms = df.bedrooms.median()

df.bedrooms=df.bedrooms.fillna(median_bedrooms)

print(df)
 
model = lm.LinearRegression()

model.fit(df.drop("price",axis="columns"),df.price)

P_price=model.predict([[3000,3,10]])

print(P_price)

P_price=model.predict([[2500,4,5]])

print(f"Predicted price for your house is: {P_price}")


