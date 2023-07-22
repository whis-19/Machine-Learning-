import pandas as pd
import numpy as np
from sklearn import linear_model as lm
import matplotlib.pyplot as plt

df = pd.read_csv("canada_per_capita_income.csv")

plt.xlabel("year")
plt.ylabel("PCI") #per capita income
plt.scatter(df.year,df.PCI,color="green",marker="+")



new_df = df.drop("PCI",axis="columns")

model=lm.LinearRegression()

model.fit(new_df,df.PCI)

R_PCI = model.predict([[2020]])[0]

print(f"2020 per capita income of Canada will be: {R_PCI}")

plt.show()