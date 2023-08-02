import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("insurance_data.csv")

#plt.scatter(df['age'],df['bought_insurance'],marker='+',color='red')

#plt.show()

x_train,x_test,y_train,y_test = train_test_split(df['age'],df['bought_insurance'],test_size=0.1)

x_train = x_train.values.reshape(-1,1)
x_test = x_test.values.reshape(-1,1)
y_train = y_train.values

model = LogisticRegression()
model.fit(x_train,y_train)

print(model.predict([[40]]))
print(model.predict_proba(x_test))
print(model.score(x_test,y_test))
