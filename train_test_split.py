from hmac import new
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv("carprices.csv")


# plt.scatter(df['Mileage'],df['Sell Price($)'])

# plt.show()

# plt.scatter(df['Age(yrs)'],df['Sell Price($)'])
# plt.show()

x = df[['Age(yrs)','Mileage']]
y = df['Sell Price($)']

print(x)
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=14)



model = LinearRegression()
model.fit(x_train,y_train)
print(x_test)
print(model.predict(x_test))
print(y_test)

print(model.score(x_test,y_test))