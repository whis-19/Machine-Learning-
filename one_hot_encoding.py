import pandas as pd
import numpy as np
import matplotlib.pyplot
from sklearn import linear_model as lm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


df = pd.read_csv("homeprices.csv")


dumies = pd.get_dummies(df.town)

      
df_dumies = pd.concat([df,dumies],axis='columns')

y= df_dumies.price

df_dumies.drop(['town','west windsor','price'],axis="columns",inplace=True )

x = df_dumies



model = lm.LinearRegression()

model.fit(x,y)

predict_ww = model.predict([[3400,0,0]])
predict_rw = model.predict([[2800,0,1]])
predict_mt = model.predict([[2600,1,0]])

print(f"The predicted price for house in west windsor is: {predict_ww}")
print(f"The predicted price for house in robinsville is: {predict_rw}")
print(f"The predicted price for house in monroe township is: {predict_mt}")



#new metrhod

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
le = LabelEncoder()


dfle = df

dfle.town = le.fit_transform(dfle.town)

x = dfle[['town','area']].values

y = dfle.price.values

ct = ColumnTransformer([('town',OneHotEncoder(),[0])],remainder='passthrough')

x=ct.fit_transform(x) 

x= x[:,1:]


model.fit(x,y)

predict_WW = model.predict([[0,1,3400]])
predict_RW = model.predict([[1,0,2800]])
predict_MT = model.predict([[0,0,2600]])

print(f"The predicted price for house in west windsor is: {predict_WW}")
print(f"The predicted price for house in robinsville is: {predict_RW}")
print(f"The predicted price for house in monroe township is: {predict_MT}")

