import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression 


df = pd.read_csv("carprices.csv")



df_dummies = pd.get_dummies(df['Car Model'])


new_df = pd.concat([df,df_dummies],axis='columns')
new_df=new_df.drop(['Car Model','Mercedez Benz C class'],axis = 'columns')


X = new_df.drop("Sell Price($)",axis='columns')

Y= new_df['Sell Price($)']

model = LinearRegression()

model.fit(X,Y)

mercedez = model.predict([[45000,4,0,0]])
bmw = model.predict([[86000,7,0,1]])

print(mercedez)
print(bmw)