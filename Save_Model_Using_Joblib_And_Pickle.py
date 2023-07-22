import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot
import math
import pickle
import joblib


#df = pd.read_csv("homeprices.csv")
#print(df)

# model = linear_model.LinearRegression()
# model.fit(df[['area']],df.price)

# with open('model_pickel','wb') as file:
#     pickle.dump(model,file)
    
# with open('model_pickel','rb') as f:
#     mp = pickle.load(f)
    

#print(mp.predict([[5000]]))

#joblib.dump(model,'model_joblib')
mj = joblib.load('model_joblib')

cf = mj.coef_
price = mj.predict([[5000]])
print(price)


