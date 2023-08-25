import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


iris = load_iris()

print(dir(iris))
df = pd.DataFrame(iris.data,columns=iris.feature_names)


df['target']= iris.target

x=df.drop(['target'],axis='columns')
y=df.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


model = RandomForestClassifier(n_estimators=69)

model.fit(x_train,y_train)

print(model.score(x_test,y_test))

