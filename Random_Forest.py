import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


digits = load_digits()

df=pd.DataFrame(digits.data)

df['target'] = digits.target

x = df.drop('target',axis='columns')

y = df.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)



model = RandomForestClassifier(n_estimators=25)

model.fit(x_train,y_train)

y_predicted=model.predict(x_test)

print(model.score(x_test,y_test))

cm = confusion_matrix(y_test,y_predicted)

