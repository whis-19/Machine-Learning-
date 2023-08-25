import pandas as pd
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sn

digits = load_digits()



# print(dir(digits))
# print(digits.feature_names)

df = pd.DataFrame(digits.data,columns=digits.feature_names)

df['target'] = digits.target

df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]

x=df.drop(['target'],axis = 'columns')
y=df.target


x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.2,random_state=1)

model = KNeighborsClassifier(n_neighbors=5)

model.fit(x_train,y_train)

print(model.score(x_test,y_test))

y_pred = model.predict(x_test)

cm = confusion_matrix(y_test,y_pred)



sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth Value')
plt.show()
print(classification_report(y_test,y_pred))