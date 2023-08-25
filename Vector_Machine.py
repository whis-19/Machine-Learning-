from sklearn.datasets import load_iris 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
iris = load_iris()

df = pd.DataFrame(iris.data,columns=iris.feature_names)
df['target'] = iris.target
df['flower_names']=df.target.apply(lambda x: iris.target_names[x])

print(df[45:55]) 
df0 = df[:50]
df1 = df[50:100]
df2 = df[100:] 

# plt.xlabel('Sepal Length')
# plt.ylabel('Sepal Width')

# plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color='green',marker='+')
# plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color='blue',marker='+')
# plt.scatter(df2['sepal length (cm)'],df2['sepal width (cm)'],color='red',marker='+')
# plt.show()

# plt.xlabel('Petal Length')
# plt.ylabel('Petal Width')

# plt.scatter(df0['petal length (cm)'],df0['petal width (cm)'],color='green',marker='+')
# plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color='blue',marker='+')
# plt.scatter(df2['petal length (cm)'],df2['petal width (cm)'],color='red',marker='+')
# plt.show()

X = df.drop(['target','flower_names'],axis='columns')

Y = df.target

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size= 0.2)

model = SVC()
model_ = SVC(C= 20)
#gamma and C to manupilate model score
model.fit(x_train,y_train)
model_.fit(x_train,y_train)

print(model.score(x_test,y_test))
print(model_.score(x_test,y_test))





