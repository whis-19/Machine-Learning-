from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler


iris=load_iris()

df = pd.DataFrame(iris.data,columns=iris.feature_names)
#creating dataframe from iris dataset

df.drop(['sepal length (cm)','sepal width (cm)'],axis='columns',inplace=True)
#dropping sepal length and sepal width from dataframe

km = KMeans(n_clusters=3)

yp = km.fit_predict(df)
df['cluster'] = yp


df1 = df[df.cluster ==0]
df2 = df[df.cluster ==1]
df3 = df[df.cluster ==2]
# plt.scatter(x="petal length (cm)",y="petal width (cm)" , data=df1 )  #
# plt.scatter(x="petal length (cm)",y="petal width (cm)" , data=df2 )  #
# plt.scatter(x="petal length (cm)",y="petal width (cm)" , data=df3 )  #
# plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
# plt.xlabel('petal length (cm)')
# plt.ylabel('petal width (cm)')
# plt.legend()
# plt.show()


sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df)
    sse.append(km.inertia_)


plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)
plt.show()