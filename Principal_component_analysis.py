import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA


digits = load_digits()

# print(pd.DataFrame(digits))

# print(digits.keys())

digits.data[0].reshape(8,8)
plt.gray()
# plt.matshow(digits.data[0].reshape(8,8))
scalar = StandardScaler()

# print(digits.target)

df = pd.DataFrame(digits.data)

x = df
y = digits.target

x_scaled = scalar.fit_transform(x)

pca = PCA(0.95)
x_pca = pca.fit_transform(x)
print(x_pca.shape)

print(x_scaled.shape)

x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,test_size=0.2)

model = LogisticRegression()

model.fit(x_train,y_train)   
print(model.score(x_test,y_test))

x_train,x_test,y_train,y_test = train_test_split(x_pca,y,test_size=0.2)

model = LogisticRegression()

model.fit(x_train,y_train)
print(model.score(x_test,y_test))



