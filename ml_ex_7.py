import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

digits = load_digits()

df = pd.DataFrame(digits.data, columns=digits.feature_names)
df['target'] = digits.target

x = df.drop(['target'], axis='columns')
y = df['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

model = SVC(kernel='rbf')
mod = SVC(kernel='linear')

model.fit(x_train, y_train)
mod.fit(x_train, y_train)

print(model.score(x_test, y_test))
print(mod.score(x_test, y_test))
