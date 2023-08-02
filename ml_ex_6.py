import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('titanic.csv')

df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

# Fill missing values with 0
df.fillna(0, inplace=True)

inputs = df.drop('Survived', axis=1)
target = df.Survived

inputs['Sex'] = inputs['Sex'].astype(str).map({'male': 1, 'female': 2})  # Map and update 'Sex' column

x_train, x_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2)

print(len(x_train), " ", len(x_test))

model = DecisionTreeClassifier()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))
