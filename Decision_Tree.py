import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

df = pd.read_csv("salaries.csv")

inputs = df.drop('salary_more_then_100k',axis='columns')

target = df['salary_more_then_100k']

le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_job.fit_transform(inputs['job'])
inputs['degree_n'] = le_company.fit_transform(inputs['degree'])
print(inputs)
inputs_n = inputs.drop(['company','job','degree'],axis='columns')

model = tree.DecisionTreeClassifier()

model.fit(inputs_n,target)
score = model.score(inputs_n,target)

print(score)


print(model.predict([[2,1,1]]))