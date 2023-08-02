import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
df = pd.read_csv('HR_comma_sep.csv')



# left = df[df.left==1]
# print(left.shape)

# retained = df[df.left==0]
# print(retained.shape)

sol=df.groupby('left').mean()
sol.to_csv('satisfaction_of_df.csv')

# pd.crosstab(df.salary,df.left).plot(kind='bar')
# plt.show()
# pd.crosstab(df.Department,df.left).plot(kind='bar')
# plt.show()

subdf = df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]


salary_dummies = pd.get_dummies(subdf.salary,prefix='salary')
df_with_dummies = pd.concat([subdf,salary_dummies],axis='columns')

df_with_dummies.drop('salary',axis='columns',inplace=True)

#print(df_with_dummies)
x = df_with_dummies
y=df.left
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.7)

model = LogisticRegression()
print(x_test)
model.fit(x_train,y_train)
print(model.predict(x_test))

print(model.score(x_test,y_test))
