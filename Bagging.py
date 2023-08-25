import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('diabetes.csv')

# print(df.isnull().sum())

x = df.drop('Outcome',axis='columns')
y = df.Outcome


scalar = StandardScaler()
x_scaled = scalar.fit_transform(x)
#print(x_scaled)

x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,stratify=y,random_state=10)

scores=cross_val_score(DecisionTreeClassifier(),x,y,cv=5)
 
print(scores.mean())

bag_model = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,max_samples=0.8,
    oob_score=True,
    random_state=0
)

bag_model.fit(x_train,y_train)

scores=cross_val_score(bag_model,x,y,cv=5)
 
print(scores.mean())

scores=cross_val_score(RandomForestClassifier(n_estimators=50),x,y,cv=5)
 
print(scores.mean())