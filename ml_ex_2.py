import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import linear_model as lm
from word2number import w2n


df = pd.read_csv("hiring.csv")


df.experience = df.experience.fillna("zero")

df.experience = df.experience.apply(w2n.word_to_num)
df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(df['test_score(out of 10)'].mean())


model = lm.LinearRegression()
model.fit(df[['experience','test_score(out of 10)','interview_score(out of 10)']],df['salary($)'])

experience = int(input("Enter your experience: "))
testscore = int(input("Enter your test score (out of 10): "))
interview = int(input("Enter your interview score (out of 10): "))


P_salary = model.predict([[experience,testscore,interview]])

print(f"Your salary will be: {P_salary}")