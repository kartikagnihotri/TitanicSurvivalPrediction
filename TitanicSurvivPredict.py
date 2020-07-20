# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 18:52:53 2020

@author: kartik
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train=pd.read_csv('C:/Users/kartik/Desktop/projects/TitanicSurvival/train.csv')
test=pd.read_csv('C:/Users/kartik/Desktop/projects/TitanicSurvival/test.csv')

columns=['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

#Graph showing survival depending on gender of passengers
sex_pivot = train.pivot_table(index="Sex",values="Survived")
sex_pivot.plot.bar()
plt.show()

#Graph showing survival depending on Passenger Class
class_pivot = train.pivot_table(index="Pclass",values="Survived")
class_pivot.plot.bar()
plt.show()

#Histograms to show those that survived vs those who died across different age ranges
survived = train[train["Survived"] == 1]
died = train[train["Survived"] == 0]
survived["Age"].plot.hist(alpha=0.5,color='red',bins=50)
died["Age"].plot.hist(alpha=0.5,color='blue',bins=50)
plt.legend(['Survived','Died'])
plt.show()



#encoding categorical variables
train['Sex']=train['Sex'].map({'male':0,'female':1})
test['Sex']=test['Sex'].map({'male':0,'female':1})

train['Embarked']=train['Embarked'].map({'S':1,'C':2,'Q':3})
test['Embarked']=test['Embarked'].map({'S':1,'C':2,'Q':3})

#impute missing values
train['Age'].fillna(train['Age'].mean(),inplace=True)
test['Age'].fillna(test['Age'].mean(),inplace=True)
train['Embarked'].fillna(train['Embarked'].mean(),inplace=True)
test['Embarked'].fillna(test['Embarked'].mean(),inplace=True)

holdout=test
X_columns=['Pclass','Sex','SibSp','Parch','Embarked']
from sklearn.model_selection import train_test_split
all_X=train[X_columns]
all_Y=train['Survived']

train_X,test_X,train_Y,test_Y=train_test_split(all_X,all_Y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
lr.fit(train_X,train_Y)
predictions=lr.predict(test_X)

from sklearn.metrics import accuracy_score
print(accuracy_score(test_Y,predictions))

#Using Cross-Validation to check error metrics we are getting from your model are accurate.
from sklearn.model_selection import cross_val_score

lr = LogisticRegression()
scores = cross_val_score(lr, all_X, all_Y, cv=10)
scores.sort()
accuracy = scores.mean()

print(scores)
print(accuracy)



