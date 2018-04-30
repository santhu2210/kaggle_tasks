#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 16:25:33 2018

@author: shanthakumarp
"""

import pandas as pd
import matplotlib.pyplot as plt

#dataset = pd.read_csv('train.csv')
#y = dataset.iloc[:, 1].values
#X = dataset.iloc[:, [2,4,5,6,7,9,11]].values

from sklearn import metrics

dfTrain = pd.read_csv('train.csv')
dfTest = pd.read_csv('test.csv')
ztest = pd.read_csv('gender_submission.csv')

submission = pd.DataFrame()
submission["PassengerId"]=dfTest["PassengerId"]

dfTrain["CAge"] = pd.cut(dfTrain["Age"], bins=[0,10,18,40,max(dfTrain["Age"])], labels=["Child", "Myoung", "Young","Older"])
dfTest["CAge"] = pd.cut(dfTest["Age"], bins=[0,10,18,40,max(dfTest["Age"])], labels=["Child", "Myoung", "Young","Older"])


dfTrain = pd.get_dummies(data=dfTrain, dummy_na=True, prefix=["Pclass","Sex","Embarked","Age"], columns=["Pclass", "Sex","Embarked","CAge"])
dfTest = pd.get_dummies(data=dfTest, dummy_na=True,  columns=["Pclass", "Sex","Embarked","CAge"])

Y_train = dfTrain["Survived"]

dfTrain = dfTrain[dfTrain.columns.difference(["Age","Survived","PassengerId","Name","Ticket","Cabin"])]

dfTest = dfTest[dfTest.columns.difference(["Age","PassengerId","Name","Ticket","Cabin"])]

X = dfTrain.iloc[:].values
y = Y_train.iloc[:].values

Z = dfTest.iloc[:].values
z_test = ztest.iloc[:,1].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis=0 )
imputer = imputer.fit(X[:, 9:10])
X[:, 9:10 ] = imputer.transform(X[:, 9:10 ])

imputer2 = imputer.fit(Z[:, 9:10])
Z[:, 9:10 ] = imputer2.transform(Z[:, 9:10 ])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

## Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

## Fitting K-NN to the Training set
#from sklearn.neighbors import KNeighborsClassifier
#classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)

## Fitting Naive Bayes to the Training set
#from sklearn.naive_bayes import GaussianNB
#classifier = GaussianNB()

#from sklearn.ensemble import RandomForestClassifier
#classifier = RandomForestClassifier(n_estimators = 30, max_features = "auto", criterion = 'gini', random_state = 0)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

z_pred = classifier.predict(Z)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)               # for local checking with train data

cm_org = confusion_matrix(z_test, z_pred)           # checking & verifing with original submitted data getting 95.4%

classifier.score(X_test, y_test)

Zpred = pd.DataFrame(z_pred, columns=["Survived"])

submission = submission.join(Zpred, how="inner")

submission.to_csv("submit_data_LogReg.csv", index=False)

