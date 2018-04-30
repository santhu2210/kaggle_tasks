#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 14:44:58 2018

@author: shanthakumarp
"""

import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('train.csv')
y = dataset.iloc[:, 1].values
X = dataset.iloc[:, [2,4,5,6,7,9,11]].values
#X = dataset.iloc[:,2:10 ].values

#test_data_set
t_dataset = pd.read_csv('test.csv')
a_dataset = pd.read_csv('gender_submission.csv')
y_ans_test = a_dataset.iloc[:, 1].values
X_ans_test = t_dataset.iloc[:, [1,3,4,5,6,8,10]].values

ex_dataset = pd.read_csv('submit_data_NB.csv')
ex_y = ex_dataset.iloc[:,1].values

# Taking care of missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis=0 )
imputer = imputer.fit(X[:, 2:3])
X[:, 2:3] = imputer.transform(X[:, 2:3])


# Taking care of missing values of ans_test
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis=0 )
imputer = imputer.fit(X_ans_test[:, [2,5]])
X_ans_test[:, [2,5]] = imputer.transform(X_ans_test[:, [2,5]])

#Encoding Categorial data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_1 = LabelEncoder()
X[:, 6] = labelencoder_x_1.fit_transform(X[:, 6])
X[:, 1] = labelencoder_x_1.fit_transform(X[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

#Avoid Dummy variable
X = X[:, 1:]

#Encoding Categorial data of ans_test
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_2 = LabelEncoder()
X_ans_test[:, 6] = labelencoder_x_2.fit_transform(X_ans_test[:, 6])
X_ans_test[:, 1] = labelencoder_x_2.fit_transform(X_ans_test[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X_ans_test = onehotencoder.fit_transform(X_ans_test).toarray()

#Avoid Dummy variable of ans_test
X_ans_test = X_ans_test[:, 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#feature scalling for ans_test
X_ans_test = sc.fit_transform(X_ans_test)


## Fitting K-NN to the Training set
#from sklearn.neighbors import KNeighborsClassifier
#classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

## Fitting Logistic Regression to the Training set
#from sklearn.linear_model import LogisticRegression
#classifier = LogisticRegression(random_state = 0)

## Fitting Naive Bayes to the Training set
#from sklearn.naive_bayes import GaussianNB
#classifier = GaussianNB()

## Fitting SVM to the Training set
#from sklearn.svm import SVC
#classifier = SVC(kernel = 'rbf', random_state = 0)

## Fitting Decision Tree Classification to the Training set
#from sklearn.tree import DecisionTreeClassifier
#classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 30, max_features = "auto", criterion = 'gini', random_state = 0)

classifier.fit(X_train, y_train)

## Fitting XGBoost to the Training set
#from xgboost import XGBClassifier
#classifier = XGBClassifier()
#classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# predicting the ans_test set results
y_pred_ans = classifier.predict(X_ans_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


cm_ans = confusion_matrix(y_ans_test, y_pred_ans)

cm_ex = confusion_matrix(y_ans_test, ex_y)


classifier.score(X_train, y_train)
classifier.score(X_test, y_test)

classifier.score(X_ans_test, y_ans_test)
#classifier.intercept_
#classifier.coef_

## Applying k-Fold Cross Validation
#from sklearn.model_selection import cross_val_score
#accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
#accuracies.mean()
#accuracies.std()
