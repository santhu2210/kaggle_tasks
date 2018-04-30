#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 14:36:38 2018

@author: shanthakumarp
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dfTest  = pd.read_csv("test.csv")
dfTrain = pd.read_csv("train.csv")

submission = pd.DataFrame()
submission['Id']=dfTest['Id']

#submission2 = pd.DataFrame()
#submission2['Id']=dfTest['Id']

Y_train = dfTrain['SalePrice']

# feature engineering at training data
dfTrain['cYearBuilt'] = pd.cut(dfTrain['YearBuilt'], bins = [1870,1900,1930,1960,1990,2011], labels=['VyOld','Old','Intern','Modern','New'])

dfTrain['cYearRemodAdd'] = pd.cut(dfTrain['YearRemodAdd'], bins = [1949,1970,1990,2011], labels=['Intern','Modern','New'])

dfTrain['cGarageYrBlt'] = pd.cut(dfTrain['GarageYrBlt'], bins = [1890,1930,1960,1990,2011], labels=['Old','Intern','Modern','New'])

dfTrain = pd.get_dummies(data=dfTrain, dummy_na=False, columns=["YrSold"])

dfTrain = pd.get_dummies(data=dfTrain, dummy_na=True, prefix=["YearBuilt","YearRemodAdd",'GarageYrBlt'], columns=["cYearBuilt","cYearRemodAdd",'cGarageYrBlt'])

dfTrain = pd.get_dummies(data=dfTrain, dummy_na=True, prefix=["MSZoning","Street","Alley","LotShape","LandContour","Utilities","LotConfig","LandSlope","Neighborhood","Condition1","Condition2","BldgType","HouseStyle","RoofStyle","RoofMatl","Exterior1st","Exterior2nd","MasVnrType","ExterQual","ExterCond","Foundation","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","Heating","HeatingQC","CentralAir","Electrical","KitchenQual","Functional","FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond","PavedDrive","PoolQC","Fence","MiscFeature","SaleType","SaleCondition"], columns=["MSZoning","Street","Alley","LotShape","LandContour","Utilities","LotConfig","LandSlope","Neighborhood","Condition1","Condition2","BldgType","HouseStyle","RoofStyle","RoofMatl","Exterior1st","Exterior2nd","MasVnrType","ExterQual","ExterCond","Foundation","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","Heating","HeatingQC","CentralAir","Electrical","KitchenQual","Functional","FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond","PavedDrive","PoolQC","Fence","MiscFeature","SaleType","SaleCondition"])

dfTrain = dfTrain[dfTrain.columns.difference(["Id","YearBuilt","YearRemodAdd","GarageYrBlt","SalePrice"])]


# feature enggineering at testing data
dfTest['cYearBuilt'] = pd.cut(dfTest['YearBuilt'], bins = [1870,1900,1930,1960,1990,2011], labels=['VyOld','Old','Intern','Modern','New'])

dfTest['cYearRemodAdd'] = pd.cut(dfTest['YearRemodAdd'], bins = [1949,1970,1990,2011], labels=['Intern','Modern','New'])

dfTest['cGarageYrBlt'] = pd.cut(dfTest['GarageYrBlt'], bins = [1890,1930,1960,1990,2011], labels=['Old','Intern','Modern','New'])

dfTest = pd.get_dummies(data=dfTest, dummy_na=False, columns=["YrSold"])

dfTest = pd.get_dummies(data=dfTest, dummy_na=True, prefix=["YearBuilt","YearRemodAdd",'GarageYrBlt'], columns=["cYearBuilt","cYearRemodAdd",'cGarageYrBlt'])

dfTest = pd.get_dummies(data=dfTest, dummy_na=True, prefix=["MSZoning","Street","Alley","LotShape","LandContour","Utilities","LotConfig","LandSlope","Neighborhood","Condition1","Condition2","BldgType","HouseStyle","RoofStyle","RoofMatl","Exterior1st","Exterior2nd","MasVnrType","ExterQual","ExterCond","Foundation","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","Heating","HeatingQC","CentralAir","Electrical","KitchenQual","Functional","FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond","PavedDrive","PoolQC","Fence","MiscFeature","SaleType","SaleCondition"], columns=["MSZoning","Street","Alley","LotShape","LandContour","Utilities","LotConfig","LandSlope","Neighborhood","Condition1","Condition2","BldgType","HouseStyle","RoofStyle","RoofMatl","Exterior1st","Exterior2nd","MasVnrType","ExterQual","ExterCond","Foundation","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","Heating","HeatingQC","CentralAir","Electrical","KitchenQual","Functional","FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond","PavedDrive","PoolQC","Fence","MiscFeature","SaleType","SaleCondition"])

dfTest = dfTest[dfTest.columns.difference(["Id","YearBuilt","YearRemodAdd","GarageYrBlt"])]


# remove unwanted columns in training dataset & testing dataset
dfTrain = dfTrain[dfTrain.columns.difference(["Exterior1st_ImStucc","Exterior1st_Stone","Electrical_Mix","Condition2_RRNn","Condition2_RRAn","Condition2_RRAe","Utilities_AllPub","Utilities_NoSeWa","RoofMatl_ClyTile","PoolQC_Fa","RoofMatl_Membran","RoofMatl_Metal","RoofMatl_Roll","MiscFeature_TenC","HouseStyle_2.5Fin","Heating_Floor","Heating_OthW","GarageQual_Ex","Exterior2nd_Other"])]
dfTest = dfTest[dfTest.columns.difference(["Utilities_AllPub"])]

# assign x,y , z form dataset
X = dfTrain.iloc[:].values
y = Y_train.iloc[:].values

Z = dfTest.iloc[:].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis=0 )
imputer = imputer.fit(X[:])
X[:] = imputer.transform(X[:])

Zimputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis=0 )
Zimputer = Zimputer.fit(Z[:])
Z[:] = Zimputer.transform(Z[:])

#spliting training and testing data
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0, test_size=0.25)

## Fitting multiple linear regression to training set
#from sklearn.linear_model import LinearRegression
#regressor = LinearRegression()
#regressor.fit(X_train, y_train)
#
##predict test set using training regressor model
#y_pred_lr = regressor.predict(X_test)

# Fitting random forest predictor model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=400, random_state=0)
regressor.fit(X_train, y_train)

# Predicting a new result
y_pred_rf = regressor.predict(X_test)

# testing & prediction on test data using RF
regressor.fit(X,y)
new_test_reg_pred = regressor.predict(Z)

# Fitting XGBoost to the Training set
import xgboost
xgb = xgboost.XGBRegressor(n_estimators=500, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
xgb.fit(X_train, y_train)

# Predicting the new results
y_pred_xg = xgb.predict(X_test)


# Test data fitted using XGboost
xgb.fit(X, y)
new_test_XG_pred = xgb.predict(Z)

from sklearn.metrics import explained_variance_score

xg_score = explained_variance_score(y_pred_xg,y_test) # getting high score
rf_score = explained_variance_score(y_pred_rf,y_test) # getting low score than xg


XGBpred = pd.DataFrame(new_test_XG_pred, columns=["SalePrice"])

submission = submission.join(XGBpred, how="inner")

submission.to_csv("submit_data_XGB4.csv", index=False)


#RFpred = pd.DataFrame(new_test_reg_pred, columns=["SalePrice"])
#
#submission2 = submission2.join(RFpred, how="inner")
#
#submission2.to_csv("submit_data_RF.csv", index=False)


#from sklearn import metrics
#from math import sqrt

#root_mean_sqr_err = sqrt(metrics.mean_squared_error(y_test, y_pred_xg))
#mean_abs_err = metrics.mean_absolute_error(y_test, y_pred_xg)
#mean_sqr_err = metrics.mean_squared_error(y_test, y_pred_xg)

## Make an optimal model using backward elimination
#import statsmodels.formula.api as sm
##X_try = np.append(arr = np.ones((50,1)).astype(int), values=X, axis=0)
#X2 = np.append(arr = np.ones((1460,328)).astype(int), values=X, axis=1)
#X_opt = X2[:, [0,1,2,3,4,5,6,7,8,9,10,11,13,14,15]]
#regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#regressor_OLS.summary()
