import pandas as pd 
from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn import metrics

dfTrain = pd.read_csv('train.csv')
dfTest = pd.read_csv('test.csv')

dfTrain["CAge"] = pd.cut(dfTrain["Age"], bins=[0,10,18,40,max(dfTrain["Age"])], labels=["Child", "Myoung", "Young","Older"])
dfTest["CAge"] = pd.cut(dfTest["Age"], bins=[0,10,18,40,max(dfTest["Age"])], labels=["Child", "Myoung", "Young","Older"])


dfTrain = pd.get_dummies(data=dfTrain, dummy_na=True, prefix=["Pclass","Sex","Embarked","Age"], columns=["Pclass", "Sex","Embarked","CAge"])
dfTest = pd.get_dummies(data=dfTest, dummy_na=True, prefix=["Pclass","Sex","Embarked","Age"], columns=["Pclass", "Sex","Embarked","CAge"])

# print dfTrain
# print dfTest

Y_train = dfTrain["Survived"]

submission = pd.DataFrame()
submission["PassengerId"]=dfTest["PassengerId"]


dfTrain = dfTrain[dfTrain.columns.difference(["Age","Survived","PassengerId","Name","Ticket","Cabin"])]


dfTest = dfTest[dfTest.columns.difference(["Age","PassengerId","Name","Ticket","Cabin"])]


dfTest["Fare"].iloc[dfTest[dfTest["Fare"].isnull()].index] = dfTest[dfTest["Pclass_3.0"]==1]["Fare"].median()

# print dfTest["Fare"][145:155]

model = LogisticRegression()

# model = LinearRegression()

model.fit(dfTrain,Y_train)

prediciton = model.predict(dfTest)



dfTarget = pd.read_csv('gender_submission.csv')

target =  dfTarget['Survived'].values


pred = pd.DataFrame(prediciton, columns=["Survived"])

submission = submission.join(pred, how="inner")

submission.to_csv("submit_data_logistics.csv", index=False)


# print target, "\n", prd_cl

print(metrics.classification_report(target, prediciton)) , "\n"

print(metrics.confusion_matrix(target,prediciton)), "\n"

print(metrics.accuracy_score(target,prediciton)), "\n"   # finding accuracy

# print pd.value_counts(target, sort=False), "\n"			finding each values count

# print pd.value_counts(prediciton, sort=False), "\n"