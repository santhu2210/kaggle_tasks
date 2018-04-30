import pandas as pd 
from sklearn.naive_bayes import GaussianNB

dfTrain = pd.read_csv('train.csv')
dfTest = pd.read_csv('test.csv')

dfTrain["CAge"] = pd.cut(dfTrain["Age"], bins=[0,10,18,40,max(dfTrain["Age"])], labels=["Child", "Myoung", "Young","Older"])
dfTest["CAge"] = pd.cut(dfTest["Age"], bins=[0,10,18,40,max(dfTest["Age"])], labels=["Child", "Myoung", "Young","Older"])


dfTrain = pd.get_dummies(data=dfTrain, dummy_na=True, prefix=["Pclass","Sex","Embarked","Age"], columns=["Pclass", "Sex","Embarked","CAge"])
dfTest = pd.get_dummies(data=dfTest, dummy_na=True, prefix=["Pclass","Sex","Embarked","Age"], columns=["Pclass", "Sex","Embarked","CAge"])

# print dfTrain
# print "&&&&&&&&&&&&&&&&&&&&&&&&&&"

# print dfTest

Y_train = dfTrain["Survived"]

submission = pd.DataFrame()
submission["PassengerId"]=dfTest["PassengerId"]


dfTrain = dfTrain[dfTrain.columns.difference(["Age","Survived","PassengerId","Name","Ticket","Cabin"])]


dfTest = dfTest[dfTest.columns.difference(["Age","PassengerId","Name","Ticket","Cabin"])]


dfTest["Fare"].iloc[dfTest[dfTest["Fare"].isnull()].index] = dfTest[dfTest["Pclass_3.0"]==1]["Fare"].median()

# print dfTest["Fare"][145:155]

clf = GaussianNB()

clf.fit(dfTrain,Y_train)

prd_cl = clf.predict(dfTest)

print prd_cl

pred = pd.DataFrame(prd_cl, columns=["Survived"])

submission = submission.join(pred, how="inner")

submission.to_csv("submit_data.csv", index=False)

# submission.head(18)


