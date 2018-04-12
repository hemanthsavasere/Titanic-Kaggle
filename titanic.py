#importing nesscary packages

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def read_dataset(test, train):
    train = pd.read_csv(train)
    test = pd.read_csv(test)
    return test, train

def encode(columns):
    columns = LabelEncoder().fit_transform(columns)
    return columns

# laoding testing and training dataset
test, train = read_dataset("test.csv", "train.csv")

# filling the missing data with mean of age
train["Age"] = train["Age"].fillna(train["Age"].mean())
#encoding the data
train['Sex'] = encode(train['Sex'])
# filling the missing data
train["Embarked"] = train["Embarked"].fillna("S")
#encoding the data 
train['Embarked'] = encode(train['Embarked'])

#creating a new feature family size that is sum of Parents and Siblings and himself
train["Family_size"] = train["Parch"] + train["SibSp"] + 1
features = train[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","Family_size"]].values

# using standard scaler to scale the data 
features = StandardScaler().fit_transform(features)
target = train[["Survived"]].values
print (features)

# using random froest classifier as Model
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 1000, random_state = 1)
my_forest = forest.fit(features,target)
print(my_forest.score(features, target))


# filling missing values and encoding the data
test["Embarked"] = encode(test["Embarked"])
test["Sex"] = encode(test["Sex"])
test["Family_size"] = test["Parch"] + test["SibSp"] + 1
test["Age"] = test["Age"].fillna(test["Age"].mean())
test["Fare"] = test["Fare"].fillna(test["Fare"].median())


test_features = test[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","Family_size"]].values  
# using standard scaler for test features
test_features = StandardScaler().fit_transform(test_features)
prediction = my_forest.predict(test_features)
print (len(prediction))

# saving the predection as CSV
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(prediction, PassengerId, columns = ["Survived"])
print(my_solution)
my_solution.to_csv("submission.csv")
