import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import os
from sklearn import  metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.stats.distributions import uniform, randint




TITANIC_PATH = os.path.join("datasets", "titanic")
def load_titanic_data(filename, titanic_path=TITANIC_PATH):
    csv_path = os.path.join(titanic_path, filename)
    return pd.read_csv(csv_path)
train_data = load_titanic_data("train.csv")
test_data = load_titanic_data("test.csv")

train_data=train_data.replace({'Sex':{'male':1,'female':0}})
med=np.median(train_data.dropna()['Age'])
train_data=train_data.replace({'Sex':{'male':1,'female':0}})
train_data['Age']=train_data['Age'].fillna(med)
train_data=train_data.assign(C=train_data['Embarked']=='C')
train_data=train_data.assign(Q=train_data['Embarked']=='Q')
train_data=train_data.assign(S=train_data['Embarked']=='S')
train_data=train_data.drop(["Name","Ticket","Cabin","Embarked"],axis=1)
X_train = train_data.drop(["Survived"],axis=1)
y_train = train_data["Survived"]

test_data=test_data.replace({'Sex':{'male':1,'female':0}})
test_data=test_data.replace({'Sex':{'male':1,'female':0}})
test_data['Age']=test_data['Age'].fillna(med)
med2=np.median(train_data.dropna()['Fare'])
test_data['Fare']=test_data['Fare'].fillna(med2)
test_data=test_data.assign(C=test_data['Embarked']=='C')
test_data=test_data.assign(Q=test_data['Embarked']=='Q')
test_data=test_data.assign(S=test_data['Embarked']=='S')
test_data=test_data.drop(["Name","Ticket","Cabin","Embarked"],axis=1)


kfold = StratifiedKFold(n_splits=5)



pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC(probability=True))])
param_grid = {
            'classifier__gamma': uniform(0.01,0.5),
            'classifier__C': uniform(0.1,5)
}
grid_1 = RandomizedSearchCV(pipe, param_grid,n_iter=5, cv=kfold, return_train_score=True,scoring="accuracy",random_state=5)
grid_1.fit(X_train, y_train)
print(grid_1.best_params_)


predictions=grid_1.predict(test_data)
res = pd.DataFrame(predictions)
res.index = test_data.index+892
res.columns = ["prediction"]
res.to_csv("submission.csv")
