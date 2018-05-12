import pandas as pd
import numpy as np
train  = pd.read_csv("~/Adult/train.csv")
test = pd.read_csv("~/Adult/test.csv")
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score

train.info()
print("The train data has :", train.shape)
print("The test data has :",test.shape)
train.head()
nans = train.shape[0] - train.dropna().shape[0]
print("%d No. of ros having missing values are :"%nans)

train.isnull().sum()

null_columns=train.columns[train.isnull().any()]
train[null_columns].isnull().sum()

cat = train.select_dtypes(include=['O'])
cat.apply(pd.Series.nunique)

train.workclass.value_counts(sort = True)
train.workclass.fillna('private',inplace=True)

train.occupation.value_counts(sort=True)
train.occupation.fillna('prof_speciality',inplace=True)

train['native.country'].value_counts(sort=True)
train['native.country'].fillna('prof_speciality',inplace=True)

train.isnull().sum()

y = train['target']
del train['target']

X = train
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)

#train the RF classifier
clf = RandomForestClassifier(n_estimators = 500, max_depth = 6)
clf.fit(X_train,y_train)

    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=6, max_features='auto', max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                n_estimators=500, n_jobs=1, oob_score=False, random_state=None,
                verbose=0, warm_start=False)

clf.predict(X_test)
prediction = clf.predict(X_test)
acc =  accuracy_score(np.array(y_test),prediction)
print ('The accuracy of Random Forest is {}'.format(acc))