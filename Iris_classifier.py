from sklearn import datasets
from sklearn import metrics
from sklearn.svm import SVC
ds =datasets.load_iris()
model= SVC()
model.fit(ds.data, ds.target)
print(model)
expected = ds.target
predicted = model.predict(ds.data)
print(metrics.classification_report(expected, predicted))