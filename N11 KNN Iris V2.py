import numpy as np 
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
########################################################################
data = datasets.load_iris(return_X_y=True)
features = data[0]
labels = data[1]
ftrain,ftest,ltrain,ltest = train_test_split(features,labels,test_size=0.34)
########################################################################
model = KNeighborsClassifier(n_neighbors=3)
model.fit(ftrain,ltrain)
########################################################################
tresult = model.score(ftest,ltest)
result = model.predict(ftest)
print(len(ftest)/len(features))