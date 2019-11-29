import sklearn as sk
import numpy as np 
from sklearn.neural_network import MLPClassifier

features = [[1,0,0],[1,1,0],[1,1,1],[1,0,1],[5,0,0],[0,5,0],[0,5,5],[0,5,0]]
labels = ["Mo","Mo","Lo","Mo","Tr","Xr","Tr","Tr"]

model = MLPClassifier()
model.fit(np.array(features),labels)
x = [0,0,0]

result = model.predict(np.array(x).reshape(1, -1))
score = model.predict_proba(np.array(x).reshape(1, -1))
print(result[0],len(score[0]))