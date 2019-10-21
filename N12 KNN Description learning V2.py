import numpy as np 
import cv2 as cv 
from sklearn.neighbors import KNeighborsClassifier
########################################################################
features = np.array([[1,3],[0,1],[1,1.5],[1,5],[0,2],[0,3.5]])
labels = ["yes","no","yes","yes","no","no"]
########################################################################
model = KNeighborsClassifier(n_neighbors=3)
model.fit(features,labels)
########################################################################
result = model.predict(np.array([[1,2.5]]))
print(result)