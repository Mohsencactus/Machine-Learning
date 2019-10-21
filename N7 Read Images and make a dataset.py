#########################---- ----###########################
from sklearn.datasets import fetch_openml
from skimage import feature
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import cv2 as cv 
import os

#########################----Listing Files----###########################
files = []
path = '/home/mohsencactus/Opencv/HW5/Q1'
for filename in os.listdir(path):
   files.append(filename)

#########################----Reading Files----###########################
images = []
labels = []
features = []
for i in range (len(files)):
    filename = files[i]
    labels.append(filename.replace('.PNG',''))
    source = cv.imread(path + '/' + filename)
    source = cv.resize(source,(28,28))
    vfeature = feature.hog(source)
    images.append(source)
    features.append(vfeature)

#########################----MCL Start----###########################
#model = KNeighborsClassifier(n_neighbors=1)
model = LinearSVC()
model.fit(np.array(features),labels)
