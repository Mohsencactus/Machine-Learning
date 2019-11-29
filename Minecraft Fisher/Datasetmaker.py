import cv2 as cv
import numpy as np
from skimage import feature
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import os

def clearer(path):
    label = path.replace('.jpg','')
    label = label.replace('pics/','')
    label = ''.join([i for i in label if not i.isdigit()])
    label = label.lower()
    return label

Path = []
SX = 40
SY = 40
Labels = []
Features = []

for filename in os.listdir("pics"):
    Path.append("pics/"+filename)

try:
    a = 0/1
    b = 1/0
#    model = joblib.load('saved_model.pkl') 
except:
    for path in Path:
        img = cv.imread(path)

        hsv = cv.cvtColor(img,cv.COLOR_RGB2HSV)
        mins = np.array([0,200,0])
        maxs = np.array([10,255,255])
        binaried = cv.inRange(hsv,mins,maxs)
        filtered = cv.bitwise_and(img,img,mask = binaried)

        img2 = cv.resize(filtered,(SX,SY))
        Fvec = feature.hog(img2) 
        Features.append(Fvec)
        label = clearer(path)
        Labels.append(label)
        print(path)
        cv.imshow('window',filtered)
        cv.waitKey(0)

model = MLPClassifier()
model.fit(np.array(Features),Labels)
joblib.dump(model, 'Fish.pkl') 

cv.destroyAllWindows()
