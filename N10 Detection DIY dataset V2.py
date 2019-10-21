from sklearn.datasets import fetch_openml
from skimage import feature
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import cv2 as cv 

#########################----Numbers----#############################
path = "/home/mohsencactus/Opencv/HW5/Q1/"
features = []
numbers = []
labels = []
for i in range (10):
    number = cv.imread(path + str(i) + '.PNG')
    number = cv.resize(number,(28,28))
    numbers.append(number)

#########################----Data----################################
    vfeature = feature.hog(number)
    labels.append(str(i))
    features.append(vfeature)

#########################----MCL Start----###########################
#model = KNeighborsClassifier(n_neighbors=1)
model = LinearSVC()
model.fit(np.array(features),labels)

#########################----Multi Target----########################
frame = cv.imread("/home/mohsencactus/Opencv/HW5/digit2.jpg")
grayed = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
binaried = cv.inRange(grayed, 0, 90)
contours = cv.findContours(binaried,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)[1]
if len(contours) > 0:
    for cnt in contours:
        x,y,w,h = cv.boundingRect(cnt)
        x,y,w,h = x-20,y-20,w+40,h+40
        target = frame[y:y+h,x:x+w]
        target = cv.resize(target,(28,28))        
        tfeat = feature.hog(target)
        result = model.predict(np.array([tfeat]))
        cv.putText(frame,str(result[0]),(x,y-20),cv.FONT_HERSHEY_DUPLEX,1.0,(255, 0, 0), 1)

#########################----Single Target----#######################
#target = cv.imread("/home/mohsencactus/Opencv/HW5/digit2.jpg")
#target = cv.resize(target,(28,28))
#tfeat = feature.hog(target)
#result = model.predict(np.array([tfeat]))
#print(result)

#########################----Result----##############################
cv.imshow("target",frame)
cv.waitKey(0)