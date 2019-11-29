import cv2 as cv
import numpy as np
from skimage import feature
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import datetime as dt
import os

###############################
counter = 0
SX = 40
SY = 40
avgscore = 96
zrb = 0.025
saver = True
pervi = False
result = ['']
Features = []
Labels = []
Path = []

###############################
def percent(X):
    percent = int(X*1000)/10
    return percent 

def clearer(path):
    label = path.replace('.jpg','')
    label = label.replace('Face/','')
    label = ''.join([i for i in label if not i.isdigit()])
    label = label.lower()
    return label

###############################
targetcas = cv.CascadeClassifier("Face.xml")
#model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
model = MLPClassifier()
webcam = cv.VideoCapture(0)

###############################
for filename in os.listdir("Face"):
    Path.append("Face/"+filename)

#try :
#    model = joblib.load('saved_model.pkl') 
#except:
for path in Path:
    face = cv.imread(path)
    face = cv.resize(face,(SX,SY))
    Fvec = feature.hog(face) 
    Features.append(Fvec)
    label = clearer(path)
    print(label)
    Labels.append(label)
model.fit(np.array(Features),Labels)
joblib.dump(model, 'saved_model.pkl') 
###############################
while True:
    frame = webcam.read()[1]
    frame2 = frame.copy()
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    target = targetcas.detectMultiScale(gray,1.1,8)    
    #target = targetcas.detectMultiScale(gray)    
######################################################################################
    if len(target) > 0:
        scorelist = []
        faceroi = []
        for (x,y,w,h) in target:
            cv.rectangle(frame2,(x,y),(x+w,y+h),(255,0,0),2)
            roi = frame[y:y+h,x:x+w]
            faceroi.append(roi)
            roi2 = cv.resize(roi,(SX,SY))
            Fvec = feature.hog(roi2)
            score = model.predict_proba(np.array(Fvec.reshape(1, -1)))
            for smt in score[0]:
                scorelist.append(percent(smt))
            cv.putText(frame2, 'Score: '+str(max(scorelist)), (x,y+h+40), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)
            if max(scorelist) < avgscore:
                if pervi == False:
                    counter = counter + 1
                    pervi = True
                cv.putText(frame2, 'Target'+str(counter), (x,y+h+25), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 1)
                ###############################
                if saver == True:
                    cv.imshow("Frame2",frame2)
                    cv.waitKey(20)
                    label = input("Who is this? ")
                    avgscore = avgscore - zrb
                    label = clearer(label)
                    Labels.append(label)
                    Features.append(Fvec)
                    model.fit(np.array(Features),Labels)
                    date = ((dt.datetime.now().strftime("%X")).replace(':','')) + ((dt.datetime.now().strftime("%x")).replace('/',''))
                    filename = label + date +'.jpg'
                    cv.imwrite("Face/"+filename,roi)
                ###############################
            else:    
                result = model.predict(np.array(Fvec).reshape(1, -1))
                result = result[0][0].upper()+result[0][1:]
                cv.putText(frame2,result, (x,y+h+25), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 1)
    else:
        pervi = False
    cv.putText(frame2, 'Unknown: '+str(counter), (0,20), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 1)
######################################################################################
    cv.imshow("Frame2",frame2)
    if ord("q") == cv.waitKey(1):
        cv.destroyAllWindows()
        break