from PIL import ImageGrab
import cv2 as cv
import numpy as np
from skimage import feature
from sklearn.externals import joblib
import os
from time import sleep,time
import win32api, win32con

def click():
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP,0,0)
    
def percent(X):
    percent = int(X*1000)/10
    return percent 

ye = 1080*(3/7)
xe = 1920*(7/8)
xs = 1920*(5/8)
ys = 1080*(2/7)

Ncheck = 8
avgscore = 97
SX = 40
SY = 40
scorelist = []
checker = []

cv.namedWindow("window",cv.WINDOW_NORMAL)
model = joblib.load('Fish.pkl') 

mins = np.array([0,200,0])
maxs = np.array([10,255,255])
sleep(3)
while True:
    sttime = time()
    click()
    caught = False
    while ((time() - sttime) < 45):
        screen = np.array(ImageGrab.grab(bbox=(int(xs),int(ys),int(xe),int(ye))))

        frame = cv.cvtColor(screen,cv.COLOR_BGR2RGB)

        hsv = cv.cvtColor(frame,cv.COLOR_RGB2HSV)
        binaried = cv.inRange(hsv,mins,maxs)
        filtered = cv.bitwise_and(frame,frame,mask = binaried)

        img = cv.resize(filtered,(SX,SY))
        Fvec = feature.hog(img) 

        score = model.predict_proba(np.array(Fvec.reshape(1, -1)))
        for smt in score[0]:
            scorelist.append(percent(smt))
        if max(scorelist) < avgscore:
            print("Unknown: ",max(scorelist))    
        else:
            result = model.predict(np.array(Fvec).reshape(1, -1))
            if result[0] == 'n':
                checker = []
            else:
                checker.append('c')
                if len(checker) >= Ncheck:
                    print("Will draw back in one sec")
                    sleep(1.9)
                    click()
                    caught = True
                    break

            print(result[0])

        cv.imshow('window',filtered)
        if cv.waitKey(1) == ord('q'):
            cv.destroyAllWindows()
            break
    if caught != True:
        click()
