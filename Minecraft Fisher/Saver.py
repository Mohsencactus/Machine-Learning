import numpy as np
import cv2 as cv
from PIL import ImageGrab
from time import sleep

#def nthn(X):
#    pass

ye = 1080*(3/7)
xe = 1920*(7/8)
xs = 1920*(5/8)
ys = 1080*(2/7)

saver = True
counter = 12
cv.namedWindow("window",cv.WINDOW_NORMAL)

#cv.createTrackbar("Hmin","window",0,255,nthn)
#cv.createTrackbar("Hmax","window",0,180,nthn)
#cv.createTrackbar("Smin","window",0,255,nthn)
#cv.createTrackbar("Smax","window",0,255,nthn)
#cv.createTrackbar("Vmin","window",0,255,nthn)
#cv.createTrackbar("Vmax","window",0,255,nthn)
#cv.setTrackbarPos("Hmax","window",180)
#cv.setTrackbarPos("Smax","window",255)
#cv.setTrackbarPos("Vmax","window",255)
mins = np.array([0,200,0])
maxs = np.array([10,255,255])

while True:
    screen = np.array(ImageGrab.grab(bbox=(int(xs),int(ys),int(xe),int(ye))))
    frame = cv.cvtColor(screen,cv.COLOR_BGR2RGB)
    print(frame.shape)
    hsv = cv.cvtColor(screen,cv.COLOR_BGR2HSV)

    #Hmin = cv.getTrackbarPos("Hmin","window")
    #Smin = cv.getTrackbarPos("Smin","window")
    #Vmin = cv.getTrackbarPos("Vmin","window")
    #Hmax = cv.getTrackbarPos("Hmax","window")
    #Smax = cv.getTrackbarPos("Smax","window")
    #Vmax = cv.getTrackbarPos("Vmax","window")
    
    #mins = np.array([Hmin,Smin,Vmin])
    #maxs = np.array([Hmax,Smax,Vmax])

    binaried = cv.inRange(hsv,mins,maxs)
    filtered = cv.bitwise_and(frame,frame,mask = binaried)

    if saver == True:
        counter = counter + 1
        label = "N" + str(counter)
        filename = "pics/" + label + '.jpg'
        cv.imwrite(filename,frame)

    cv.imshow('window',frame)
    if cv.waitKey(240) == ord('q'):
        cv.destroyAllWindows()
        break
