import cv2
import numpy as np
print(cv2.__version__)

def nothing(x):
    pass

cv2.namedWindow('Trackbars')
cv2.moveWindow('Trackbars',0,0)
cv2.createTrackbar('hueLower','Trackbars',50,179,nothing)
cv2.createTrackbar('hueHigher','Trackbars',100,179,nothing)
cv2.createTrackbar('satLow','Trackbars',100,255,nothing)
cv2.createTrackbar('satHigh','Trackbars',255,255,nothing)
cv2.createTrackbar('valLow','Trackbars',100,255,nothing)
cv2.createTrackbar('valHigh','Trackbars',255,255,nothing)




#cam=cv2.VideoCapture(0)
while True:
    #ret, frame = cam.read()
    #cv2.imshow('nanoCam',frame)
    frame=cv2.imread('goruntuler/Red_Robot.jpeg')
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    hueLow=cv2.getTrackbarPos('hueLower', 'Trackbars')
    hueUp=cv2.getTrackbarPos('hueHigher', 'Trackbars')
    
    Ls=cv2.getTrackbarPos('satLow', 'Trackbars')
    Us=cv2.getTrackbarPos('satHigh', 'Trackbars')

    Lv=cv2.getTrackbarPos('valLow', 'Trackbars')
    Uv=cv2.getTrackbarPos('valHigh','Trackbars')
    
    l_b=np.array([hueLow,Ls,Lv])
    u_b=np.array([hueUp,Us,Uv])

    FGmask=cv2.inRange(hsv,l_b,u_b)
    cv2.imshow('FGmask',FGmask)

    contours, hierarchy = cv2.findContours(FGmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #contours=cv2.findContours(FGmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(frame,contours,-1, (0,255,0), 3)
    (x,y,w,h)=cv2.boundingRect(contours)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('Smarties',frame)
    if cv2.waitKey(1)==ord('q'):
        break
#cam.release()
#cv2.destroyAllWindows()

"""Kirmizi icin:
    hueLower: 157
    hueHigher: 171
    satLow:74
    satHigh:159
    valLow:238
    valHigh:255"""