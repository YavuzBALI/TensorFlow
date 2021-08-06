import cv2
print(cv2.__version__)
import numpy as np
import time
#from tensorflow.keras.preprocessing import image
#import tensorflow as tf

def nothing(x):
    pass
prev_frame_time=0
new_frame_time=0
cv2.namedWindow('Trackbars')
cv2.moveWindow('Trackbars',1320,0)

cv2.createTrackbar('hueLower', 'Trackbars',50,179,nothing)
cv2.createTrackbar('hueUpper', 'Trackbars',100,179,nothing)

cv2.createTrackbar('hue2Lower', 'Trackbars',50,179,nothing)
cv2.createTrackbar('hue2Upper', 'Trackbars',100,179,nothing)

cv2.createTrackbar('satLow', 'Trackbars',100,255,nothing)
cv2.createTrackbar('satHigh', 'Trackbars',255,255,nothing)
cv2.createTrackbar('valLow','Trackbars',100,255,nothing)
cv2.createTrackbar('valHigh','Trackbars',255,255,nothing)


dispW=640
dispH=480
flip=2
#Uncomment These next Two Line for Pi Camera
cam= cv2.VideoCapture('goruntuler/deneme.avi')

#Or, if you have a WEB cam, uncomment the next line
#(If it does not work, try setting to '1' instead of '0')
#cam=cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    #frame=cv2.imread('smarties.png')

    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    hueLow=cv2.getTrackbarPos('hueLower', 'Trackbars')
    hueUp=cv2.getTrackbarPos('hueUpper', 'Trackbars')

    hue2Low=cv2.getTrackbarPos('hue2Lower', 'Trackbars')
    hue2Up=cv2.getTrackbarPos('hue2Upper', 'Trackbars')

    Ls=cv2.getTrackbarPos('satLow', 'Trackbars')
    Us=cv2.getTrackbarPos('satHigh', 'Trackbars')

    Lv=cv2.getTrackbarPos('valLow', 'Trackbars')
    Uv=cv2.getTrackbarPos('valHigh', 'Trackbars')

    l_b=np.array([hueLow,Ls,Lv])
    u_b=np.array([hueUp,Us,Uv])

    l_b2=np.array([hue2Low,Ls,Lv])
    u_b2=np.array([hue2Up,Us,Uv])

    FGmask=cv2.inRange(hsv,l_b,u_b)
    FGmask2=cv2.inRange(hsv,l_b2,u_b2)
    FGmaskComp=cv2.add(FGmask,FGmask2)

    FG=cv2.bitwise_and(frame, frame ,mask=FGmaskComp)

    contours,hierarchy=cv2.findContours(FGmaskComp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours=sorted(contours,key=lambda x:cv2.contourArea(x),reverse=True)
    for cnt in contours:
        area=cv2.contourArea(cnt)
        (x,y,w,h)=cv2.boundingRect(cnt)
        if area>=50:
            #cv2.drawContours(frame,[cnt],0,(255,0,0),3)
            cv2.rectangle(frame,(x-30,y-30),(x+w+30,y+h+30),(255,0,0),3)
            '''
            Robot=frame[(y-30):(y+h+30),(x-30):(x+w+30)]
            Robot_pre=cv2.resize(Robot,(101,101))
            Robot_pre=Robot_pre/255
            X = image.img_to_array(Robot_pre)
            X = np.expand_dims(X,axis=0)
            #images = np.vstack([X])
            image_input=tf.constant(X.astype('float32'))
            print(image_input)
            '''
    new_frame_time=time.time()
    second=new_frame_time-prev_frame_time
    fps=1/(new_frame_time-prev_frame_time)
    prev_frame_time=new_frame_time    
    fps=int(fps)
    fps_1=str(fps)
    second_1=str(round(second,3))
    cv2.putText(frame,fps_1,(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.putText(frame,second_1,(10,45),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.imshow('xx',frame)
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()