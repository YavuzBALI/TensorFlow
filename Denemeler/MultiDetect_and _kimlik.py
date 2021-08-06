import cv2
print(cv2.__version__)
import numpy as np
import time
import os
prev_frame_time=0
new_frame_time=0

cam= cv2.VideoCapture('goruntuler/deneme.avi')
'''
Bura da sirasiyla kirmizi ve yesil icin HSV code parametrelerini belirliyoruz
l_b=[HougeHight_value,Saturation_HightValue,Valuehigh]
u_b=[Hougelow_value,Saturation_LowValue,ValueLow]
'''
l_b=np.array([53,111,167])#Bunlar yesil icin
u_b=np.array([107,219,241])#bunlar yesil icin
l_b2=np.array([129,111,167])#Bunlar Kirmizi icin
u_b2=np.array([179,255,255])#bunlar Kirmizi icin
while True:
    ret, frame = cam.read()
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)#maskeleme yapmak icin BGR formatindan HSV formatina du;uruyoruz
    FGmask=cv2.inRange(hsv,l_b,u_b)#Yesil icin filtre uyguluyoruz
    FGmask2=cv2.inRange(hsv,l_b2,u_b2)#kirmizi Icin filtre uyguluyoruz
    FGmaskComp=cv2.add(FGmask,FGmask2)#Filtreleri birlestiriyoruz
    
    FG=cv2.bitwise_and(frame, frame ,mask=FGmaskComp)#Burada BGR formatindaki frame ile filtreleri maskeliyoruz
    
    contours,hierarchy=cv2.findContours(FGmaskComp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours=sorted(contours,key=lambda x:cv2.contourArea(x),reverse=True)
    for cnt in contours:
        area=cv2.contourArea(cnt)
        (x,y,w,h)=cv2.boundingRect(cnt)
        if area>=50:
            #cv2.drawContours(frame,[cnt],0,(255,0,0),3)
            cv2.rectangle(frame,(x-30,y-30),(x+60,y+60),(255,0,0),3)
            color=frame[y,x]
            if color[0]>107:
                Robot_color='Green'
            else:
                Robot_color='Red'
            cv2.putText(frame,Robot_color,(x-30,y-30),cv2.FONT_HERSHEY_SIMPLEX,1,(int(color[0]),255,255),2)

    new_frame_time=time.time()
    second=new_frame_time-prev_frame_time
    fps=1/(new_frame_time-prev_frame_time)
    prev_frame_time=new_frame_time    
    fps=int(fps)
    fps_1=str(fps)
    second_1=str(round(second,4))
    cv2.putText(frame,fps_1,(10,25),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.putText(frame,second_1,(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.imshow('xx',frame)

    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()