import cv2
import time

print(cv2.__version__)
prev_frame_time=0
new_frame_time=0
cam2=cv2.VideoCapture('goruntuler/deneme.avi')
while True:
    ret, frame2 = cam2.read()
    

    if cv2.waitKey(1)==ord('q'):
        break
    new_frame_time=time.time()
    fps=1/(new_frame_time-prev_frame_time)
    prev_frame_time=new_frame_time
    fps=int(fps)
    fps_1=str(fps)
    cv2.putText(frame2,fps_1,(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    cv2.imshow('nanoCam2',frame2)
cam2.release()
cv2.destroyAllWindows()