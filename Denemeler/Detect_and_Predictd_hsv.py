import cv2
print(cv2.__version__)
import numpy as np
import time
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow as tf
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

output_saved_model="Egit_agirliklari/Robot_Angle_pre_SavedModel_FP16"
saved_model_loaded=tf.saved_model.load(output_saved_model)

infer =saved_model_loaded.signatures['serving_default']
#oncesinde kayili bir session kaldiysa hafizayi temizle
class_Angle=['Angle0', 'Angle10', 'Angle100', 'Angle105', 'Angle110', 'Angle115', 'Angle120', 'Angle125', 'Angle130', 'Angle135', 'Angle140', 'Angle145', 'Angle15', 'Angle150', 'Angle155', 'Angle160', 'Angle165', 'Angle170', 'Angle175', 'Angle180', 'Angle185', 'Angle190', 'Angle195', 'Angle20', 'Angle200', 'Angle205', 'Angle210', 'Angle215', 'Angle220', 'Angle225', 'Angle230', 'Angle235', 'Angle240', 'Angle245', 'Angle25', 'Angle250', 'Angle255', 'Angle260', 'Angle265', 'Angle270', 'Angle275', 'Angle280', 'Angle285', 'Angle290', 'Angle295', 'Angle30', 'Angle300', 'Angle305', 'Angle310', 'Angle315', 'Angle320', 'Angle325', 'Angle330', 'Angle335', 'Angle340', 'Angle345', 'Angle35', 'Angle350', 'Angle355', 'Angle40', 'Angle45', 'Angle5', 'Angle50', 'Angle55', 'Angle60', 'Angle65', 'Angle70', 'Angle75', 'Angle80', 'Angle85', 'Angle90', 'Angle95']
prev_frame_time=0
new_frame_time=0


cam= cv2.VideoCapture('goruntuler/deneme.avi')

l_b=np.array([53,111,167])#Bunlar yesil icin
u_b=np.array([107,219,241])#bunlar yesil icin
while True:
    ret, frame = cam.read()
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    FGmask=cv2.inRange(hsv,l_b,u_b)
    
    contours,hierarchy=cv2.findContours(FGmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours=sorted(contours,key=lambda x:cv2.contourArea(x),reverse=True)
    for cnt in contours:
        area=cv2.contourArea(cnt)
        (x,y,w,h)=cv2.boundingRect(cnt)
        if area>=50:
            #cv2.drawContours(frame,[cnt],0,(255,0,0),3)
            #cv2.rectangle(frame,(x-30,y-30),(x+w+30,y+h+30),(255,0,0),3)
            Robot=frame[(y-30):(y+h+30),(x-30):(x+w+30)]
            Robot_pre=cv2.resize(Robot,(299,299))
            hsv=cv2.cvtColor(Robot_pre,cv2.COLOR_BGR2HSV)
            FGmask2=cv2.inRange(hsv,l_b,u_b)
            FG=cv2.bitwise_and(Robot_pre, Robot_pre ,mask=FGmask2)
    #Predict icin image'i hazrilayoruz
    X = image.img_to_array(FG)
    X = np.expand_dims(X,axis=0)
    #images = np.vstack([X])
    image_input=tf.constant(X.astype('float32'))

    #tahmini calistiriyoruz
    preds = infer(image_input)


    new_frame_time=time.time()
    fps=1/(new_frame_time-prev_frame_time)
    prev_frame_time=new_frame_time    
    fps=int(fps)
    fps_1=str(fps)
    cv2.putText(frame,fps_1,(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.imshow('xx',frame)

    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()