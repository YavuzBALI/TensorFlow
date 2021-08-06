#Bu Kod houge circle ile .trt karisimidir
import numpy as np
import cv2 as cv
from tensorflow.keras.preprocessing import image
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import tensorflow as tf
import os
import time
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
#Loaded model how saved Saved<odel format

output_saved_model="Egit_agirliklari/Robot_Angle_pre_SavedModel_FP16"
saved_model_loaded=tf.saved_model.load(output_saved_model)

infer =saved_model_loaded.signatures['serving_default']
#oncesinde kayili bir session kaldiysa hafizayi temizle

#Robot Alaini ve rek araliklari belirlendi
kesit=50
l_b=np.array([54,91,180])
u_b=np.array([179,204,255])

#Siniflari ekrana basmak i.in bir array olusturduk
class_Angle=['Angle0', 'Angle10', 'Angle100', 'Angle105', 'Angle110', 'Angle115', 'Angle120', 'Angle125', 'Angle130', 'Angle135', 'Angle140', 'Angle145', 'Angle15', 'Angle150', 'Angle155', 'Angle160', 'Angle165', 'Angle170', 'Angle175', 'Angle180', 'Angle185', 'Angle190', 'Angle195', 'Angle20', 'Angle200', 'Angle205', 'Angle210', 'Angle215', 'Angle220', 'Angle225', 'Angle230', 'Angle235', 'Angle240', 'Angle245', 'Angle25', 'Angle250', 'Angle255', 'Angle260', 'Angle265', 'Angle270', 'Angle275', 'Angle280', 'Angle285', 'Angle290', 'Angle295', 'Angle30', 'Angle300', 'Angle305', 'Angle310', 'Angle315', 'Angle320', 'Angle325', 'Angle330', 'Angle335', 'Angle340', 'Angle345', 'Angle35', 'Angle350', 'Angle355', 'Angle40', 'Angle45', 'Angle5', 'Angle50', 'Angle55', 'Angle60', 'Angle65', 'Angle70', 'Angle75', 'Angle80', 'Angle85', 'Angle90', 'Angle95']

#deneme kaydi okunur
prev_frame_time=0
new_frame_time=0
cam=cv.VideoCapture('goruntuler/deneme.avi')

while True:
    #image alinir
    ret, img = cam.read()

    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray,5)
    cimg = cv.cvtColor(gray,cv.COLOR_GRAY2BGR)
    circles = cv.HoughCircles(gray,cv.HOUGH_GRADIENT,1,30,
                                param1=50,param2=30,minRadius=15,maxRadius=35)

    #burada np.any ile arrayin bos mu dolu mu oldugunu kontrol ettik
    if np.any(circles):
        circles = np.uint16(np.around(circles))

        for i in circles[0,:]:
            # Get frame of Robot and resize
            Robot=img[(i[1]-kesit):(i[1]+kesit),(i[0]-kesit):(i[0]+kesit)]
        
        Robot_pre=cv.resize(Robot,(299,299))
        hsv=cv.cvtColor(Robot_pre,cv.COLOR_BGR2HSV)
        FGmask=cv.inRange(hsv,l_b,u_b)
        FG=cv.bitwise_and(Robot_pre, Robot_pre ,mask=FGmask)

        #Predict icin image'i hazrilayoruz
        X = image.img_to_array(FG)
        X = np.expand_dims(X,axis=0)
        #images = np.vstack([X])
        image_input=tf.constant(X.astype('float32'))

        #tahmini calistiriyoruz
        preds = infer(image_input)




        cv.rectangle(img,(i[0]-kesit,i[1]-kesit),(i[0]+kesit,i[1]+kesit),(255,0,0),2)
    
    new_frame_time=time.time()
    fps=1/(new_frame_time-prev_frame_time)
    prev_frame_time=new_frame_time
    fps=int(fps)

    print(fps)
    cv.imshow('detected circles',img)

    if cv.waitKey(1)==ord('q'):
        break

cv.waitKey(0)
cv.destroyAllWindows()