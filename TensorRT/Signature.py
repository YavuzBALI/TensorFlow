import tensorflow as tf 
from tensorflow.python.compiler.tensorrt import trt_convert as trt

#Registered model address is entered
output_saved_model="/home/yavuz/OpenCv2Tensorflow_RobocupAngle/Robot_Angle_pre_SavedModel_101x101_FP16_build_2"
#Load Model
saved_model_loaded=tf.saved_model.load(output_saved_model,)
#The model signature key is taken.
signature_keys= list(saved_model_loaded.signatures.keys())
#Printing signatur key
print(signature_keys)

#The model object is retrieved.
infer =saved_model_loaded.signatures['serving_default']
#Printing model object
print(infer.structured_outputs)

