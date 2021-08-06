import tensorflow as tf 
from tensorflow.python.compiler.tensorrt import trt_convert as trt

output_saved_model="/home/yavuz/OpenCv2Tensorflow_RobocupAngle/Robot_Angle_pre_SavedModel_101x101_FP16_build_2"
saved_model_loaded=tf.saved_model.load(output_saved_model,)
signature_keys= list(saved_model_loaded.signatures.keys())
print(signature_keys)

infer =saved_model_loaded.signatures['serving_default']
print(infer.structured_outputs)

