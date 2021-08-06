import tensorflow as tf 
from tensorflow.python.compiler.tensorrt import trt_convert as trt 
import numpy as np

#saved model loaded to convert
saved_model_dir='101x101_image_size/Robot_Angle_pre_SavedModel_101_colab'

#An offline converter for TF-TRT transformation for TF 2.0 SavedModels.
#Currently this is not available on Windows platform.

# Setting convert paramater
conversion_param = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
    precision_mode=trt.TrtPrecisionMode.FP16,
    max_workspace_size_bytes=80000000
)

#An offline converter for TF-TRT transformation for TF 2.0 SavedModels.
#Currently this is not available on Windows platform.
converter=trt.TrtGraphConverterV2(
    input_saved_model_dir=saved_model_dir,
    conversion_params=conversion_param
)
#convert model TensorRT optimizer
converter.convert()

#Run inference with converted graph in order to build TensorRT engines.
def my_input_fn():
  inp1 = np.random.normal(size=(1, 101, 101, 3)).astype(np.float32)
  yield [inp1]
converter.build(input_fn=my_input_fn())

#The converted model is saved
saved_model_dir_trt='Robot_Angle_pre_SavedModel_101x101_FP16_build_3'
converter.save(saved_model_dir_trt)

print('***********************Convert is Done********************')