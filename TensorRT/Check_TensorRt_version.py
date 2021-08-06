import tensorflow as tf 
print('Tensorflow Version:',tf.version.VERSION)

#Check TensorRT Version
print('TensorRT Version')

!dpkg -l | grep nvinfer
