"""
Check the GPU status and version of the tensorflow installation
"""
import tensorflow as tf

print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
