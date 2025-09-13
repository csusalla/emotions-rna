import tensorflow as tf
print("GPUs:", tf.config.list_physical_devices("GPU"))
print("DML:",  tf.config.list_physical_devices("DML"))  # se usar tensorflow-directml no Windows