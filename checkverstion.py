import tensorflow as tf
import sys
import os

# In ra phiên bản TensorFlow và Python
print("TensorFlow version:", tf.__version__)
print("Python version:", sys.version)
print("Python version info:", sys.version_info)

# Lấy thông tin build của TensorFlow
build_info = tf.sysconfig.get_build_info()
print("\nTensorFlow Build Info:")
for key, value in build_info.items():
    print(f"{key}: {value}")

# In thêm compile flags và link flags
compile_flags = tf.sysconfig.get_compile_flags()
print("\nTensorFlow Compile Flags:")
print(compile_flags)

link_flags = tf.sysconfig.get_link_flags()
print("\nTensorFlow Link Flags:")
print(link_flags)

# In ra các đường dẫn include và thư viện của TensorFlow
include_path = tf.sysconfig.get_include()
lib_path = tf.sysconfig.get_lib()
print("\nTensorFlow Include Path:", include_path)
print("TensorFlow Library Path:", lib_path)

# In ra thông tin GPU (nếu có)
gpus = tf.config.list_physical_devices('GPU')
print("\nInstalled CUDA Devices:")
print(gpus)

# In ra các biến môi trường liên quan đến CUDA/cuDNN (nếu có)
print("\nCUDA/cuDNN Environment Variables:")
cuda_env_vars = {k: os.environ[k] for k in os.environ if 'CUDA' in k or 'CUDNN' in k or 'cuDNN' in k}
print(cuda_env_vars)
