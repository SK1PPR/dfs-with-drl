import tensorflow as tf

# Check if TensorFlow is built with GPU support
print("Is TensorFlow built with CUDA support? ", tf.test.is_built_with_cuda())

# List available physical devices (GPUs)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs detected: {len(gpus)}")
    for gpu in gpus:
        print(f"GPU Name: {gpu.name}")
else:
    print("No GPUs detected.")

# Optional: List all local devices (including CPU and GPU)
from tensorflow.python.client import device_lib
devices = device_lib.list_local_devices()
print("Available devices:")
for device in devices:
    print(device)