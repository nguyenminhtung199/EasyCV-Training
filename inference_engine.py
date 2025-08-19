import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

# ====== Load engine ======
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
engine_path = "/workspace/weight/model_v1/model_mask.onnx_b8_gpu0_fp32.engine"

with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# ====== Preprocess image ======
def preprocess(image_path, target_size=(224,224)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = np.float32(image)
    image = np.transpose(image, (2,0,1))   # CHW
    image = np.expand_dims(image, axis=0)  # NCHW
    image = np.ascontiguousarray(image, dtype=np.float32) 
    return image
image = preprocess("/workspace/photo_2025-08-16_11-32-06.jpg")

# ====== Allocate buffers ======
input_shape = (1,3,224,224)
output_shape = (1,1)

d_input = cuda.mem_alloc(image.nbytes)
h_output = np.empty(output_shape, dtype=np.float32)
d_output = cuda.mem_alloc(h_output.nbytes)

bindings = [int(d_input), int(d_output)]

# ====== Run inference ======
stream = cuda.Stream()

context.set_binding_shape(0, (1,3,224,224))

assert context.all_binding_shapes_specified  # check chắc chắn

cuda.memcpy_htod_async(d_input, image, stream)
context.execute_async_v2(bindings, stream.handle)
cuda.memcpy_dtoh_async(h_output, d_output, stream)
stream.synchronize()

print("TensorRT output:", h_output)