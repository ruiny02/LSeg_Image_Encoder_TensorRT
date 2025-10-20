import tensorrt as trt

engine_file = "lseg_img_enc_vit_demo_e200__fp16_sparse_cublas_cudnn_ws1024MiB.trt"
logger = trt.Logger(trt.Logger.WARNING)

with open(engine_file, "rb") as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

print("=== Input/Output Tensor Info ===")
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    dtype = engine.get_tensor_dtype(name)
    shape = engine.get_tensor_shape(name)
    role = "INPUT" if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else "OUTPUT"
    print(f"[{role}] {name} | shape={shape} | dtype={dtype}")
