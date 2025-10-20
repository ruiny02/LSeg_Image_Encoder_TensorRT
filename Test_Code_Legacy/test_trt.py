import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # 자동으로 CUDA 초기화
import numpy as np


test_w = 640
test_h = 480

# TensorRT 로거 생성
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def load_engine(engine_file_path):
    with open(engine_file_path, "rb") as f:
        engine_data = f.read()
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_data)
    return engine

# 엔진 로드 (엔진 파일 경로 수정)
engine_path = "lseg_img_enc_vit_ade20k.trt"
engine = load_engine(engine_path)
context = engine.create_execution_context()

def get_binding_index_by_name(engine, binding_name):
    n = engine.num_io_tensors    # ✅ 올바른 속성
    for i in range(n):
        if engine.get_tensor_name(i) == binding_name:
            return i
    raise ValueError(f"Binding name {binding_name} not found in engine.")

input_binding_idx = get_binding_index_by_name(engine, "input")
output_binding_idx = get_binding_index_by_name(engine, "output")
input_name  = engine.get_tensor_name(input_binding_idx)
output_name = engine.get_tensor_name(output_binding_idx)
print("Input tensor shape:",  engine.get_tensor_shape(input_name))
print("Output tensor shape:", engine.get_tensor_shape(output_name))

# 1) 바인딩 이름 가져오기
input_name   = engine.get_tensor_name(input_binding_idx)
output_name  = engine.get_tensor_name(output_binding_idx)

# 2) 실제 shape 조회 (문자열 이름을 넘겨야 함)
in_dims  = engine.get_tensor_shape(input_name)    # Dims(batch, C, H, W)
out_dims = engine.get_tensor_shape(output_name)

# 3) Tuple 로 변환
input_shape  = tuple(in_dims)   # (1,3,H,W) 또는 dynamic profile 이라면 (1,3,?,?)
input_shape = (1, 3, test_h, test_w)
output_shape = tuple(out_dims)

# 4) dynamic engine 이면 반드시 set_input_shape 호출
context.set_input_shape(input_name, input_shape)

# (2) 실제 출력 텐서 shape 얻기 — 올바른 API
out_dims = context.get_tensor_shape(output_name)   # Dims(batch, C, H, W)
output_shape = tuple(out_dims)
output_size = int(np.prod(output_shape) * np.dtype(np.float32).itemsize)

# 5) 메모리 할당 (올바른 방식: 실제 dummy_input 크기로)
dummy_input = np.random.randn(*input_shape).astype(np.float32)
input_size   = dummy_input.nbytes
d_input      = cuda.mem_alloc(input_size)

# output_size 는 이미 int 캐스트 되어 있으니 그대로
d_output = cuda.mem_alloc(output_size)

# (5) binding 주소 등록 (여기가 빠져 있었습니다)
context.set_tensor_address(input_name,  int(d_input))
context.set_tensor_address(output_name, int(d_output))

bindings = [int(d_input), int(d_output)]
stream = cuda.Stream()


# 입력 데이터를 GPU 메모리로 복사 후 추론 실행
cuda.memcpy_htod_async(d_input, dummy_input, stream)
context.execute_async_v3(stream.handle)

# 출력 데이터를 CPU 메모리로 복사
output_data = np.empty(output_shape, dtype=np.float32)
cuda.memcpy_dtoh_async(output_data, d_output, stream)
stream.synchronize()

print("TensorRT 엔진 추론 출력 shape:", output_data.shape)
print("TensorRT 엔진 추론 출력 (일부):", output_data.flatten()[:10])
