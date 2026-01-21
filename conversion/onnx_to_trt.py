import argparse
import tensorrt as trt
import os

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def build_dynamic_engine(onnx_path,
                         engine_path,
                         use_fp16: bool,
                         disable_timing_cache: bool,
                         gpu_fallback: bool,
                         debug_mode: bool,
                         use_sparse: bool,
                         use_cublas: bool,
                         use_cudnn: bool,
                         workspace_size: int):
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(EXPLICIT_BATCH) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser, \
         builder.create_builder_config() as config:

        # ⭐️ 플래그로 제어 가능한 최적화 옵션 (순서 재배치)
        # 1) FP16
        if use_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            config.clear_flag(trt.BuilderFlag.FP16)

        # 2) Sparse Weights
        if use_sparse:
            config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
        else:
            config.clear_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

        # 4) Disable Timing Cache
        if disable_timing_cache:
            config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
        else:
            config.clear_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)

        # 6) GPU Fallback (INT8)
        if gpu_fallback:
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

        # 8) DEBUG
        if debug_mode:
            config.set_flag(trt.BuilderFlag.DEBUG)

        # 10) 워크스페이스 메모리 한도
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            workspace_size
        )

        # Tactic sources 플래그 조립
        tactic_mask = 0
        if args.use_cublas:
            tactic_mask |= 1 << int(trt.TacticSource.CUBLAS)
        if args.use_cudnn:
            tactic_mask |= 1 << int(trt.TacticSource.CUDNN)
        config.set_tactic_sources(tactic_mask)


        # ── ONNX 파싱 ──────────────────────────────────────────────
        #
        # ① 외부-가중치(external data) 를 포함한 대형 ONNX는
        #    parse() 대신  parse_from_file() 을 써야 경로를
        #    자동으로 추적합니다.
        #
        print(f"🔍  parsing  ONNX : {onnx_path}")
        if not parser.parse_from_file(onnx_path.encode()):
            print("❌ ONNX parse-error(s):")
            for i in range(parser.num_errors):
                print(f"   ▶ {parser.get_error(i)}")
            raise RuntimeError("ONNX parsing failed")

        input_tensor = network.get_input(0)
        profile = builder.create_optimization_profile()
        # Scannet 원본 encode_images.py 입력과 동일 범위를 보장:
        #  base 320x240 × 0.75 → 240x180 까지 허용
        #  (N,C,H,W) = (1,3,180,240) .. (1,3,1024,1024)
        # 배치도 동적으로: flip 배치=2를 위해 N∈[1,2]
        profile.set_shape(
            input_tensor.name,
            (1, 3, 288, 512),   # MIN (N=1)
            (1, 3, 288, 512),   # OPT (N=1)
            (1, 3, 288, 512),   # MAX (N=1)
        )
        config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("❌ buildSerializedNetwork returned None (profile/shape constraint violation)")
        engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(serialized)
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())
        print(f"✅ Dynamic TRT 엔진 저장: {engine_path}")

    return engine


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build dynamic TRT engine with optional optimizations"
    )

    # ─── Required I/O ──────────────────────────────────────────────
    parser.add_argument(
        "--onnx", required=True,
        help="Path to input ONNX file"
    )

    # ─── Resource limits ──────────────────────────────────────────
    parser.add_argument(
        "--workspace", type=int, default=1 << 29,
        help="Workspace size in bytes (default: 1<<29)"
    )

    # ─── Precision & sparsity ────────────────────────────────────
    parser.add_argument("--fp16",     dest="fp16",    action="store_true",  default=True,
                        help="Enable FP16 precision")
    parser.add_argument("--no-fp16",  dest="fp16",    action="store_false",
                        help="Disable FP16 precision")
    parser.add_argument("--sparse",   dest="sparse",  action="store_true",  default=True,
                        help="Enable sparse weights tactics")
    parser.add_argument("--no-sparse",dest="sparse",  action="store_false",
                        help="Disable sparse weights tactics")

    # ─── Builder flags ───────────────────────────────────────────
    parser.add_argument("--disable-timing-cache",   action="store_true", default=False,
                        help="Disable timing cache")
    parser.add_argument("--gpu-fallback",           action="store_true", default=False,
                        help="Allow GPU fallback for INT8")
    
    # ─── Tactic Sources ───────────────────────────────────────────
    parser.add_argument("--use-cublas",   dest="use_cublas",   action="store_true",  default=True,
                    help="Enable cuBLAS tactics")
    parser.add_argument("--no-cublas",    dest="use_cublas",   action="store_false",
                        help="Disable cuBLAS tactics")
    parser.add_argument("--use-cudnn",    dest="use_cudnn",    action="store_true",  default=True,
                        help="Enable cuDNN tactics")
    parser.add_argument("--no-cudnn",     dest="use_cudnn",    action="store_false",
                        help="Disable cuDNN tactics")
    
    # ─── Debug & profiling ───────────────────────────────────────
    parser.add_argument("--debug",               action="store_true", default=False,
                        help="Enable debug mode")

    args = parser.parse_args()

    # ─── 프로젝트 루트 & 모델 폴더 경로 ────────────────────────
    script_dir   = os.path.dirname(__file__)
    project_dir  = os.path.abspath(os.path.join(script_dir, os.pardir))
    trt_dir      = os.path.join(project_dir, "models", "trt_engines")
    os.makedirs(trt_dir, exist_ok=True)

    # 1) onnx 파일명(확장자 제외) 추출
    base   = os.path.splitext(os.path.basename(args.onnx))[0]

    # 2) 빌드 옵션 태그 생성
    flags = [
        "fp16" if args.fp16 else "fp32",
        "sparse"        if args.sparse else None,
        "noTC"          if args.disable_timing_cache else None,
        "gpuFB"         if args.gpu_fallback else None,
        "dbg"           if args.debug else None,
        "cublas" if args.use_cublas else None,
        "cudnn" if args.use_cudnn else None,
        f"ws{args.workspace>>20}MiB"
    ]
    flags = [f for f in flags if f]
    suffix          = "_".join(flags)
    engine_basename = f"{base}__{suffix}.trt"
    engine_path     = os.path.join(trt_dir, engine_basename)

    # 4) 빌드 호출
    build_dynamic_engine(
        args.onnx,
        engine_path,
        use_fp16               = args.fp16,
        disable_timing_cache   = args.disable_timing_cache,
        gpu_fallback           = args.gpu_fallback,
        debug_mode             = args.debug,
        use_sparse             = args.sparse,
        use_cublas           = args.use_cublas,
        use_cudnn            = args.use_cudnn,
        workspace_size         = args.workspace
    )

    print(f"\n✅ Engine saved as: {engine_path}")