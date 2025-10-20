import os, glob, subprocess, torch, time, re
import tensorrt as trt
import numpy as np
import pandas as pd
from tqdm import tqdm
import pycuda.driver as cuda
import pycuda.autoinit
import argparse

# Lightning modules
from modules.lseg_module import LSegModule      # ViT
from modules.lseg_module_zs import LSegModuleZS  # ResNet-ZS

torch.backends.cudnn.benchmark = True

# Utility: run a shell command
def run_subprocess(cmd, cwd=None):
    print(f"[CMD] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)

# Measure PyTorch GPU inference
def measure_pytorch_inference_time(net, inp, iterations=100):
    device = torch.device('cuda')
    net.to(device).eval()
    inp = inp.to(device)
    with torch.no_grad(): net(inp)
    for _ in tqdm(range(10), desc="Warm-up PyTorch"): net(inp)
    times = []
    for _ in tqdm(range(iterations), desc="PyTorch Inference"):  
        start = time.time(); _ = net(inp); torch.cuda.synchronize(); times.append((time.time()-start)*1000)
    net.to('cpu'); torch.cuda.empty_cache()
    return float(np.mean(times)), float(np.std(times))

# Measure TensorRT Python inference
def measure_tensorrt_inference_time(engine_path, inp, iterations=100, dynamic=True):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path,'rb') as f:
        rt = trt.Runtime(TRT_LOGGER); engine = rt.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()
    in_name, out_name = engine.get_tensor_name(0), engine.get_tensor_name(1)
    if dynamic: ctx.set_input_shape(in_name, tuple(inp.shape))
    out_shape = tuple(ctx.get_tensor_shape(out_name))
    h_in = inp.cpu().numpy().astype(np.float32)
    h_out = np.empty(out_shape, dtype=np.float32)
    d_in = cuda.mem_alloc(h_in.nbytes); d_out = cuda.mem_alloc(h_out.nbytes)
    ctx.set_tensor_address(in_name, int(d_in)); ctx.set_tensor_address(out_name, int(d_out))
    stream = cuda.Stream()
    def infer():
        cuda.memcpy_htod_async(d_in, h_in, stream)
        ctx.execute_async_v3(stream.handle)
        cuda.memcpy_dtoh_async(h_out, d_out, stream)
        stream.synchronize(); return h_out
    for _ in tqdm(range(10), desc="Warm-up TRT"): infer()
    times = []
    for _ in tqdm(range(iterations), desc="TensorRT Inference"):  
        start = time.time(); _ = infer(); times.append((time.time()-start)*1000)
    return float(np.mean(times)), float(np.std(times))

# Locate engine file helper
def find_engine_file(trt_dir, base, suffix):
    engine_name = f"{base}__{suffix}.trt"
    path = os.path.join(trt_dir, engine_name)
    if os.path.exists(path):
        return path
    # fallback glob
    candidates = glob.glob(os.path.join(trt_dir, f"{base}__*.trt"))
    return candidates[0] if candidates else None

# Run C++ benchmark via main.cpp constructed executable
def run_cpp_benchmark(engine_file, iterations, height, width):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(script_dir, 'CPP_Project', 'Inference_Time_Tester', 'build')
    exe = os.path.join(build_dir, 'trt_cpp_infer_time_tester')
    # build if missing
    if not os.path.exists(exe):
        os.makedirs(build_dir, exist_ok=True)
        run_subprocess(['cmake', '..'], cwd=build_dir)
        cpus = os.cpu_count() or 1
        run_subprocess(['make', f'-j{cpus}'], cwd=build_dir)
    # execute
    cmd = [exe, engine_file, str(iterations), str(height), str(width)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
    avg = std = None
    for line in proc.stdout.splitlines():
        m = re.search(r'Avg=([\d\.]+) ms \u00B1 ([\d\.]+) ms', line)
        if m: avg, std = float(m.group(1)), float(m.group(2)); break
    return avg, std

# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_dir', type=str, default='models/weights')
    parser.add_argument('--img_sizes', nargs='+', type=int,
                        default=[256,320,384,480,640,768,1024])
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--resize', type=int, default=None)
    parser.add_argument('--trt_workspace', type=int, default=1<<30,
                        help='Workspace in bytes (1<<30=1GiB)')
    # TRT flags
    parser.add_argument('--trt_fp16', dest='trt_fp16', action='store_true', default=True)
    parser.add_argument('--no-trt_fp16', dest='trt_fp16', action='store_false')
    parser.add_argument('--trt_sparse', dest='trt_sparse', action='store_true', default=True)
    parser.add_argument('--no-trt_sparse', dest='trt_sparse', action='store_false')
    parser.add_argument('--trt_no_tc', action='store_true', default=False)
    parser.add_argument('--trt_gpu_fb', action='store_true', default=False)
    parser.add_argument('--trt_debug', action='store_true', default=False)
    parser.add_argument('--trt_cublas', action='store_true', default=True)
    parser.add_argument('--no-trt_cublas', dest='trt_cublas', action='store_false')
    parser.add_argument('--trt_cudnn', action='store_true', default=True)
    parser.add_argument('--no-trt_cudnn', dest='trt_cudnn', action='store_false')
    args = parser.parse_args()

    records = []
    backbone_list = ['ViT', 'Resnet']


    # 1) 각 백본별 ckpt 개수 * size 개수로 전체 작업 수 계산
    total_sizes = len(args.img_sizes)
    tasks_per_backbone = []
    for backbone in backbone_list:
        ckpt_dir = os.path.join(args.weights_dir, backbone)
        num_ckpts = len(glob.glob(os.path.join(ckpt_dir, '*.ckpt')))
        tasks_per_backbone.append(num_ckpts * total_sizes)
    total_tasks = sum(tasks_per_backbone)
    global_step = 0

    for b_idx, backbone in enumerate(backbone_list, start=1):
        ckpt_dir = os.path.join(args.weights_dir, backbone)
        onnx_script = 'conversion/model_to_onnx.py' if backbone=='ViT' else 'conversion/model_to_onnx_zs.py'
        ckpt_list = sorted(glob.glob(os.path.join(args.weights_dir, backbone, '*.ckpt')))
        total_ckpts = len(ckpt_list)

        for c_idx, ckpt in enumerate(ckpt_list, start=1):
            total_sizes = len(args.img_sizes)

            tag = os.path.splitext(os.path.basename(ckpt))[0]

            # ViT은 vit, Resnet-ZS는 rn101 키를 써야 onnx/model_to_onnx_zs.py 에서 저장한 이름과 일치합니다
            backbone_key = 'vit' if backbone=='ViT' else 'rn101'
            base = f"lseg_img_enc_{backbone_key}_{tag}"
            
            # ONNX
            onnx_path = f"models/onnx_engines/{base}.onnx"
            if os.path.exists(onnx_path):
                print(f"✅ ONNX exists, skip: {onnx_path}")
            else:
                print(f"Building ONNX: {onnx_path}")
                run_subprocess(['python3', onnx_script, '--weights', ckpt])
            # TRT build
            trt_dir = os.path.join('models','trt_engines'); os.makedirs(trt_dir, exist_ok=True)
            flags = [
                'fp16' if args.trt_fp16 else 'fp32',
                'sparse' if args.trt_sparse else 'nosparse',
                'noTC' if args.trt_no_tc else 'tc',
                'gpuFB' if args.trt_gpu_fb else 'nogpuFB',
                'dbg' if args.trt_debug else 'nodebug',
                'cublas' if args.trt_cublas else 'nocublas',
                'cudnn' if args.trt_cudnn else 'nocudnn',
                f"ws{args.trt_workspace>>20}MiB"
            ]
            suffix = '_'.join(flags)
            engine_file = find_engine_file(trt_dir, base, suffix)
            if engine_file:
                print(f"✅ TRT engine exists, skip: {engine_file}")
            else:
                print(f"Building TRT engine: {base}__{suffix}.trt")
                cmd = ['python3', 'conversion/onnx_to_trt.py', '--onnx', onnx_path, '--workspace', str(args.trt_workspace)]
                cmd += ['--fp16'] if args.trt_fp16 else ['--no-trt_fp16']
                cmd += ['--sparse'] if args.trt_sparse else ['--no-trt_sparse']
                if args.trt_no_tc: cmd.append('--disable-timing-cache')
                if args.trt_gpu_fb: cmd.append('--gpu-fallback')
                if args.trt_debug: cmd.append('--debug')
                cmd.append('--use-cublas' if args.trt_cublas else '--no-cublas')
                cmd.append('--use-cudnn' if args.trt_cudnn else '--no-cudnn')
                run_subprocess(cmd)
                engine_file = find_engine_file(trt_dir, base, suffix)
            print(f"Using TRT engine: {engine_file}")
            # Load model once
            max_crop = max(args.img_sizes)
            if backbone=='ViT':
                module = LSegModule.load_from_checkpoint(
                    checkpoint_path=ckpt, map_location='cpu', backbone='clip_vitl16_384', aux=False,
                    num_features=256, crop_size=max_crop, readout='project', aux_weight=0,
                    se_loss=False, se_weight=0, ignore_index=255, dropout=0.0,
                    scale_inv=False, augment=False, no_batchnorm=False,
                    widehead=True, widehead_hr=False, arch_option=0,
                    block_depth=0, activation='lrelu'
                ).net
            else:
                module = LSegModuleZS.load_from_checkpoint(
                    checkpoint_path=ckpt, map_location='cpu', data_path='data/',
                    dataset='ade20k', backbone='clip_resnet101', aux=False,
                    num_features=256, aux_weight=0, se_loss=False, se_weight=0,
                    base_lr=0, batch_size=1, max_epochs=0, ignore_index=255,
                    dropout=0.0, scale_inv=False, augment=False,
                    no_batchnorm=False, widehead=False, widehead_hr=False,
                    arch_option=0, use_pretrained='True', strict=False,
                    logpath='fewshot/logpath_4T/', fold=0, block_depth=0,
                    nshot=1, finetune_mode=False, activation='lrelu'
                ).net
            # Benchmark per size
            for s_idx, size in enumerate(args.img_sizes, start=1):
                # 2) 글로벌 스텝 +1, percent 계산
                global_step += 1
                percent = global_step / total_tasks * 100

                # 3) 진행 상황 출력
                print(
                    f"\n>> Progress: {global_step}/{total_tasks} "
                    f"({percent:.1f}%)\n"
                    f"   → Backbone = {backbone} "
                    f"({b_idx}/{len(backbone_list)}) | "
                    f"Checkpoint = {os.path.basename(ckpt)} "
                    f"({c_idx}/{len(ckpt_list)}) | "
                    f"Size = {size} ×  {size} "
                    f"({s_idx}/{total_sizes})"
                )

                height, width = (args.resize, args.resize) if args.resize else (size, size)
                inp = torch.ones(1,3,height,width)
                pt_avg, pt_std = measure_pytorch_inference_time(module, inp, args.iterations)
                trt_avg, trt_std = measure_tensorrt_inference_time(engine_file, inp, args.iterations, dynamic=True)
                print("C++ Inference Benchmark is running...")
                cpp_avg, cpp_std = run_cpp_benchmark(engine_file, args.iterations, height, width)
                record = {
                    'Backbone': backbone,
                    'Checkpoint': os.path.basename(ckpt),
                    'Size': size,
                    'Crop Size': max_crop,
                    'PyTorch Avg(ms)': pt_avg, 'PyTorch Std(ms)': pt_std,
                    'TRT Python Avg(ms)': trt_avg, 'TRT Python Std(ms)': trt_std,
                    'TRT C++ Avg(ms)': cpp_avg, 'TRT C++ Std(ms)': cpp_std
                }

                # 리스트에 추가
                records.append(record)

                # 중간 결과 출력
                print(
                    f">> Appended Record – "
                    f"Backbone = {record['Backbone']} | "
                    f"Checkpoint = {record['Checkpoint']} | "
                    f"Size = {record['Size']} ×  {record['Size']} | "
                    f"PyTorch = {record['PyTorch Avg(ms)']:.1f} ±  {record['PyTorch Std(ms)']:.1f}ms | "
                    f"TRT Py = {record['TRT Python Avg(ms)']:.1f} ±  {record['TRT Python Std(ms)']:.1f}ms | "
                    f"TRT C++ = {record['TRT C++ Avg(ms)']:.1f} ±  {record['TRT C++ Std(ms)']:.1f}ms\n"
                )

    df = pd.DataFrame(records)
    pivot = df.pivot_table(
        index=['Backbone','Checkpoint','Size','Crop Size'],
        values=[
            'PyTorch Avg(ms)', 'PyTorch Std(ms)',
            'TRT Python Avg(ms)', 'TRT Python Std(ms)',
            'TRT C++ Avg(ms)', 'TRT C++ Std(ms)'
        ]
    )

    print("\n===== Inference Benchmark Summary =====")
    print(pivot.to_string())

    print("\n===== Inference Benchmark Summary (Markdown) =====")
    print(pivot.to_markdown())

if __name__=='__main__':
    main()
