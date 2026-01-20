#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:/opt/hpcx/ucc/lib:${LD_LIBRARY_PATH:-}

# 1) TensorRT 경로 보정 (repo의 CMakeLists.txt가 /usr/local/tensorrt 기준)
arch="$(uname -m)"
mkdir -p /usr/local/tensorrt

if [ -d "/usr/include/${arch}-linux-gnu" ]; then
  ln -sf "/usr/include/${arch}-linux-gnu" /usr/local/tensorrt/include || true
else
  ln -sf "/usr/include" /usr/local/tensorrt/include || true
fi

if [ -d "/usr/lib/${arch}-linux-gnu" ]; then
  ln -sf "/usr/lib/${arch}-linux-gnu" /usr/local/tensorrt/lib || true
else
  ln -sf "/usr/lib" /usr/local/tensorrt/lib || true
fi

# 2) prebuilt C++ 바이너리를 /workspace(호스트 마운트)에 복사 (코드 수정 없이 inferenceTimeTester의 빌드 스킵 유도)
if [ -d "/workspace" ]; then
  # Inference_Time_Tester
  mkdir -p /workspace/CPP_Project/Inference_Time_Tester/build
  if [ -f "/opt/prebuilt/Inference_Time_Tester/trt_cpp_infer_time_tester" ] && \
     [ ! -f "/workspace/CPP_Project/Inference_Time_Tester/build/trt_cpp_infer_time_tester" ]; then
    cp -f /opt/prebuilt/Inference_Time_Tester/trt_cpp_infer_time_tester \
          /workspace/CPP_Project/Inference_Time_Tester/build/
    chmod +x /workspace/CPP_Project/Inference_Time_Tester/build/trt_cpp_infer_time_tester || true
  fi

  # Feature_Extractor
  mkdir -p /workspace/CPP_Project/Feature_Extractor/build
  if [ -f "/opt/prebuilt/Feature_Extractor/trt_feature_extractor" ] && \
    [ ! -f "/workspace/CPP_Project/Feature_Extractor/build/trt_feature_extractor" ]; then
    cp -f /opt/prebuilt/Feature_Extractor/trt_feature_extractor \
          /workspace/CPP_Project/Feature_Extractor/build/
    chmod +x /workspace/CPP_Project/Feature_Extractor/build/trt_feature_extractor || true
  fi
fi

exec "$@"
