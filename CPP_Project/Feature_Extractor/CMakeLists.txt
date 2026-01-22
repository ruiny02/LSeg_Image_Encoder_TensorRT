cmake_minimum_required(VERSION 3.10)
project(trt_feature_extractor)

# 1) C++17 사용
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2) 의존성 찾기
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ZLIB REQUIRED)

# 3) include 경로
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${ZLIB_INCLUDE_DIRS}          # ZLIB 헤더
    ${CMAKE_SOURCE_DIR}/../third_party/cnpy
)

# 4) 라이브러리 경로 (Jetson DLA / cuDLA 포함)
set(TEGRA_LIB_DIR /usr/lib/aarch64-linux-gnu/tegra)
set(NVIDIA_LIB_DIR /usr/lib/aarch64-linux-gnu/nvidia)
set(CUDA_COMPAT_DIR /usr/local/cuda/compat)
link_directories(
    /usr/local/lib
    /usr/local/cuda/lib64
    ${CUDA_COMPAT_DIR}
    /usr/local/tensorrt/lib
    ${TEGRA_LIB_DIR}
    ${NVIDIA_LIB_DIR}
)

# 5) DLA / cuDLA 관련 라이브러리 찾기 (없으면 링크 생략)
find_library(CUDLA_LIB cudla PATHS /usr/local/cuda/lib64 ${CUDA_COMPAT_DIR} ${TEGRA_LIB_DIR} ${NVIDIA_LIB_DIR} /usr/lib/aarch64-linux-gnu)
find_library(NVCUDLA_LIB nvcudla PATHS ${TEGRA_LIB_DIR} ${NVIDIA_LIB_DIR} /usr/lib/aarch64-linux-gnu /usr/local/cuda/lib64 ${CUDA_COMPAT_DIR})
find_library(NVDLA_COMPILER_LIB nvdla_compiler PATHS ${TEGRA_LIB_DIR} ${NVIDIA_LIB_DIR} /usr/lib/aarch64-linux-gnu)
find_library(NVDLA_RUNTIME_LIB nvdla_runtime PATHS ${TEGRA_LIB_DIR} ${NVIDIA_LIB_DIR} /usr/lib/aarch64-linux-gnu)
set(TRT_EXTRA_LIBS "")
if(CUDLA_LIB)
  list(APPEND TRT_EXTRA_LIBS ${CUDLA_LIB})
endif()
if(NVCUDLA_LIB)
  list(APPEND TRT_EXTRA_LIBS ${NVCUDLA_LIB})
endif()
if(NVDLA_COMPILER_LIB)
  list(APPEND TRT_EXTRA_LIBS ${NVDLA_COMPILER_LIB})
endif()
if(NVDLA_RUNTIME_LIB)
  list(APPEND TRT_EXTRA_LIBS ${NVDLA_RUNTIME_LIB})
endif()

# 6) cnpy 정적 라이브러리
add_library(cnpy STATIC
    ${CMAKE_SOURCE_DIR}/../third_party/cnpy/cnpy.cpp
)

# 7) 실행 파일
add_executable(trt_feature_extractor main.cpp)

# 8) 링크할 라이브러리
target_link_libraries(trt_feature_extractor PRIVATE
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    cnpy
    ZLIB::ZLIB
    nvinfer
    nvonnxparser
    cudart
    ${TRT_EXTRA_LIBS}
    # 만약 GCC<9 이면 아래 주석 해제
    # stdc++fs
)

# 9) (선택) 디버그 + 최적화 플래그
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g -fdiagnostics-color=always")

# 10) 런타임에 tegra/cuda 경로가 잡히도록 RPATH 설정
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "/usr/lib/aarch64-linux-gnu/tegra;/usr/local/cuda/lib64")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
