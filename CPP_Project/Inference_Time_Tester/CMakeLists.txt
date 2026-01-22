cmake_minimum_required(VERSION 3.10)
project(trt_cpp_infer_time_tester)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ZLIB REQUIRED) # 추가

# Include 디렉토리 설정
include_directories(
    ${PROJECT_SOURCE_DIR}/
    ${CUDA_INCLUDE_DIRS}
    /usr/local/include/opencv4
    /usr/local/tensorrt/include
    ${CMAKE_SOURCE_DIR}/../third_party/cnpy
)

# 라이브러리 경로 설정 (Jetson DLA / cuDLA 포함)
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

# DLA / cuDLA 관련 라이브러리 찾기 (Jetson 빌드에서 없으면 링크 생략)
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

# cnpy 라이브러리 추가
set(CNPY_PATH "${CMAKE_SOURCE_DIR}/../third_party/cnpy")
add_library(cnpy STATIC
    "${CNPY_PATH}/cnpy.cpp"
)
# 실행 파일 설정
add_executable(trt_cpp_infer_time_tester main.cpp)

# 링크할 라이브러리 설정
target_link_libraries(trt_cpp_infer_time_tester
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    cudart
    ${TRT_EXTRA_LIBS}
    nvinfer
    nvonnxparser
    cnpy
    ZLIB::ZLIB   # 🔥 libz 추가
)

# 런타임에 tegra/cuda 경로가 잡히도록 RPATH 설정
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "/usr/lib/aarch64-linux-gnu/tegra;/usr/local/cuda/lib64")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g -fdiagnostics-color=always")
