cmake_minimum_required(VERSION 3.10)
project(trt_feature_extractor)

# 1) C++17 사용
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2) 의존성 찾기
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ZLIB REQUIRED)

# (추가) Jetson에서 cudart 경로가 /usr/local/cuda/targets/... 인 경우 대비
if (EXISTS "/usr/local/cuda/targets/aarch64-linux/lib")
  link_directories("/usr/local/cuda/targets/aarch64-linux/lib")
endif()
if (EXISTS "/usr/local/cuda/lib64")
  link_directories("/usr/local/cuda/lib64")
endif()

# 3) include 경로
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${ZLIB_INCLUDE_DIRS}          # ZLIB 헤더
    ${CMAKE_SOURCE_DIR}/../third_party/cnpy
)

# 4) cnpy 정적 라이브러리
add_library(cnpy STATIC
    ${CMAKE_SOURCE_DIR}/../third_party/cnpy/cnpy.cpp
)

# 5) 실행 파일
add_executable(trt_feature_extractor main.cpp)

# 6) 링크할 라이브러리
target_link_libraries(trt_feature_extractor PRIVATE
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    cnpy
    ZLIB::ZLIB
    nvinfer
    nvonnxparser
    cudart
    # 만약 GCC<9 이면 아래 주석 해제
    # stdc++fs
)

# 7) (선택) 디버그 + 최적화 플래그
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g -fdiagnostics-color=always")
