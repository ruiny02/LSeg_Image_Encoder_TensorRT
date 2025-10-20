#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <cmath>
#include <cuda_runtime.h>
#include <NvInfer.h>      // TensorRT core API
#include <cstring>        // strcmp
#include "NvInferRuntime.h"
#include <fstream>

using namespace nvinfer1;
using Clock = std::chrono::high_resolution_clock;

class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
};

ICudaEngine* loadEngine(const std::string& enginePath, ILogger& logger) {
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error opening engine file: " << enginePath << std::endl;
        return nullptr;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();

    IRuntime* runtime = createInferRuntime(logger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(buffer.data(), size);
    delete runtime;

    return engine;
}


void printProgress(int current, int total) {
    const int barWidth = 40;
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(barWidth * progress);

    std::cout << "\r[";
    for (int i = 0; i < barWidth; ++i) {
        std::cout << (i < pos ? '=' : ' ');
    }
    std::cout << "] " << int(progress * 100.0) << "% (" 
              << current << "/" << total << ")" << std::flush;
}


void measureTRT(const std::string& enginePath, int H, int W, int iterations = 100) {
    Logger logger;
    ICudaEngine* engine = loadEngine(enginePath, logger);
    if (!engine) {
        std::cerr << "Failed to load engine: " << enginePath << std::endl;
        return;
    }

    IExecutionContext* context = engine->createExecutionContext();

    // ① binding index 찾기
    int inputIdx  = -1, outputIdx = -1;
    int nbIO = engine->getNbIOTensors();           // 전체 I/O 텐서 개수
    for(int i = 0; i < nbIO; ++i) {
        const char* name = engine->getIOTensorName(i);
        auto mode = engine->getTensorIOMode(name);
        if(mode == nvinfer1::TensorIOMode::kINPUT  && !strcmp(name,"input"))
            inputIdx = i;
        if(mode == nvinfer1::TensorIOMode::kOUTPUT && !strcmp(name,"output"))
            outputIdx = i;
    }
    if(inputIdx < 0 || outputIdx < 0)
        throw std::runtime_error("cannot find input/output tensor index");
 
    std::string inputTensorName = engine->getIOTensorName(0);
    std::string outputTensorName = engine->getIOTensorName(1);

    int inputElems  = 3 * H * W;
    // 출력의 경우, ViT 패치 그리드 크기에 맞추려면 H/patch, W/patch 로 계산하거나
    // 기존처럼 (512 × H × W) 를 써도 무방합니다.
    int outputElems = 512 * (H/2) * (W/2);
    size_t inputBytes = inputElems * sizeof(float);
    size_t outputBytes = outputElems * sizeof(float);

    // Allocate
    float* dummy = new float[inputElems];
    // ▶ 모든 값을 1.0f 로 채워서, sum = 1.0 × 3 × H × W 이 되도록
    std::fill(dummy, dummy+inputElems, 1.0f);
    void* d_input; void* d_output;
    cudaMalloc(&d_input, inputBytes);
    cudaMalloc(&d_output, outputBytes);
    cudaMemcpy(d_input, dummy, inputBytes, cudaMemcpyHostToDevice);

    context->setTensorAddress(inputTensorName.c_str(), d_input);
    context->setTensorAddress(outputTensorName.c_str(), d_output);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Warm-up 10회 (매 번 dynamic H×W 지정!)
    for(int i = 0; i < 10; ++i) {
        nvinfer1::Dims4 inDim{1,3,H,W};
        context->setInputShape("input", inDim);
        context->enqueueV3(stream);
        cudaStreamSynchronize(stream);
    }

    // ▶▶ 디버그 한 번만 돌아갈 부분
    // (1) hostOutput, outSize 선언
    std::vector<float> hostOutput(outputElems);
    size_t outSize = outputElems * sizeof(float);

    // (2) dynamic shape 지정
    nvinfer1::Dims4 inDim{1,3,H,W};
    context->setInputShape("input", inDim);

    // (3) inference + 복사
    context->enqueueV3(stream);
    cudaStreamSynchronize(stream);
    cudaMemcpyAsync(hostOutput.data(), d_output, outSize, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // (4) sum/mean 계산
    float sum = std::accumulate(hostOutput.begin(), hostOutput.end(), 0.0f);
    std::cout << "[DEBUG] C++ TRT single-run sum=" << sum
            << ", mean=" << sum/hostOutput.size() << std::endl;

    std::vector<double> times;
    times.reserve(iterations);
    for(int i = 0; i < iterations; i++){
        auto t0 = Clock::now();

        // ─── TensorRT10+ 에서는 setInputShape(name, Dims) 로만 shape 설정 ───
        nvinfer1::Dims4 inDim{1,3,H,W};
        context->setInputShape("input", inDim);

        // 원래 있던 추론 호출
        context->enqueueV3(stream);
        cudaStreamSynchronize(stream);

        auto t1 = Clock::now();
        times.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count()
        );

        // printProgress(i+1, iterations);  // ← 주석 처리
    }
    // std::cout << std::endl;  // progress bar 없앨 때 주석 처리

    double mean = std::accumulate(times.begin(), times.end(), 0.0)/iterations;
    double var = 0;
    for(double t: times) var += (t-mean)*(t-mean);
    var /= iterations;
    double stddev = std::sqrt(var);

    std::cout << "[RESULT] Size="<< W << "x" << H <<" Avg="<<mean<<" ms ± "<<stddev<<" ms\n";

    // Cleanup
    delete[] dummy;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream);
    delete context;
    delete engine;
}


int main(int argc, char** argv) {
    // 이제 <H1> <W1> [H2 W2 ...] 페어로 받습니다
    if(argc < 5 || (argc - 3) % 2 != 0) {
        std::cerr << "Usage: " << argv[0]
                  << " <engine.trt> <iterations> <H1> <W1> [H2 W2 ...]\n";
        return -1;
    }
    int iters = std::stoi(argv[2]);
    int H     = std::stoi(argv[3]);
    int W     = std::stoi(argv[4]);


    std::string enginePath = argv[1];
    int iterations = std::stoi(argv[2]);
    // (H, W) 페어 단위로 measureTRT 호출
    for(int i = 3; i + 1 < argc; i += 2) {
        int H = std::stoi(argv[i]);
        int W = std::stoi(argv[i+1]);
        measureTRT(enginePath, H, W, iterations);
    }

    return 0;
}