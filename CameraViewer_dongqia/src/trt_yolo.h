#ifndef TRT_YOLO_H
#define TRT_YOLO_H

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <QDebug>

// TensorRT 日志记录器
class TrtLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            qDebug() << "[TensorRT]" << msg;
    }
};

// 检测结果
struct Detection
{
    int class_id;
    float score;
    int x;
    int y;
    int w;
    int h;
};

// TensorRT YOLO 检测引擎封装（仅 Detection 模式）
class TrtYolo
{
public:
    TrtYolo(const std::string& enginePath, nvinfer1::ILogger& logger,
            float confThreshold = 0.25f);
    ~TrtYolo();

    void preprocess(const cv::Mat& img);
    void infer();
    void postprocess(std::vector<Detection>& results);
    /// 返回所有类别检测结果（不过滤 class_id，用于诊断）
    void postprocessAll(std::vector<Detection>& results, float lowThreshold = 0.05f);

    float confThreshold;

private:
    bool loadEngine(const std::string& enginePath);
    size_t getSizeByDim(const nvinfer1::Dims& dims);

    nvinfer1::ILogger& logger;
    nvinfer1::IRuntime* runtime{nullptr};
    nvinfer1::ICudaEngine* engine{nullptr};
    nvinfer1::IExecutionContext* context{nullptr};

    cudaStream_t stream{};

    int inputW{640};
    int inputH{640};

    size_t inputSize{0};
    size_t outputSize{0};

    std::vector<float> hostInput;
    std::vector<float> hostOutput;

    cv::Mat lastImage;

    float padX = 0.f;
    float padY = 0.f;
    float scale = 1.f;

    std::string inputTensorName;
    std::string outputTensorName;
    void* buffers[2];
};

#endif // TRT_YOLO_H
