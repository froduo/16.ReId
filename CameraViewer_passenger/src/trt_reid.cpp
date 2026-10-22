#include "trt_reid.h"
#include "logger.h"

#include <NvOnnxParser.h>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <cmath>

using namespace nvinfer1;

// ImageNet 均值/方差（与训练/导出 OSNet 一致）
static const float REID_MEAN[3] = {0.485f, 0.456f, 0.406f};
static const float REID_STD[3]  = {0.229f, 0.224f, 0.225f};

TrtReid::TrtReid(const std::string &onnxPath, ILogger &logger, bool preferFp16)
    : logger(logger)
{
    // 引擎缓存路径：与 ONNX 同名、扩展名替换为 .engine
    std::string enginePath = onnxPath;
    const std::string ext = ".onnx";
    if (enginePath.size() >= ext.size()
        && enginePath.compare(enginePath.size() - ext.size(), ext.size(), ext) == 0) {
        enginePath = enginePath.substr(0, enginePath.size() - ext.size()) + ".engine";
    } else {
        enginePath += ".engine";
    }

    bool ok = false;
    {
        std::ifstream f(enginePath, std::ios::binary);
        ok = f.good();
        f.close();
    }
    if (ok) {
        ok = loadFromFile(enginePath);
        if (ok)
            LOG_INFO("[TrtReid] 已加载缓存的 TensorRT 引擎: " + QString::fromStdString(enginePath));
    }
    if (!ok) {
        // 无缓存引擎 → 从 ONNX 构建（针对当前 GPU 架构），成功后缓存到磁盘
        if (buildFromOnnx(onnxPath, enginePath, preferFp16))
            ok = loadFromFile(enginePath);
    }
    if (!ok) {
        LOG_WARN("[TrtReid] TensorRT 引擎不可用（将回退其它后端）");
        return;
    }

    if (!setupIO()) {
        LOG_WARN("[TrtReid] IO 初始化失败");
        return;
    }
    m_ready = true;
    LOG_INFO(QString("[TrtReid] TensorRT ReID 就绪(GPU): in=%1x%2 out=1x%3x%4x%5")
                 .arg(inW).arg(inH).arg(outC).arg(outH).arg(outW));
}

TrtReid::~TrtReid()
{
    if (dIn) cudaFree(dIn);
    if (dOut) cudaFree(dOut);
    if (stream) cudaStreamDestroy(stream);
    if (context) delete context;
    if (engine) delete engine;
    if (runtime) delete runtime;
}

bool TrtReid::loadFromFile(const std::string &enginePath)
{
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.is_open()) return false;

    file.seekg(0, std::ios::end);
    const size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    if (size < 4) return false;

    std::vector<char> data(size);
    file.read(data.data(), size);
    file.close();

    runtime = createInferRuntime(logger);
    if (!runtime) return false;
    engine = runtime->deserializeCudaEngine(data.data(), size);
    if (!engine) return false;
    context = engine->createExecutionContext();
    return context != nullptr;
}

bool TrtReid::buildFromOnnx(const std::string &onnxPath, const std::string &enginePath,
                            bool preferFp16)
{
    LOG_INFO("[TrtReid] 未找到缓存引擎，正在从 ONNX 构建 TensorRT 引擎（首次启动较慢）: "
             + QString::fromStdString(onnxPath));

    IBuilder *builder = createInferBuilder(logger);
    if (!builder) return false;

    INetworkDefinition *network = builder->createNetworkV2(0U);
    nvonnxparser::IParser *parser = nvonnxparser::createParser(*network, logger);
    if (!parser
        || !parser->parseFromFile(onnxPath.c_str(),
                                  static_cast<int>(ILogger::Severity::kWARNING))) {
        LOG_WARN("[TrtReid] ONNX 解析失败: " + QString::fromStdString(onnxPath));
        delete parser;
        delete network;
        delete builder;
        return false;
    }

    IBuilderConfig *config = builder->createBuilderConfig();
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1ULL << 30);
    if (preferFp16 && builder->platformHasFastFp16())
        config->setFlag(BuilderFlag::kFP16);

    IHostMemory *serialized = builder->buildSerializedNetwork(*network, *config);
    bool ok = false;
    if (serialized) {
        std::ofstream out(enginePath, std::ios::binary);
        if (out.is_open()) {
            out.write(static_cast<const char *>(serialized->data()), serialized->size());
            out.close();
            LOG_INFO(QString("[TrtReid] 引擎构建完成并缓存: %1 (%2 bytes)")
                         .arg(QString::fromStdString(enginePath)).arg(serialized->size()));
        }
        ok = true;
    } else {
        LOG_WARN("[TrtReid] 引擎构建失败（可能是 ONNX 含不支持算子）");
    }

    delete serialized;
    delete config;
    delete parser;
    delete network;
    delete builder;
    return ok;
}

bool TrtReid::setupIO()
{
    const int n = engine->getNbIOTensors();
    inName.clear();
    outName.clear();
    for (int i = 0; i < n; ++i) {
        const char *nm = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(nm) == TensorIOMode::kINPUT)
            inName = nm;
        else if (outName.empty())
            outName = nm;
    }
    if (inName.empty() || outName.empty()) return false;

    const Dims inD = engine->getTensorShape(inName.c_str());
    if (inD.nbDims >= 4) { inH = inD.d[2]; inW = inD.d[3]; }

    const Dims outD = engine->getTensorShape(outName.c_str());
    if (outD.nbDims == 4)      { outC = outD.d[1]; outH = outD.d[2]; outW = outD.d[3]; }
    else if (outD.nbDims == 2) { outC = outD.d[1]; outH = 1; outW = 1; }
    else return false;

    inputSize  = static_cast<size_t>(3) * inW * inH * sizeof(float);
    outputSize = static_cast<size_t>(outC) * outH * outW * sizeof(float);
    hostInput.assign(inputSize / sizeof(float), 0.f);
    hostOutput.assign(outputSize / sizeof(float), 0.f);

    if (cudaMalloc(&dIn, inputSize) != cudaSuccess) return false;
    if (cudaMalloc(&dOut, outputSize) != cudaSuccess) return false;
    cudaStreamCreate(&stream);
    return true;
}

cv::Mat TrtReid::runForward(const cv::Mat &rgb128x256)
{
    cv::Mat blob = cv::dnn::blobFromImage(rgb128x256);   // 1×3xH×W（HWC→CHW）
    std::memcpy(hostInput.data(), blob.ptr<float>(),
                std::min(inputSize, blob.total() * blob.elemSize()));

    cudaMemcpyAsync(dIn, hostInput.data(), inputSize, cudaMemcpyHostToDevice, stream);
    context->setTensorAddress(inName.c_str(), dIn);
    context->setTensorAddress(outName.c_str(), dOut);
    if (!context->enqueueV3(stream)) return cv::Mat();
    cudaMemcpyAsync(hostOutput.data(), dOut, outputSize, cudaMemcpyDeviceToHost, stream);
    if (cudaStreamSynchronize(stream) != cudaSuccess) return cv::Mat();

    // 全局平均池化 → 1×C
    cv::Mat feat(1, outC, CV_32F, cv::Scalar(0));
    const int plane = outH * outW;
    for (int c = 0; c < outC; ++c) {
        float sum = 0.f;
        const float *p = hostOutput.data() + static_cast<size_t>(c) * plane;
        for (int i = 0; i < plane; ++i) sum += p[i];
        feat.at<float>(0, c) = sum / static_cast<float>(plane);
    }
    const double nrm = cv::norm(feat, cv::NORM_L2);
    if (nrm > 1e-12) feat /= static_cast<float>(nrm);
    return feat;
}

cv::Mat TrtReid::embed(const cv::Mat &frame, const cv::Rect &personBox)
{
    if (!m_ready || frame.empty() || personBox.empty())
        return cv::Mat();

    const cv::Rect box = personBox & cv::Rect(0, 0, frame.cols, frame.rows);
    if (box.width < 4 || box.height < 8)
        return cv::Mat();

    cv::Mat crop = frame(box).clone();
    if (crop.channels() == 3)      cv::cvtColor(crop, crop, cv::COLOR_BGR2RGB);
    else if (crop.channels() == 4) cv::cvtColor(crop, crop, cv::COLOR_BGRA2RGB);
    else                           cv::cvtColor(crop, crop, cv::COLOR_GRAY2RGB);

    cv::resize(crop, crop, cv::Size(inW, inH), 0, 0, cv::INTER_LINEAR);
    crop.convertTo(crop, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> ch(3);
    cv::split(crop, ch);
    for (int c = 0; c < 3; ++c)
        ch[c] = (ch[c] - REID_MEAN[c]) / REID_STD[c];
    cv::merge(ch, crop);

    return runForward(crop);
}
