#ifndef TRT_REID_H
#define TRT_REID_H

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <string>
#include <vector>

/**
 * @brief 基于 TensorRT 的深度 ReID(OSNet) 推理封装
 *
 * 目的：把「识别」侧最重的 OSNet 前向从 CPU 卸载到 GPU(TensorRT)，
 * 显著降低 CPU 占用，并让实时推理更流畅（在采集/处理线程内完成，不阻塞主界面显示）。
 *
 * 引擎来源：
 *   - 若存在与 ONNX 同名的 .engine 文件 → 直接反序列化（启动即用）；
 *   - 否则用 nvonnxparser 解析 ONNX 并在本机构建引擎（针对当前 GPU 架构，如 sm_75），
 *     构建成功后缓存为 .engine，后续启动直接复用。
 *
 * 输入：人体框裁剪 → 256×128 RGB，ImageNet 均值/方差归一化；
 * 输出：1×512 全局平均池化 + L2 归一化嵌入（余弦相似度）。
 */
class TrtReid
{
public:
    TrtReid(const std::string &onnxPath, nvinfer1::ILogger &logger, bool preferFp16 = true);
    ~TrtReid();

    bool ready() const { return m_ready; }
    static int featureDim() { return 512; }

    /// 提取人体框特征（1×512 CV_32F，L2 归一化）；失败返回空 Mat
    cv::Mat embed(const cv::Mat &frame, const cv::Rect &personBox);

private:
    bool loadFromFile(const std::string &enginePath);
    bool buildFromOnnx(const std::string &onnxPath, const std::string &enginePath, bool preferFp16);
    bool setupIO();
    cv::Mat runForward(const cv::Mat &rgb128x256);

    nvinfer1::ILogger &logger;
    nvinfer1::IRuntime     *runtime{nullptr};
    nvinfer1::ICudaEngine  *engine{nullptr};
    nvinfer1::IExecutionContext *context{nullptr};
    cudaStream_t stream{};

    int inW{128}, inH{256};
    size_t inputSize{0}, outputSize{0};
    std::vector<float> hostInput, hostOutput;

    int outC{512}, outH{16}, outW{8};
    std::string inName, outName;
    void *dIn{nullptr}, *dOut{nullptr};

    bool m_ready{false};
};

#endif // TRT_REID_H
