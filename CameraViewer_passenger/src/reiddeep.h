#ifndef REIDDEEP_H
#define REIDDEEP_H

#include <QMutex>
#include <QString>
#include <opencv2/opencv.hpp>

class TrtReid;     // TensorRT ReID 推理（GPU，见 trt_reid.h）
class TrtLogger;   // TensorRT 日志记录器（见 trt_yolo.h）

/**
 * @brief 深度行人重识别（OSNet x1.0 / MSMT17 预训练），基于 OpenCV DNN 推理
 *
 * 背景：手工颜色/纹理直方图在「鱼眼俯视 + 相似着装」场景下区分力不足
 *      （实测同一人与不同人的相似度重叠），故引入深度外观嵌入。
 *
 * 特性：
 *  - 输入人体框 → 输出 512 维 L2 归一化全局特征（余弦相似度）
 *  - 对光照、朝向、着装细节（如衣服印花）具有显著更强的区分力
 *  - 使用已链接的 opencv_dnn 模块，无需额外第三方依赖
 */
class DeepReID
{
public:
    DeepReID();
    ~DeepReID();

    /// 加载 ONNX 模型；成功返回 true
    bool load(const QString &onnxPath);
    /// 是否就绪
    bool ready() const { return m_ready; }
    /// 当前模型路径
    QString modelPath() const { return m_path; }

    /// 提取人体框特征（1×512 CV_32F，L2 归一化）；失败返回空 Mat
    cv::Mat embed(const cv::Mat &frame, const cv::Rect &personBox);

    /// 余弦相似度（两个 L2 归一化特征），映射到 [0,1]
    float compare(const cv::Mat &a, const cv::Mat &b) const;

    static int featureDim() { return 512; }

    /// 是否优先使用 GPU 后端（需在 load() 前设置；不可用时自动回退 CPU）
    void setPreferableBackend(bool useCuda) { m_preferCuda = useCuda; }
    /// 当前是否实际运行在 GPU 上（TensorRT 或 OpenCV-CUDA）
    bool cudaActive() const { return m_trt != nullptr || m_cudaActive; }
    /// 当前实际推理后端名（"TensorRT(GPU)" / "CUDA(GPU)" / "CPU"）
    QString backendName() const { return m_backend; }

private:
    mutable QMutex m_mutex;   // cv::dnn::Net::forward 非线程安全
    cv::dnn::Net m_net;
    bool m_ready{false};
    bool m_preferCuda{true};    // 优先 GPU 后端
    bool m_cudaActive{false};   // OpenCV DNN CUDA 后端是否生效
    QString m_backend{QStringLiteral("CPU")};  // 实际生效的后端名
    TrtReid   *m_trt{nullptr};        // TensorRT ReID（优先，落 GPU）
    TrtLogger *m_trtLogger{nullptr};  // TrtReid 生命周期内必须存活
    QString m_path;
};

#endif // REIDDEEP_H
