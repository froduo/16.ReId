#ifndef REIDDEEP_H
#define REIDDEEP_H

#include <QMutex>
#include <QString>
#include <opencv2/opencv.hpp>

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

    /// 加载 ONNX 模型；成功返回 true
    bool load(const QString &onnxPath);
    /// 是否就绪
    bool ready() const { return m_ready; }
    /// 当前模型路径
    QString modelPath() const { return m_path; }
    /// 当前是否实际运行在 CUDA(GPU) 上（load() 之后有效）
    bool cudaActive() const { return m_cudaActive; }
    /// 是否允许使用 CUDA（需在 load() 之前调用；不可用时自动回退 CPU）。默认 true
    void setUseCuda(bool v) { m_useCuda = v; }

    /// 提取人体框特征（1×512 CV_32F，L2 归一化）；失败返回空 Mat
    cv::Mat embed(const cv::Mat &frame, const cv::Rect &personBox);

    /// 余弦相似度（两个 L2 归一化特征），映射到 [0,1]
    float compare(const cv::Mat &a, const cv::Mat &b) const;

    static int featureDim() { return 512; }

private:
    mutable QMutex m_mutex;   // cv::dnn::Net::forward 非线程安全
    cv::dnn::Net m_net;
    bool m_ready{false};
    bool m_cudaActive{false}; // 是否实际运行在 CUDA(GPU)
    bool m_useCuda{true};     // 是否允许使用 CUDA（默认允许，不可用则回退 CPU）
    QString m_path;
};

#endif // REIDDEEP_H
