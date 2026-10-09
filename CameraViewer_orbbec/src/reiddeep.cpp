#include "reiddeep.h"
#include "logger.h"

#include <algorithm>
#include <vector>

#include <opencv2/dnn.hpp>
#ifdef USE_OPENCV_CUDA
#include <opencv2/core/cuda.hpp>
#endif

// OSNet 输入规格与 ImageNet 归一化
static constexpr int IN_W = 128;
static constexpr int IN_H = 256;
static const float MEAN[3] = {0.485f, 0.456f, 0.406f};
static const float STD[3]  = {0.229f, 0.224f, 0.225f};

DeepReID::DeepReID()
{
}

bool DeepReID::load(const QString &onnxPath)
{
    QMutexLocker locker(&m_mutex);
    m_ready = false;
    m_path.clear();

    if (onnxPath.isEmpty()) {
        LOG_WARN("[DeepReID] 模型路径为空");
        return false;
    }

    try {
        m_net = cv::dnn::readNetFromONNX(onnxPath.toStdString());
    } catch (const cv::Exception &e) {
        LOG_WARN(QString("[DeepReID] 加载 ONNX 失败: %1 (%2)")
                     .arg(onnxPath).arg(QString::fromStdString(e.what())));
        return false;
    }

    if (m_net.empty()) {
        LOG_WARN("[DeepReID] 加载 ONNX 失败(空网络): " + onnxPath);
        return false;
    }

    // ===== OpenCV DNN 后端选择：优先 CUDA(GPU)，不可用/不兼容时自动回退 CPU =====
    // 说明：OSNet(ONNX) 是标准 CNN，CUDA 后端结果正确，可显著降低 CPU 占用；
    //      而 YOLOv8 ONNX 的 CUDA 后端在本机 OpenCV 上有已知错误（见 persondetector.cpp），
    //      故检测走 TensorRT 引擎，ReID 走这里的 CUDA。
    m_cudaActive = false;
#ifdef USE_OPENCV_CUDA
    int cudaDev = 0;
    try { cudaDev = cv::cuda::getCudaEnabledDeviceCount(); } catch (...) { cudaDev = 0; }

    if (m_useCuda && cudaDev > 0) {
        try {
            m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            // 预热一次：若存在 CUDA 不支持的层会在此抛异常，从而及时回退 CPU
            cv::Mat warm = cv::Mat::zeros(IN_H, IN_W, CV_32FC3);
            cv::Mat warmBlob = cv::dnn::blobFromImage(warm);
            m_net.setInput(warmBlob);
            cv::Mat warmOut = m_net.forward();
            m_cudaActive = !warmOut.empty();
        } catch (const cv::Exception &e) {
            LOG_WARN(QString("[DeepReID] CUDA 后端不可用，回退 CPU: %1")
                         .arg(QString::fromStdString(e.what())));
            m_cudaActive = false;
        } catch (...) {
            m_cudaActive = false;
        }
    }
#endif

    if (!m_cudaActive) {
        try {
            m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        } catch (...) {
        }
    }

    m_ready = true;
    m_path = onnxPath;
    LOG_INFO(QString("[DeepReID] 深度 ReID 模型已加载: %1 | backend=%2")
                 .arg(onnxPath)
                 .arg(m_cudaActive ? QStringLiteral("CUDA(GPU)") : QStringLiteral("CPU")));
    return true;
}

cv::Mat DeepReID::embed(const cv::Mat &frame, const cv::Rect &personBox)
{
    if (!m_ready || frame.empty() || personBox.empty())
        return cv::Mat();

    cv::Rect box = personBox & cv::Rect(0, 0, frame.cols, frame.rows);
    if (box.width < 4 || box.height < 8)
        return cv::Mat();

    cv::Mat crop = frame(box).clone();

    // BGR->RGB，缩放到 256x128
    if (crop.channels() == 3) {
        cv::cvtColor(crop, crop, cv::COLOR_BGR2RGB);
    } else if (crop.channels() == 4) {
        cv::cvtColor(crop, crop, cv::COLOR_BGRA2RGB);
    } else {
        cv::cvtColor(crop, crop, cv::COLOR_GRAY2RGB);
    }
    cv::resize(crop, crop, cv::Size(IN_W, IN_H), 0, 0, cv::INTER_LINEAR);

    // 归一化：/255, (x-mean)/std
    crop.convertTo(crop, CV_32F, 1.0 / 255.0);
    std::vector<cv::Mat> ch(3);
    cv::split(crop, ch);
    for (int c = 0; c < 3; ++c) {
        ch[c] = (ch[c] - MEAN[c]) / STD[c];
    }
    cv::merge(ch, crop);

    cv::Mat blob = cv::dnn::blobFromImage(crop); // 1x3x256x128 (已归一化, 不再缩放)

    cv::Mat out;
    {
        QMutexLocker locker(&m_mutex);
        try {
            m_net.setInput(blob);
            out = m_net.forward();   // 1 x 512 x H' x W'
        } catch (const cv::Exception &e) {
            LOG_WARN(QString("[DeepReID] 推理失败: %1").arg(QString::fromStdString(e.what())));
            return cv::Mat();
        }
    }
    if (out.empty())
        return cv::Mat();

    cv::Mat feat;
    if (out.dims == 4) {
        // 4D 输出 1×C×H'×W'（本工程导出的 OSNet 为 1×512×16×8）→ 全局平均池化
        const int C = out.size[1];
        const int H = out.size[2];
        const int W = out.size[3];
        feat = cv::Mat(1, C, CV_32F, cv::Scalar(0));
        const float *src = reinterpret_cast<const float *>(out.data);
        const int plane = H * W;
        for (int c = 0; c < C; ++c) {
            float sum = 0.f;
            const float *p = src + static_cast<size_t>(c) * plane;
            for (int i = 0; i < plane; ++i) sum += p[i];
            feat.at<float>(0, c) = sum / static_cast<float>(plane);
        }
    } else if (out.dims == 2) {
        // 兼容直接输出 1×C（已池化）的导出格式
        feat = out.reshape(1, 1).clone();
        feat.convertTo(feat, CV_32F);
    } else {
        return cv::Mat();
    }

    // L2 归一化
    const double n = cv::norm(feat, cv::NORM_L2);
    if (n > 1e-12) feat /= static_cast<float>(n);
    return feat;
}

float DeepReID::compare(const cv::Mat &a, const cv::Mat &b) const
{
    if (a.empty() || b.empty())
        return 0.0f;
    const cv::Mat x = a.reshape(1, 1);
    const cv::Mat y = b.reshape(1, 1);
    if (x.total() != y.total() || x.channels() != y.channels())
        return 0.0f;

    // 两者均已 L2 归一化 → 点积即余弦相似度
    const double dot = x.dot(y);
    const float s = static_cast<float>(std::max(0.0, std::min(1.0, dot)));
    return s;
}
