#include "reiddeep.h"
#include "trt_reid.h"
#include "trt_yolo.h"
#include "logger.h"

#include <algorithm>
#include <vector>
#include <exception>

// OSNet 输入规格与 ImageNet 归一化
static constexpr int IN_W = 128;
static constexpr int IN_H = 256;
static const float MEAN[3] = {0.485f, 0.456f, 0.406f};
static const float STD[3]  = {0.229f, 0.224f, 0.225f};

DeepReID::DeepReID()
{
}

DeepReID::~DeepReID()
{
    delete m_trt;
    m_trt = nullptr;
    delete m_trtLogger;
    m_trtLogger = nullptr;
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

    // ===== 后端选择：TensorRT(GPU) → OpenCV-CUDA(GPU) → CPU =====
    // OSNet 每帧对每个乘客做一次前向，是本工程「识别」侧最大的 CPU 开销。
    // 优先走 TensorRT（针对本机 GPU 架构构建，sm_75 等均可用），
    // 其次 OpenCV DNN CUDA（部分环境 CUDA 架构不匹配会失败），最后回退 CPU。
    m_cudaActive = false;
    m_backend = QStringLiteral("CPU");

    if (m_preferCuda) {
        try {
            if (!m_trtLogger) m_trtLogger = new TrtLogger();
            m_trt = new TrtReid(onnxPath.toStdString(), *m_trtLogger, /*preferFp16=*/true);
            if (m_trt && m_trt->ready()) {
                m_backend = QStringLiteral("TensorRT(GPU)");
            } else {
                delete m_trt; m_trt = nullptr;
            }
        } catch (const std::exception &e) {
            LOG_WARN(QString("[DeepReID] TensorRT 初始化失败: %1")
                         .arg(QString::fromUtf8(e.what())));
            delete m_trt; m_trt = nullptr;
        } catch (...) {
            delete m_trt; m_trt = nullptr;
        }
    }

    // TensorRT 不可用时，尝试 OpenCV DNN CUDA 后端（自检失败再回退 CPU）
    if (!m_trt && m_preferCuda) {
        try {
            m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            // 自检：用一次真实尺寸(256×128)前向验证 CUDA 后端真正可用
            // （部分环境设置后端不报错，直到 forward 才抛异常）；失败则回退 CPU。
            cv::Mat dummy = cv::Mat::zeros(IN_H, IN_W, CV_32FC3);
            cv::Mat blob = cv::dnn::blobFromImage(dummy);
            m_net.setInput(blob);
            m_net.forward();
            m_cudaActive = true;
            m_backend = QStringLiteral("CUDA(GPU)");
        } catch (const cv::Exception &e) {
            LOG_WARN(QString("[DeepReID] OpenCV CUDA 后端不可用，回退 CPU: %1")
                         .arg(QString::fromStdString(e.what())));
            m_cudaActive = false;
        }
    }
    if (!m_trt && !m_cudaActive) {
        try {
            m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        } catch (...) {
        }
        m_backend = QStringLiteral("CPU");
    }

    m_ready = true;
    m_path = onnxPath;
    LOG_INFO(QString("[DeepReID] 深度 ReID 模型已加载: %1 | 推理后端=%2")
                 .arg(onnxPath).arg(m_backend));
    return true;
}

cv::Mat DeepReID::embed(const cv::Mat &frame, const cv::Rect &personBox)
{
    if (!m_ready || frame.empty() || personBox.empty())
        return cv::Mat();

    cv::Rect box = personBox & cv::Rect(0, 0, frame.cols, frame.rows);
    if (box.width < 4 || box.height < 8)
        return cv::Mat();

    // ===== 优先使用 TensorRT(GPU) 后端（识别推理卸载到 GPU）=====
    if (m_trt && m_trt->ready()) {
        QMutexLocker locker(&m_mutex);   // 同一引擎/流不可并发使用
        return m_trt->embed(frame, box);
    }

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
