#include "depthgrabthread.h"
#include "logger.h"
#include <QElapsedTimer>
#include <algorithm>
#include <cmath>
#include <vector>

// ===== 深度帧 → 伪彩显示（纯 OpenCV 实现）=====
namespace {

cv::Mat depthToColor(const cv::Mat &depthMM, float minMM = 300.0f, float maxMM = 3000.0f)
{
    if (depthMM.empty() || depthMM.type() != CV_16U)
        return cv::Mat();

    cv::Mat mask = depthMM > 0;
    cv::Mat f;
    depthMM.convertTo(f, CV_32F);

    cv::Mat norm = (f - minMM) * (255.0f / std::max(1.0f, maxMM - minMM));
    cv::min(norm, 255.0f, norm);
    cv::max(norm, 0.0f, norm);
    cv::Mat u8;
    norm.convertTo(u8, CV_8U);

    cv::Mat color;
    cv::applyColorMap(u8, color, cv::COLORMAP_JET);
    color.setTo(cv::Scalar(0, 0, 0), ~mask);
    return color;
}

cv::Mat depthToColorAuto(const cv::Mat &depthMM)
{
    if (depthMM.empty() || depthMM.type() != CV_16U)
        return cv::Mat();

    static constexpr int BIN_SHIFT   = 3;        // 8mm/箱
    static constexpr int BIN_COUNT   = 2048;     // 覆盖 0..16383mm
    static constexpr int MAX_SAMPLES = 65536;

    const int total  = depthMM.rows * depthMM.cols;
    const int stride = std::max(1, total / MAX_SAMPLES);

    std::vector<int> hist(BIN_COUNT, 0);
    int valid = 0;
    int idx = 0;
    for (int y = 0; y < depthMM.rows; ++y) {
        const uint16_t *p = depthMM.ptr<uint16_t>(y);
        for (int x = 0; x < depthMM.cols; ++x, ++idx) {
            if ((idx % stride) != 0) continue;
            const uint16_t d = p[x];
            if (d == 0) continue;
            int b = d >> BIN_SHIFT;
            if (b >= BIN_COUNT) b = BIN_COUNT - 1;
            hist[b]++;
            valid++;
        }
    }
    if (valid == 0)
        return depthToColor(depthMM);

    const int loTarget = static_cast<int>(valid * 0.05);
    const int hiTarget = static_cast<int>(valid * 0.95);
    int acc = 0, loBin = 0, hiBin = BIN_COUNT - 1;
    for (int b = 0; b < BIN_COUNT; ++b) {
        acc += hist[b];
        if (acc >= loTarget) { loBin = b; break; }
    }
    acc = 0;
    for (int b = 0; b < BIN_COUNT; ++b) {
        acc += hist[b];
        if (acc >= hiTarget) { hiBin = b; break; }
    }

    float lo = static_cast<float>(loBin << BIN_SHIFT);
    float hi = static_cast<float>((hiBin << BIN_SHIFT) + ((1 << BIN_SHIFT) - 1));
    if (hi - lo < 100.0f) { lo = 300.0f; hi = 3000.0f; }
    return depthToColor(depthMM, lo, hi);
}

cv::Mat depthFrameToDisplay(const cv::Mat &depth)
{
    if (depth.empty()) return cv::Mat();

    switch (depth.type()) {
    case CV_16U:
        return depthToColorAuto(depth);
    case CV_8UC1: {
        cv::Mat color;
        cv::applyColorMap(depth, color, cv::COLORMAP_JET);
        return color;
    }
    case CV_8UC3: {
        cv::Mat gray;
        cv::cvtColor(depth, gray, cv::COLOR_BGR2GRAY);
        cv::Mat back;
        cv::cvtColor(gray, back, cv::COLOR_GRAY2BGR);
        if (cv::norm(depth, back, cv::NORM_INF) == 0) {
            cv::Mat color;
            cv::applyColorMap(gray, color, cv::COLORMAP_JET);
            return color;
        }
        return depth.clone();
    }
    case CV_8UC4: {
        cv::Mat bgr;
        cv::cvtColor(depth, bgr, cv::COLOR_BGRA2BGR);
        cv::Mat gray;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
        cv::Mat back;
        cv::cvtColor(gray, back, cv::COLOR_GRAY2BGR);
        if (cv::norm(bgr, back, cv::NORM_INF) == 0) {
            cv::Mat color;
            cv::applyColorMap(gray, color, cv::COLORMAP_JET);
            return color;
        }
        return bgr;
    }
    default:
        return cv::Mat();
    }
}

} // namespace

DepthGrabThread::DepthGrabThread(CameraController *cam, const QString &channelName, QObject *parent)
    : QThread(parent)
    , m_camera(cam)
    , m_channelName(channelName)
{
}

void DepthGrabThread::run()
{
    cv::Mat depthFrame;
    int frameIndex = 0;
    QElapsedTimer frameTimer;
    frameTimer.start();

    LOG_INFO(QString("[DepthGrabThread-%1] Started grabbing depth frames").arg(m_channelName));

    while (m_running.load())
    {
        if (!m_camera->getDepthFrame(depthFrame)) {
            m_consecutiveFailCount++;
            if (m_consecutiveFailCount == 1) {
                LOG_WARN(QString("[DepthGrabThread-%1] Depth frame grab failed")
                         .arg(m_channelName));
            } else if (m_consecutiveFailCount % 300 == 0) {
                LOG_WARN(QString("[DepthGrabThread-%1] Consecutive depth failures: %2")
                         .arg(m_channelName).arg(m_consecutiveFailCount));
            }
            QThread::msleep(10);
            continue;
        }

        if (m_consecutiveFailCount > 0) {
            LOG_INFO(QString("[DepthGrabThread-%1] Depth frame grab recovered after %2 failures")
                     .arg(m_channelName).arg(m_consecutiveFailCount));
            m_consecutiveFailCount = 0;
        }

        frameIndex++;

        // 缓存 16 位毫米深度（供外部读取最近深度图）
        if (depthFrame.type() == CV_16U) {
            QMutexLocker lk(&m_mutex);
            m_lastDepth = depthFrame.clone();
        }

        // 伪彩显示
        cv::Mat dcolor = depthFrameToDisplay(depthFrame);
        if (!dcolor.empty()) {
            cv::Mat drgb;
            cv::cvtColor(dcolor, drgb, cv::COLOR_BGR2RGB);
            QImage dimg(drgb.data, drgb.cols, drgb.rows,
                        static_cast<int>(drgb.step), QImage::Format_RGB888);
            emit depthFrameGrabbed(dimg.copy());
        }

        // 帧率日志（降低频率）
        if (frameIndex % 500 == 0) {
            qint64 elapsed = frameTimer.elapsed();
            LOG_INFO(QString("[DepthGrabThread-%1] Depth frame #%2, interval=%3ms")
                     .arg(m_channelName).arg(frameIndex).arg(elapsed));
            frameTimer.restart();
        }
    }

    LOG_INFO(QString("[DepthGrabThread-%1] Exited, total frames: %2")
             .arg(m_channelName).arg(frameIndex));
}
