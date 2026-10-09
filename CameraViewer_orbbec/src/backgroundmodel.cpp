#include "backgroundmodel.h"

BackgroundModel::BackgroundModel()
{
    // 已取消 CUDA/GPU 加速：背景更新与前景分割统一在 CPU 上完成。
}

void BackgroundModel::reset()
{
    QMutexLocker lk(&m_mutex);
    m_bg.release();
    m_frames = 0;
    m_cachedMask.release();
    m_cachedData = nullptr;
    m_cachedW = m_cachedH = 0;
}

bool BackgroundModel::ready() const
{
    QMutexLocker lk(&m_mutex);
    return !m_bg.empty() && m_frames >= m_minFrames;
}

void BackgroundModel::update(const cv::Mat &frame)
{
    if (frame.empty()) return;

    cv::Mat f32;
    frame.convertTo(f32, CV_32F);

    QMutexLocker lk(&m_mutex);

    if (m_bg.empty() || m_bg.size() != f32.size() || m_bg.type() != f32.type()) {
        m_bg = f32.clone();
        m_frames = 1;
        return;
    }

    cv::Mat diff, gray, fg;
    cv::absdiff(f32, m_bg, diff);
    cv::cvtColor(diff, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, fg, static_cast<double>(m_diffThresh), 255, cv::THRESH_BINARY);
    const double ratio = static_cast<double>(cv::countNonZero(fg)) /
                         static_cast<double>(fg.total());
    if (ratio < m_fgGate) {
        const double lr = (m_frames < m_minFrames) ? 0.15 : m_learnRate;
        cv::accumulateWeighted(f32, m_bg, lr);
    }
    if (m_frames < 1000000) m_frames++;
}

cv::Mat BackgroundModel::foregroundMask(const cv::Mat &frame) const
{
    if (frame.empty()) return cv::Mat();

    QMutexLocker lk(&m_mutex);

    // 同一帧缓存命中
    if (!m_cachedMask.empty() && m_cachedData == frame.data &&
        m_cachedW == frame.cols && m_cachedH == frame.rows) {
        return m_cachedMask;
    }

    cv::Mat f32;
    frame.convertTo(f32, CV_32F);
    if (m_bg.empty() || m_bg.size() != f32.size() || m_bg.type() != f32.type())
        return cv::Mat();

    cv::Mat diff, gray, mask32;
    cv::absdiff(f32, m_bg, diff);
    cv::cvtColor(diff, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, mask32, static_cast<double>(m_diffThresh), 255, cv::THRESH_BINARY);

    // 统一导出 CV_8UC1 掩码：下游 calcHist() 与 at<uchar>() 均按 8U 处理。
    cv::Mat mask;
    mask32.convertTo(mask, CV_8U);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);

    m_cachedMask = mask;
    m_cachedData = frame.data;
    m_cachedW = frame.cols;
    m_cachedH = frame.rows;
    return mask;
}
