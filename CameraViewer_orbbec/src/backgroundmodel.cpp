#include "backgroundmodel.h"

#ifdef USE_OPENCV_CUDA
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#endif

BackgroundModel::BackgroundModel()
{
#ifdef USE_OPENCV_CUDA
    try {
        if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
            m_useGpu = true;
            initGpuFilters();
        }
    } catch (...) {
        m_useGpu = false;
    }
#endif
}

#ifdef USE_OPENCV_CUDA
void BackgroundModel::initGpuFilters()
{
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    m_openFilter  = cv::cuda::createMorphologyFilter(cv::MORPH_OPEN,  CV_8UC1, kernel);
    m_closeFilter = cv::cuda::createMorphologyFilter(cv::MORPH_CLOSE, CV_8UC1, kernel);
}
#endif

void BackgroundModel::reset()
{
    QMutexLocker lk(&m_mutex);
    m_bg.release();
    m_frames = 0;
    m_cachedMask.release();
    m_cachedData = nullptr;
    m_cachedW = m_cachedH = 0;
#ifdef USE_OPENCV_CUDA
    m_bgGpu.release();
#endif
}

bool BackgroundModel::ready() const
{
    QMutexLocker lk(&m_mutex);
#ifdef USE_OPENCV_CUDA
    if (m_useGpu) return !m_bgGpu.empty() && m_frames >= m_minFrames;
#endif
    return !m_bg.empty() && m_frames >= m_minFrames;
}

void BackgroundModel::update(const cv::Mat &frame)
{
    if (frame.empty()) return;

    cv::Mat f32;
    frame.convertTo(f32, CV_32F);

    QMutexLocker lk(&m_mutex);

#ifdef USE_OPENCV_CUDA
    if (m_useGpu) {
        cv::cuda::GpuMat g32;
        g32.upload(f32);
        if (m_bgGpu.empty() || m_bgGpu.size() != g32.size() || m_bgGpu.type() != g32.type()) {
            m_bgGpu = g32.clone();
            m_frames = 1;
            return;
        }
        // 前景门控（GPU）
        cv::cuda::GpuMat gdiff, ggray, gfg;
        cv::cuda::absdiff(g32, m_bgGpu, gdiff);
        cv::cuda::cvtColor(gdiff, ggray, cv::COLOR_BGR2GRAY);
        cv::cuda::threshold(ggray, gfg, static_cast<double>(m_diffThresh), 255, cv::THRESH_BINARY);
        const double ratio = static_cast<double>(cv::cuda::countNonZero(gfg)) /
                             static_cast<double>(gfg.size().area());
        if (ratio < m_fgGate) {
            const double lr = (m_frames < m_minFrames) ? 0.15 : m_learnRate;
            // 等价于 accumulateWeighted(dst = lr*src + (1-lr)*dst)
            cv::cuda::addWeighted(g32, lr, m_bgGpu, 1.0 - lr, 0.0, m_bgGpu);
        }
        if (m_frames < 1000000) m_frames++;
        return;
    }
#endif

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

#ifdef USE_OPENCV_CUDA
    if (m_useGpu) {
        if (m_bgGpu.empty() || m_bgGpu.size() != frame.size()) return cv::Mat();

        cv::Mat f32;
        frame.convertTo(f32, CV_32F);
        cv::cuda::GpuMat g32, gdiff, ggray, gmask32, gmask, gopen, gclose;
        g32.upload(f32);
        cv::cuda::absdiff(g32, m_bgGpu, gdiff);
        cv::cuda::cvtColor(gdiff, ggray, cv::COLOR_BGR2GRAY);
        // 阈值化在 CV_32F 上完成（与 CPU 路径一致），再转成 CV_8U(0/255)。
        // 关键：形态学滤波器按 CV_8UC1 创建，输入掩码必须是 CV_8U，
        // 否则 apply() 会触发 "src.type() == type_" 断言失败。
        cv::cuda::threshold(ggray, gmask32, static_cast<double>(m_diffThresh), 255, cv::THRESH_BINARY);
        gmask32.convertTo(gmask, CV_8U);

        // 形态学开/闭运算：使用独立输出缓冲，避免同址读写。
        cv::cuda::GpuMat morph = gmask;
        if (m_openFilter)  { m_openFilter->apply(morph, gopen);  morph = gopen; }
        if (m_closeFilter) { m_closeFilter->apply(morph, gclose); morph = gclose; }

        cv::Mat mask;
        morph.download(mask);
        m_cachedMask = mask;
        m_cachedData = frame.data;
        m_cachedW = frame.cols;
        m_cachedH = frame.rows;
        return mask;
    }
#endif

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
