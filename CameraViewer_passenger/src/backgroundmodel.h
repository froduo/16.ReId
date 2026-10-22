#ifndef BACKGROUNDMODEL_H
#define BACKGROUNDMODEL_H

#include <QMutex>
#include <opencv2/opencv.hpp>

#ifdef USE_OPENCV_CUDA
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#endif

/**
 * @brief 固定摄像头背景模型（可选 GPU/CUDA 加速）
 *
 * 维护一张“背景参考图”，用当前帧减背景得到前景掩码，供 ReID 特征提取剔除背景。
 * 学习策略：仅在“画面基本没人”时才更新背景，避免把行人写入背景。
 *
 * 前景掩码带“同一帧缓存”：同一帧只计算一次，避免每检测到一个人就重算一次全帧
 * 差分子+形态学（这是卡顿主因）。
 *
 * 若编译时启用 USE_OPENCV_CUDA 且运行时有可用 GPU，则前景分割在 GPU 上完成，
 * 大幅降低 CPU 占用（帧上传/掩码下载的开销远小于 CPU 形态学）。
 */
class BackgroundModel
{
public:
    BackgroundModel();

    void reset();
    void update(const cv::Mat &frame);
    bool ready() const;
    cv::Mat foregroundMask(const cv::Mat &frame) const;

    void setDiffThreshold(int t) { m_diffThresh = t; }
    void setLearnRate(double a)  { m_learnRate = a; }
    void setMinFrames(int n)     { m_minFrames = n; }
    /// 是否正在使用 GPU
    bool usingGpu() const { return m_useGpu; }

private:
    mutable QMutex m_mutex;
    cv::Mat m_bg;                 // CV_32FC3 背景参考（CPU）
    int    m_diffThresh{25};
    double m_learnRate{0.03};
    double m_fgGate{0.15};        // 前景占比低于此值才更新背景
    int    m_frames{0};
    int    m_minFrames{20};

    // 前景掩码缓存（按帧）
    mutable cv::Mat m_cachedMask;
    mutable const uchar *m_cachedData{nullptr};
    mutable int m_cachedW{0};
    mutable int m_cachedH{0};

    // 后端选择标志：无论是否编译 CUDA 均需存在，供 usingGpu() 查询
    bool m_useGpu{false};

#ifdef USE_OPENCV_CUDA
    cv::cuda::GpuMat m_bgGpu;     // GPU 背景参考 CV_32FC3
    cv::Ptr<cv::cuda::Filter> m_openFilter;
    cv::Ptr<cv::cuda::Filter> m_closeFilter;
    void initGpuFilters();
#endif
};

#endif // BACKGROUNDMODEL_H
