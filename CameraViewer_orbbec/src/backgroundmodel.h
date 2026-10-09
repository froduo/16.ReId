#ifndef BACKGROUNDMODEL_H
#define BACKGROUNDMODEL_H

#include <QMutex>
#include <opencv2/opencv.hpp>

/**
 * @brief 固定摄像头背景模型（CPU 实现）
 *
 * 维护一张“背景参考图”，用当前帧减背景得到前景掩码，供 ReID 特征提取剔除背景。
 * 学习策略：仅在“画面基本没人”时才更新背景，避免把行人写入背景。
 *
 * 前景掩码带“同一帧缓存”：同一帧只计算一次，避免每检测到一个人就重算一次全帧
 * 差分子+形态学（这是卡顿主因）。
 *
 * 注意：本类已【取消 CUDA/GPU 加速】，前景分割统一在 CPU 上完成。
 *       保留 usingGpu() 接口仅为兼容调用方（恒返回 false）。
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
    /// 背景/前景分割固定使用 CPU（已取消 CUDA 加速），恒返回 false。
    /// 保留该接口以兼容调用方（如 MainWindow 的启动日志）。
    bool usingGpu() const { return false; }

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
};

#endif // BACKGROUNDMODEL_H
