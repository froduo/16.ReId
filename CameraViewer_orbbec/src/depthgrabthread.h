#ifndef DEPTHGRABTHREAD_H
#define DEPTHGRABTHREAD_H

#include <QThread>
#include <QMutex>
#include <QImage>
#include <QString>
#include <atomic>
#include <opencv2/opencv.hpp>
#include "cameracontroller.h"

/**
 * @brief 深度流采集线程（独立于 RGB 主路）
 *
 * 以流自身帧率（约 30fps）持续拉取深度流，缓存最近一帧 CV_16U 毫米深度，
 * 并将伪彩图通过 depthFrameGrabbed 信号发给 UI。深度流失败不影响 RGB 主路，
 * 也不触发整体重连。
 */
class DepthGrabThread : public QThread
{
    Q_OBJECT
public:
    explicit DepthGrabThread(CameraController *cam,
                             const QString &channelName = "Depth",
                             QObject *parent = nullptr);

    void stop() { m_running.store(false); }
    QString channelName() const { return m_channelName; }

    /// 最近一帧深度图（CV_16U，单位毫米；仅当深度流为 16 位毫米深度时有效）
    cv::Mat lastDepth() const { QMutexLocker lk(&m_mutex); return m_lastDepth.clone(); }

signals:
    void depthFrameGrabbed(QImage img);   // 深度伪彩 QImage（用于深度视图显示）

protected:
    void run() override;

private:
    CameraController *m_camera;
    QString m_channelName;
    std::atomic<bool> m_running{true};
    int m_consecutiveFailCount{0};
    cv::Mat m_lastDepth;                 // 最近一帧深度（CV_16U, 毫米；16 位深度流）
    mutable QMutex m_mutex;              // 保护 m_lastDepth（采集线程写 / UI 线程读）
};

#endif // DEPTHGRABTHREAD_H
