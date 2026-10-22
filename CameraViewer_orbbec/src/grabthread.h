#ifndef GRABTHREAD_H
#define GRABTHREAD_H

#include <QThread>
#include <QImage>
#include <QString>
#include <atomic>
#include <opencv2/opencv.hpp>
#include "cameracontroller.h"
#include "frameslot.h"

/**
 * @brief RGB 视频流采集线程（仅拉流 + 显示原始帧 + 投递最新帧给检测线程）
 *
 * 与深度流采集线程（DepthGrabThread）完全独立，各自以流自身帧率（约 30fps）
 * 拉流显示，互不阻塞。检测管线运行在独立的 DetectThread 中，因此本线程的
 * 实际拉流帧率不受 YOLO/ReID 耗时影响。
 */
class GrabThread : public QThread
{
    Q_OBJECT
public:
    explicit GrabThread(CameraController *cam,
                        LatestFrameSlot *slot,
                        const QString &channelName = "Camera",
                        QObject *parent = nullptr);

    void stop() { m_running.store(false); }
    QString channelName() const { return m_channelName; }

signals:
    void deviceLost();                              // 检测到相机掉线
    void videoFinished();                           // 本地视频文件读取完毕
    void frameGrabbed(QImage img);                  // RGB QImage（用于显示/FPS）

protected:
    void run() override;

private:
    CameraController *m_camera;
    LatestFrameSlot *m_slot;
    QString m_channelName;
    std::atomic<bool> m_running{true};
    int m_consecutiveFailCount{0};
    // 连续失败阈值：配合 FFmpeg stimeout(5s)，连续 5 次读超时（约 25~30 秒）
    // 即判定相机掉线并触发重连，缩短漏检窗口。
    static const int MAX_FAIL_COUNT = 5;
};

#endif // GRABTHREAD_H
