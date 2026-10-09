#ifndef GRABTHREAD_H
#define GRABTHREAD_H

#include <QThread>
#include <QElapsedTimer>
#include <QString>
#include <QImage>
#include <QVector>
#include <opencv2/opencv.hpp>
#include "cameracontroller.h"
#include "persondetector.h"
#include "persontracker.h"
#include "reidextractor.h"

/**
 * @brief RTSP 视频流采集线程（含检测管线）
 *
 * 从 CameraController 持续抓取帧，发出 frameGrabbed(QImage) 和 rawFrameGrabbed(cv::Mat) 信号。
 * 支持断线检测和自动重连。可选集成 YOLO 检测、IoU 跟踪和 ReID 特征提取。
 */
class GrabThread : public QThread
{
    Q_OBJECT
public:
    explicit GrabThread(CameraController* cam,
                        const QString &channelName = "Camera",
                        QObject *parent = nullptr);

    void stop();

    /** 获取通道名称 */
    QString channelName() const { return m_channelName; }

    // 检测管线设置
    void setDetector(PersonDetector *detector);
    void setTracker(PersonTracker *tracker);
    void setReidExtractor(ReIDExtractor *extractor);

signals:
    void deviceLost();                              // 检测到相机掉线
    void frameGrabbed(QImage img);                  // RGB QImage（用于显示）
    void rawFrameGrabbed(const cv::Mat &frame);     // 原始 BGR cv::Mat（用于保存）
    void frameProcessed(const QImage &img, const QVector<TrackedPerson> &persons); // 检测处理后的帧和跟踪结果

protected:
    void run() override;

private:
    bool m_running;
    CameraController* m_camera;
    QString m_channelName;
    int m_consecutiveFailCount{0};
    static const int MAX_FAIL_COUNT = 300; // 连续失败阈值（约15秒）

    // 检测管线
    PersonDetector *m_detector{nullptr};
    PersonTracker *m_tracker{nullptr};
    ReIDExtractor *m_reidExtractor{nullptr};
    QVector<TrackedPerson> m_lastTrackedPersons;
};

#endif // GRABTHREAD_H
