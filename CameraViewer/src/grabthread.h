#ifndef GRABTHREAD_H
#define GRABTHREAD_H

#include <QThread>
#include <QElapsedTimer>
#include <QString>
#include <QImage>
#include <QVector>
#include <QDateTime>
#include <atomic>
#include <opencv2/opencv.hpp>
#include "cameracontroller.h"
#include "persondetector.h"
#include "persontracker.h"
#include "reidextractor.h"
#include "backgroundmodel.h"

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
    void setBackgroundModel(BackgroundModel *model) { m_bgModel = model; }
    /// UI 是否繁忙（用于“只投递最新帧”，避免显示队列积压导致画面滞后）
    void setUiBusy(bool busy) { m_uiBusy.store(busy); }

    /**
     * @brief 人物片段录制：画面出现人即开始录制，连续无人超过 gapMs 后结束本段
     *
     * 每段单独保存为一个视频文件，文件名含起止时间：
     *   人物片段_yyyyMMdd_HHmmss-HHmmss.mp4（mp4 编码不可用时回退 avi/MJPG）
     * 帧率在首段使用 fps，之后按“实际写入帧数 / 真实耗时”自校准，保证播放时长准确。
     *
     * @param enabled 是否启用
     * @param dir     保存目录（相对路径按程序目录解析）
     * @param gapMs   连续无人超过该时长即结束当前片段（毫秒，最小 500）
     * @param fps     起始帧率
     */
    void setPersonClipRecording(bool enabled, const QString &dir, int gapMs, int fps);

signals:
    void deviceLost();                              // 检测到相机掉线
    void videoFinished();                           // 本地视频文件读取完毕
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
    // 连续失败阈值：配合 FFmpeg stimeout(5s)，连续 5 次读超时（约 25~30 秒）
    // 即判定相机掉线并触发重连，缩短漏检窗口。
    static const int MAX_FAIL_COUNT = 5;

    // 检测管线
    PersonDetector *m_detector{nullptr};
    PersonTracker *m_tracker{nullptr};
    ReIDExtractor *m_reidExtractor{nullptr};
    BackgroundModel *m_bgModel{nullptr};
    std::atomic<bool> m_uiBusy{false};
    QVector<TrackedPerson> m_lastTrackedPersons;

    // ===== 人物片段录制 =====
    bool m_clipEnabled{false};
    QString m_clipDir;
    int m_clipGapMs{3000};
    double m_clipFps{25.0};
    cv::VideoWriter m_clipWriter;
    QString m_clipFilePath;      // 当前片段的实际文件路径（用于结束后重命名）
    cv::Size m_clipFrameSize{0, 0};  // 当前片段的分辨率（变化时结束并重开）
    QDateTime m_clipStart;       // 当前片段开始时间
    qint64 m_lastPersonMs{0};    // 最近一次检测到人的时刻（毫秒）
    bool m_clipActive{false};
    int m_clipFrames{0};

    /// 每帧调用：按“有人”状态开启/写入/结束片段
    void updatePersonClip(const cv::Mat &frame, bool personPresent);
    /// 结束当前片段：关闭写入器、补上结束时间重命名、自校准 fps
    void closePersonClip();
};

#endif // GRABTHREAD_H
