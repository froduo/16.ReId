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

    /** 暂停采集/播放（仅暂停读取与处理，线程保持运行，恢复后从当前位置接续） */
    void pause();

    /** 恢复暂停的采集/播放 */
    void resume();

    /** 当前是否处于暂停状态 */
    bool isPaused() const { return m_paused.load(); }

    /** 是否正在处理一帧（模型热重载前等待其空闲） */
    bool isProcessing() const { return m_processing.load(); }

    /** 获取通道名称 */
    QString channelName() const { return m_channelName; }

    // 检测管线设置
    void setDetector(PersonDetector *detector);
    void setTracker(PersonTracker *tracker);
    void setReidExtractor(ReIDExtractor *extractor);
    void setBackgroundModel(BackgroundModel *model) { m_bgModel = model; }
    /// UI 是否繁忙（用于“只投递最新帧”，避免显示队列积压导致画面滞后）
    void setUiBusy(bool busy) { m_uiBusy.store(busy); }
    /// 视频文件倍速播放：仅控制读取/处理节奏（不改变检测流程），
    /// 实际速度上限为检测程序的最大处理速率。
    void setVideoSpeed(double speed);

    /**
     * @brief 检测降频：每 N 帧执行一次完整检测，中间帧仅卡尔曼外推（不跑 TensorRT、不提取 ReID）。
     *
     * 用于倍速播放时把逐帧检测的硬开销（实测 ~7ms/帧）分摊到多帧，从而逼近目标倍速。
     * N=1 表示逐帧检测（实时采集默认行为）。跳过检测的帧：tracker->update(空检测) 仅做
     * 预测外推，其输出的 TrackedPerson::detected=false，下游 ZoneMonitor 会忽略预测框，
     * 因此不会把外推位置误判为穿越。
     */
    void setDetectEveryN(int n) { m_detectEveryN = qMax(1, n); }
    int detectEveryN() const { return m_detectEveryN; }

    /**
     * @brief 自适应检测间隔：无乘客时降频检测，检测到乘客立即切回逐帧
     *
     * 乘客从「出现」到「跨门线」通常有 2~5 秒，而空闲跳帧间隔仅约 0.4 秒，
     * 因此一旦再次检测到乘客立即切回逐帧，即可完整覆盖穿越轨迹，无需回退/seek。
     * @param idleSkipEveryN      无乘客时的检测间隔（帧）
     * @param idleFramesBeforeSkip 连续无乘客多少帧后才进入跳帧
     */
    void setIdleSkip(int idleSkipEveryN, int idleFramesBeforeSkip) {
        m_idleSkipEveryN = qMax(1, idleSkipEveryN);
        m_idleFramesBeforeSkip = qMax(1, idleFramesBeforeSkip);
    }

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

    /// 人物片段录制降频：每 N 帧写 1 帧（1=每帧，越大越省视频编码开销）
    void setPersonClipEveryN(int n) { m_clipEveryN = qMax(1, n); }

signals:
    void deviceLost();                              // 检测到相机掉线
    void videoFinished();                           // 本地视频文件读取完毕
    void frameGrabbed(QImage img);                  // RGB QImage（用于显示）
    void rawFrameGrabbed(const cv::Mat &frame);     // 原始 BGR cv::Mat（用于保存）
    void frameProcessed(const cv::Mat &bgrFrame, const QVector<TrackedPerson> &persons); // 检测处理后的 BGR 帧和跟踪结果
    /// 每帧检测结果（不背压）：倍速播放时主线程仍需逐帧处理上下车判定，不能跳帧。
    /// imgW/imgH 为当前帧尺寸（每帧准确）；原始帧由 rawFrameGrabbed 每 5 帧刷新一次。
    void personsDetected(int imgW, int imgH, const QVector<TrackedPerson> &persons);

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
    std::atomic<bool> m_paused{false};   // 暂停标志：true 时循环仅休眠，不读取帧
    std::atomic<bool> m_processing{false}; // 正在处理一帧（模型热重载前等待其空闲）
    double m_videoSpeed{1.0};        // 视频文件倍速播放（1x~6x，上限=检测最大速率）
    int m_detectEveryN{1};           // 检测降频间隔（帧），1=逐帧检测
    int m_idleSkipEveryN{10};        // 无乘客时的检测间隔（帧），空场景快速推进
    int m_idleFramesBeforeSkip{25};  // 连续无乘客多少帧后才进入跳帧
    int m_lastPersonFrame{0};        // 最近一次检测到乘客的帧号（自适应调度用）
    QElapsedTimer m_paceTimer;       // 倍速节奏计时器
    QVector<TrackedPerson> m_lastTrackedPersons;

    // ===== 人物片段录制 =====
    bool m_clipEnabled{false};
    QString m_clipDir;
    int m_clipGapMs{3000};
    double m_clipFps{25.0};
    int m_clipEveryN{1};        // 录制降频：每 N 帧写 1 帧（1=每帧）
    int m_clipTick{0};          // 写帧节流计数
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
