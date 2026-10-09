#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QCheckBox>
#include <QComboBox>
#include <QTableWidget>
#include <QElapsedTimer>
#include <QTimer>
#include <QStorageInfo>
#include <QDir>
#include <QFileInfo>
#include <QDateTime>
#include <QMutex>
#include <QMap>
#include <QSet>
#include <opencv2/opencv.hpp>
#include "cameracontroller.h"
#include "grabthread.h"
#include "app_config.h"
#include "imageview.h"
#include "saveimageworker.h"
#include "persondetector.h"
#include "persontracker.h"
#include "reidextractor.h"
#include "regionmanager.h"
#include "logger.h"
#include "offlinedetectform.h"
#include "settingform.h"
#include "visitorlog.h"
#include "zonemonitor.h"
#include "visitorregistry.h"
#include "httpuplink.h"
#include "flowaggregator.h"

namespace Ui {
class MainWindow;
}

// 淡出标签结构体，用于离开区域后 ID 持续显示
struct FadingLabel {
    int trackId;
    int visitorId;           // 关联的访客 ID
    cv::Rect lastBbox;
    QDateTime lastSeen;     // 最后一次在活跃 track 中出现的时间
    float alpha = 1.0f;     // 当前透明度 [0, 1]
};

// 访客身份记录（跨 track 持久化）
struct VisitorRecord {
    int visitorId;            // 唯一且稳定的访客 ID（同一人只有一个，持久化）
    cv::Mat reidFeature;      // ReID 特征（用于重入匹配）
    cv::Mat enterImage;       // 进入时的图像
    cv::Mat exitImage;        // 离开时的图像
    QDateTime enterTime;      // 进入时间
    QDateTime exitTime;       // 离开时间
    bool isInside;            // 当前是否在区域内
    int currentTrackId;       // 当前关联的跟踪 ID（-1 表示无）
    int regionId;             // 所在区域 ID
    // ===== 上车/下车方向（由 ZoneMonitor 的完整穿越判定）=====
    QString direction;        // 整体方向："boarding"(上车B→A), "alighting"(下车A→B), "turnback"(折返), ""(未确定)
    QString enterDirection;   // 进入方向（同上取值）
    QString exitDirection;    // 离开方向（同上取值）
    int entrySide{-1};        // 进入侧：0=A(顶端), 1=B(底端), -1=未确定
    int exitSide{-1};         // 离开侧：0=A(顶端), 1=B(底端), -1=未确定
    qint64 dwellSeconds{0};   // 停留时长（秒）= 离开时间 - 进入时间
    qint64 loggedCrossingMs{0};  // 已写入 person_records.csv 的穿越时刻（防重复写）
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    /**
     * @brief 删除本机历史上下车/进出记录（内存状态 + 磁盘记录 + 可选抓拍图片）
     *
     * 清空：访客记录、上车/下车计数、识别时间线、访客表格、
     *       visitor_events.csv / visitor_records.json；
     * 可选同时删除：ReID 抓拍目录 person_*、访客帧目录 visitor_V*。
     * 不删除待上行的离线积压（uplink_spool）。
     *
     * @param deleteImages 是否同时删除抓拍图片
     * @return 结果摘要文本（供 UI 展示/日志）
     */
    QString clearHistoryRecords(bool deleteImages);

private slots:
    void onToggleCapture();
    void onFrameGrabbed(QImage img);
    void onGrabThreadDeviceLost();
    void onTryReconnect();
    void onSaveCurrentFrame();
    void onAutoSaveToggled(bool checked);
    void onUpdateSystemTime();

    // 检测 / ReID 相关
    void onFrameProcessed(QImage img, QVector<TrackedPerson> persons);
    void onDrawRegion();
    void onClearRegions();
    void onPolygonCompleted(QVector<QPointF> points);
    void onOfflineDetect();
    void onSetParams();
    void onClearHistory();          // 删除历史上下车记录

protected:
    void closeEvent(QCloseEvent *event) override;

private:
    Ui::MainWindow *ui;

    // 相机
    CameraController m_camera;
    GrabThread* m_grabThread{nullptr};

    // 采集状态
    bool m_isCapturing{false};
    void updateCaptureStatus();

    // FPS 计算
    QElapsedTimer m_lastUiUpdate;
    QElapsedTimer m_fpsTimer;
    int m_fpsFrameCount{0};
    double m_currentFps{0.0};
    static constexpr int UI_UPDATE_INTERVAL_MS = 33; // ~30fps UI refresh

    // 当前帧（用于手动保存）
    QImage m_currentFrame;
    QMutex m_frameMutex;

    // 原始 BGR 帧缓存（用于 ReID 图像绘制和保存）
    cv::Mat m_currentRawFrame;
    QMutex m_rawFrameMutex;

    // 主界面 ImageView
    ImageView* m_imageView{nullptr};

    // 自动保存
    bool m_autoSaveEnabled{false};

    QTimer *m_sysTimer;
    QElapsedTimer m_elapsedTimer;
    QTimer *m_diskTimer;
    bool m_diskFull = false;

    // 后台图像保存
    SaveImageWorker *m_saveWorker{nullptr};
    QThread *m_saveThread{nullptr};

    // 自动重连
    QTimer *m_reconnectTimer{nullptr};
    int m_reconnectAttempts{0};
    static const int MAX_RECONNECT_ATTEMPTS = 60;
    static const int RECONNECT_INTERVAL_MS = 5000;

    // 检测 / ReID
    PersonDetector *m_detector{nullptr};
    PersonTracker *m_tracker{nullptr};
    ReIDExtractor *m_reidExtractor{nullptr};
    RegionManager *m_regionManager{nullptr};
    bool m_isDrawingRegion{false};
    int m_currentPersonCount{0};

    // 访客记录持久化
    VisitorLog *m_visitorLog{nullptr};
    QTimer *m_visitorSaveTimer{nullptr};
    static constexpr int VISITOR_SAVE_INTERVAL_MS = 60000; // 每60秒自动保存

    // 淡出标签管理
    QMap<int, FadingLabel> m_fadingLabels;
    static constexpr float FADING_DURATION_MS = 3000.0f; // 3秒淡出

    // ===== 访客身份管理（ReID 跨 track 匹配）=====
    QMap<int, VisitorRecord> m_visitors;   // visitorId -> VisitorRecord
    int m_nextVisitorId{1};
    // ReID 匹配阈值现由 m_config.reidMatchThreshold 控制（可在系统配置中调整）

    // trackId -> visitorId 映射
    QMap<int, int> m_trackToVisitor;

    // 进入方向
    EntryDirection m_currentEntryDirection{EntryDirection::TopToBottom};

    // ===== 全局访客身份注册表（跨轨迹稳定 ID，供跟踪/上下车/显示共用）=====
    VisitorRegistry *m_registry{nullptr};

    // ===== 垂直通道监控（上车/下车检测）=====
    ZoneMonitor *m_zoneMonitor{nullptr};
    int m_boardingCount{0};
    int m_alightingCount{0};

    // ===== 动恰客流时段聚合上行 =====
    HttpUplink *m_uplink{nullptr};
    FlowAggregator *m_aggregator{nullptr};
    void initUplink();

    void onCloseCamera();
    void updateCameraStatus(bool connected);
    void applySettings();
    void updateDiskSpaceStatus();
    void initSaveWorker();
    void saveReidImage(const cv::Mat &frame, int trackId, const QString &type);

    // 淡出 ID 标签辅助方法
    void updateFadingLabels(const QVector<TrackedPerson> &persons);
    void drawFadingIdOnCvMat(cv::Mat &frame, const FadingLabel &label);
    void drawActiveIdOnCvMat(cv::Mat &frame, const TrackedPerson &person);

    // 访客身份管理
    int matchOrCreateVisitor(const TrackedPerson &person, const cv::Mat &rawFrame,
                             int regionId, int imgW, int imgH);
    void updateVisitorExit(int visitorId, const cv::Mat &exitImage);

    /**
     * @brief 落地一次完整穿越记录（上车/下车/折返）
     *
     * 统一填充「同一 ID：进入时间+进入方向 / 离开时间+离开方向 / 停留时间」，
     * 写入 person_records.csv（同一穿越只写一次），并刷新记录表。
     */
    void applyCrossingRecord(int visitorId, const ZoneEvent &evt, const QString &direction);

    /// 方向文本：boarding→上车, alighting→下车, turnback→折返, 其它→--
    static QString directionText(const QString &direction);

    // 方向感知的穿越检测
    bool detectDirectionalEntry(const TrackedPerson &person, const DetectionRegion &region, int imgH);

    // 访客记录表格更新
    void updateVisitorTable();

    // 访客记录持久化
    void initVisitorLog();
    void saveVisitorRecords();
    void loadVisitorRecords();

    // 在图像上绘制方向箭头
    void drawDirectionArrowOnFrame(cv::Mat &frame, const DetectionRegion &region, int imgW, int imgH);

    // 保存区域内持续帧（按 visitor ID 命名）
    void saveVisitorFrame(const cv::Mat &rawFrame, const cv::Rect &personBbox, int visitorId);

    // 绘制引导提示
    void showDrawGuide(const QString &message);
    void clearDrawGuide();

    AppConfig m_config;
};

#endif // MAINWINDOW_H
