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
#include <QJsonArray>
#include <QMutex>
#include <QMap>
#include <QSet>
#include <QVector>
#include <opencv2/opencv.hpp>
#include "cameracontroller.h"
#include "grabthread.h"
#include "app_config.h"
#include "imageview.h"
#include "saveimageworker.h"
#include "persondetector.h"
#include "persontracker.h"
#include "reidextractor.h"
#include "reiddeep.h"
#include "regionmanager.h"
#include "logger.h"
#include "offlinedetectform.h"
#include "settingform.h"
#include "visitorlog.h"
#include "zonemonitor.h"
#include "passengerbook.h"
#include "identitymatchworker.h"
#include "tripdatastore.h"
#include "personfolder.h"
#include "foldermatchworker.h"
#include "version.h"
#include "backgroundmodel.h"
#include <QThread>

class QNetworkAccessManager;
class StatusWebServer;

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
    int visitorId;            // 唯一访客 ID（持久化）
    cv::Mat reidFeature;      // ReID 特征（用于重入匹配）
    cv::Mat enterImage;       // 进入时的图像
    cv::Mat exitImage;        // 离开时的图像
    QDateTime enterTime;      // 进入时间
    QDateTime exitTime;       // 离开时间
    bool isInside;            // 当前是否在区域内
    int currentTrackId;       // 当前关联的跟踪 ID（-1 表示无）
    int regionId;             // 所在区域 ID
    QString direction;        // 方向："boarding"(上车), "alighting"(下车), ""(未确定)
};

// 下车身份比对任务（后台线程处理期间暂存于主窗口）
struct PendingAlight {
    ZoneEvent evt;              // 原始下车事件（含整段图像/特征）
    QDateTime alightTime;       // 下车时间
    int fallbackId{-1};         // 比对未命中时的跟踪身份回退 ID
    QVector<cv::Mat> feats;     // 探测特征序列（已组装）
};

// 人员进出记录行：一次上/下车事件，乘客ID 与保存图片目录 person_N_日期 对应
struct PersonEventRow {
    int personId{-1};           // person 序号
    QString status;             // "上车" / "下车"
    QDateTime time;             // 上/下车发生时间
    int matchedPersonId{-1};    // 匹配到的对方 person 序号（未匹配=-1）
    float similarity{0.0f};
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    /// 自动化/调试：直接加载视频文件并按指定倍速开始播放（跳过文件对话框）
    void autoPlayVideo(const QString &path, double speed);

private slots:
    void onToggleCapture();
    void onSelectVideo();
    void onTogglePlay();
    void onFrameGrabbed(QImage img);
    void onGrabThreadDeviceLost();
    void onTryReconnect();
    void onSaveCurrentFrame();
    void onAutoSaveToggled(bool checked);
    void onUpdateSystemTime();

    // 检测 / ReID 相关
    void onFrameProcessed(const cv::Mat &bgrFrame, QVector<TrackedPerson> persons);
    /// 每帧检测结果处理（上下车判定 / 图像归档，不依赖显示刷新）
    void onPersonsDetected(int imgW, int imgH, QVector<TrackedPerson> persons);
    void onDrawRegion();
    void onClearCounts();
    void onPolygonCompleted(QVector<QPointF> points);
    void onOfflineDetect();
    void onSetParams();

    // 关于 / 版本信息、检查更新
    void onAbout();
    void onCheckUpdate();

protected:
    void closeEvent(QCloseEvent *event) override;

private:
    Ui::MainWindow *ui;

    // 相机
    CameraController m_camera;
    GrabThread* m_grabThread{nullptr};

    // 采集状态
    bool m_isCapturing{false};
    bool m_videoMode{false};        // 当前是否处于「视频文件检测」模式
    double m_videoSpeed{1.0};       // 视频文件倍速播放（1x~6x，实际上限=检测最大速率）
    bool m_playPaused{false};       // 视频文件播放是否处于「暂停」状态（停止播放后接续播放）
    bool m_autoPlayMode{false};     // 自动化模式（--autoplay）：视频播放完成后自动退出
    void updateCaptureStatus();
    /// 刷新「开始播放/停止播放」按钮文字与可用状态
    void updatePlayButton();

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
    /// 深度 ReID（OSNet 现场微调模型）；启用后手工 ReIDExtractor 自动改走深度特征
    DeepReID *m_deepReid{nullptr};
    /// 当前已加载的检测模型路径（用于判断「修改模型路径后是否需要热重载」）
    QString m_loadedDetectorPath;
    BackgroundModel *m_bgModel{nullptr};   // 固定摄像头背景模型（用于背景扣除）
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
    QVector<PersonEventRow> m_personEventRows;   // 上下车事件行（表格用）
    QSet<int> m_activePersonIds;
                 // 当前仍在画面中的 person 序号（用于关闭消失乘客的视频）
    int m_nextVisitorId{1};
    // ReID 匹配阈值现由 m_config.reidMatchThreshold 控制（可在系统配置中调整）

    // trackId -> visitorId 映射
    QMap<int, int> m_trackToVisitor;

    // 进入方向
    EntryDirection m_currentEntryDirection{EntryDirection::TopToBottom};

    // ===== 垂直通道监控（上车/下车检测）=====
    ZoneMonitor *m_zoneMonitor{nullptr};
    int m_boardingCount{0};
    int m_alightingCount{0};

    // ===== 下车身份比对后台线程 =====
    QThread *m_matchThread{nullptr};
    IdentityMatchWorker *m_matchWorker{nullptr};
    quint64 m_matchJobCounter{0};
    QMap<quint64, PendingAlight> m_pendingAlights;

    /// 下车比对「攒批」缓冲：把短时间窗口内的多个下车事件合并为一批，
    /// 在后台线程做「动态加权 + 匈牙利全局匹配」，避免逐个贪心错配。
    QVector<quint64> m_alightBatchJobs;
    QTimer *m_alightBatchTimer{nullptr};
    static constexpr int ALIGHT_BATCH_WINDOW_MS = 300;
    void flushAlightBatch();

    /// 下车比对结果结算（在主线程执行：命中/回退/异常判定 + 保存 + 记录）
    void applyAlightingMatch(const PendingAlight &pend, int bestId,
                             float similarity, bool matched);

    // ===== 按乘客分目录的图像归档 + 后台整目录全图比对 =====
    /// 上车目录登记（供下车目录做时间过滤 + 未匹配过滤）
    struct BoardFolderEntry {
        int personId{-1};
        QString dir;
        qint64 timeMs{0};
        bool matched{false};
    };
    /// 一次「下车目录 vs 上车目录」后台比对任务的暂存信息
    struct FolderMatchPending {
        int alightPersonId{-1};
        QString alightDir;
        QDateTime alightTime;
    };

    PersonFolder *m_personFolder{nullptr};
    QThread *m_folderMatchThread{nullptr};
    FolderMatchWorker *m_folderMatchWorker{nullptr};
    quint64 m_folderMatchJobCounter{0};
    QVector<BoardFolderEntry> m_boardFolders;      // 上车目录登记表
    QMap<quint64, FolderMatchPending> m_folderMatchPending;
    QString m_sessionStatePath;                    // 会话状态持久化文件路径

    // ===== Web 状态服务（手机端查看上下车状态）=====
    StatusWebServer *m_webServer{nullptr};
    QJsonArray m_recentMatches;                    // 最近匹配记录（board/alight/similarity）
    void initWebServer();
    void pushWebStatus();

    void initPersonFolder();
    /// 会话状态持久化：人员进出记录 + 上车目录登记 + 计数（重启后恢复，人工清零才归零）
    void saveSessionState();
    void loadSessionState();
    /// 帧处理时：乘客出现即分配 person_N 并持续保存原始整帧（预热 + 节流）
    void savePersonAppearanceFrames(const QVector<TrackedPerson> &persons,
                                    const cv::Mat &rawFrame);
    /// 后台全图比对结果结算：写 marker + 标记去重 + 落库
    void onFolderMatchFinished(quint64 jobId, int bestPersonId, QString bestDir,
                               float similarity, bool matched);

    // ===== 乘车数据存储（每站上下车人数 + 车厢内人数；SQLite 默认，可扩展 PostgreSQL）=====
    TripDataStore *m_tripStore{nullptr};
    QString m_currentTripId;          // 当前行程 ID（一次采集会话）
    QString m_currentStationId;       // 当前站点 ID
    QString m_currentStationName;     // 当前站点名称
    int m_onboardCount{0};            // 当前车厢内人数
    void initTripStore();
    /**
     * @brief 切换当前站点（后续上下车将计入该站；stop_stats 按 (trip, station) 聚合）
     */
    void setCurrentStation(const QString &stationId, const QString &stationName);
    /// 更新状态栏/标签上的车厢内人数显示
    void updateOnboardLabel();

    // ===== 深度 ReID（OSNet 现场微调模型）=====
    void initDeepReid();
    /// 当前生效的 ReID 匹配阈值（深度后端→DeepMatchThreshold，否则→MatchThreshold）
    float effectiveReidThreshold() const;
    /// 当前生效的检测置信度阈值（head 模式用更高的 HeadConfThreshold 抑制空场景误检）
    float effectiveDetectorConf() const;
    /// 依据 detectorModelKind 应用「尺寸/长宽比」过滤参数（person / head）
    void applyDetectorFilters(PersonDetector *det) const;

    // ===== 乘客台账（上车登记 + 下车图像比对）=====
    PassengerBook *m_passengerBook{nullptr};
    void initPassengerBook();
    /**
     * @brief 保存乘客上/下车抓拍图片，文件名：P{ID}_上车_HHmmss / P{ID}_下车_HHmmss
     * @return 保存后的文件路径（失败返回空）
     */
    QString savePassengerEventImage(const cv::Mat &frame, const cv::Rect &bbox,
                                    int passengerId, bool boarding, const QDateTime &time);
    /**
     * @brief 保存“进入检测区域 → 离开检测区域”的整段序列图片
     *        目录：save_images/reid/passenger/P{ID}_{上车|下车}_{时间}/
     * @return 保存后的目录路径（失败返回空）
     */
    QString savePassengerEventImages(const QVector<cv::Mat> &crops,
                                     const QVector<cv::Rect> &boxes,
                                     int passengerId, bool boarding, const QDateTime &time);
    /**
     * @brief 保存「进入检测区域 → 离开检测区域」的整段视频（mp4，失败回退 avi）
     *        文件：save_images/reid/passenger/P{ID}_{上车|下车}_{时间}.mp4
     *        passengerId<=0 时输出「异常乘客_{上车|下车}_{时间}.mp4」
     * @param frames     整帧序列（ZoneMonitor 采样，约 960 宽）
     * @param durationMs 该段真实耗时，用于推算视频帧率（<=0 时按 12fps）
     * @return 保存后的文件路径（失败返回空）
     */
    QString savePassengerEventVideo(const QVector<cv::Mat> &frames,
                                    int passengerId, bool boarding,
                                    const QDateTime &time, qint64 durationMs);
    /**
     * @brief 保存“异常乘客下车”的整段图片（无法确定是哪位乘客下车时）
     *        目录：save_images/reid/passenger/异常乘客_下车_{时间}/
     */
    QString saveAbnormalAlightImages(const QVector<cv::Mat> &crops,
                                     const QVector<cv::Rect> &boxes,
                                     const QDateTime &time);

    void onCloseCamera();
    /// 打开本地视频文件并切换到视频模式（onSelectVideo 与 autoPlayVideo 共用）
    bool openVideoFileForPlay(const QString &path);
    /// 启动检测采集管线（RTSP 或视频文件通用）
    void startDetectionPipeline();
    /// 切换 PersonFolder 保存根目录到视频对应文件夹
    void switchPersonFolderToVideo(const QString &videoPath);
    /// 恢复 PersonFolder 保存根目录（视频检测结束后）
    void restorePersonFolderFromVideo();
    void updateCameraStatus(bool connected);
    void applySettings();
    /// 检测/ReID 模型路径变化时热重载（修改模型路径立即生效，无需重启）
    void reloadModelsIfChanged();
    void updateDiskSpaceStatus();
    void initSaveWorker();
    void saveReidImage(const cv::Mat &frame, int trackId, const QString &type);

    /**
     * @brief 选择检测模型路径：优先轻量 ONNX 小模型（若存在），否则用配置的
     *        modelPath（通常是 TensorRT engine）。多个候选目录依次查找：
     *        配置原路径 / 程序目录 / 源码目录(开发环境回退)
     */
    QString pickDetectorModelPath() const;

    /**
     * @brief 异步下载检测模型（ONNX）到指定路径，并做「能否被 OpenCV DNN 读取」校验
     * @param url ONNX 直链
     * @param targetPath 目标路径（相对路径按程序目录解析）
     */
    void downloadDetectorModel(const QString &url, const QString &targetPath);

    // 淡出 ID 标签辅助方法
    void updateFadingLabels(const QVector<TrackedPerson> &persons);
    void drawFadingIdOnCvMat(cv::Mat &frame, const FadingLabel &label);
    void drawActiveIdOnCvMat(cv::Mat &frame, const TrackedPerson &person);
    /// 是否绘制该轨迹的检测框：过滤卡尔曼外推的“幽灵框”，避免同一人出现两个框
    bool shouldDrawTrack(const TrackedPerson &person) const;

    // 访客身份管理
    int matchOrCreateVisitor(const TrackedPerson &person, const cv::Mat &rawFrame, int regionId);
    void updateVisitorExit(int visitorId, const cv::Mat &exitImage);

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

    // 依据当前区域刷新 ImageView 上的「区域多边形 + 虚拟门线」叠加显示
    void refreshRegionOverlays();

    // 保存区域内持续帧（按 visitor ID + 本次穿越段 归档，命名区分上车/下车）
    void saveVisitorFrame(const cv::Mat &rawFrame, const cv::Rect &personBbox,
                          int visitorId, int trackId);

    /**
     * @brief 归档某访客「本次穿越」的逐帧图像目录
     *
     * Inside 期间图像先写入 visitor_V{id}/_pending_HHmmss_t{track}/，
     * 上/下车事件确定方向后重命名为 visitor_V{id}/上车_{时间}/ 或 /下车_{时间}/，
     * 从而保证同一目录不会混入不同乘客，且直接可区分上/下车。
     *
     * @param visitorId 访客 ID（<=0 时忽略）
     * @param boarding  true=上车，false=下车
     * @param time      事件时间（用于目录命名）
     */
    void finalizeVisitorSegment(int visitorId, bool boarding, const QDateTime &time);

    // 绘制引导提示
    void showDrawGuide(const QString &message);
    void clearDrawGuide();

    // ===== 版本 / 更新 =====
    /// 版本号格式化为 "vX.Y.Z"，并附带构建日期
    QString appVersionText() const;
    /// 解析版本字符串为数值（MAJOR*10000+MINOR*100+PATCH），失败返回 -1
    static int parseVersionCode(const QString &ver);
    /// 检查更新：读取远端 manifest(updateUrl) 或本地 update.json 并比较版本
    void checkForUpdates();
    QNetworkAccessManager *m_netMgr{nullptr};

    // ===== 访客逐帧图像归档状态 =====
    QMap<int, QString> m_visitorPendingDir;  // visitorId -> 本次穿越的待定目录（绝对路径）
    QMap<int, int>     m_visitorFrameTick;   // visitorId -> 采样节拍（节流用）
    QMap<int, int>     m_visitorSegmentSeq;  // visitorId -> 本次穿越已保存帧序号

    AppConfig m_config;
};

#endif // MAINWINDOW_H
