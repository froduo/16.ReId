#ifndef ZONEMONITOR_H
#define ZONEMONITOR_H

#include <QObject>
#include <QMap>
#include <QVector>
#include <QPointF>
#include <QDateTime>
#include <opencv2/opencv.hpp>
#include "persontracker.h"
#include "regionmanager.h"
#include "reidextractor.h"
#include "visitorregistry.h"

/**
 * @brief 区域穿越事件类型
 */
enum class ZoneEventType {
    Boarding,    // 上车：从B端（底端）进入 → A端（顶端）离开
    Alighting    // 下车：从A端（顶端）进入 → B端（底端）离开
};

/**
 * @brief 区域穿越事件记录
 */
struct ZoneEvent {
    int visitorId;              // 访客ID（全局稳定身份，来自 VisitorRegistry）
    int trackId;                // 跟踪ID
    int regionId;               // 区域ID
    ZoneEventType type;         // 事件类型
    QDateTime enterTime;        // 进入时间
    QDateTime exitTime;         // 离开时间
    int entrySide;              // 进入侧：0=A（顶端），1=B（底端）
    int exitSide;               // 离开侧：0=A（顶端），1=B（底端）
    cv::Mat enterImage;         // 进入时图像
    cv::Mat exitImage;          // 离开时图像
};

/**
 * @brief 通道监控区域定义
 */
struct ChannelZone {
    int regionId;               // 关联的区域ID
    float topY;                 // 区域顶端Y坐标（归一化），对应A点
    float bottomY;              // 区域底端Y坐标（归一化），对应B点
    float centerX;              // 区域中心X坐标（归一化）
};

/**
 * @brief 通道内人员的追踪状态
 */
enum class ChannelTrackState {
    Outside,        // 在通道外
    Entering,       // 正在进入（确认中）
    Inside,         // 已确认在通道内
    Exiting         // 正在离开（确认中）
};

/**
 * @brief 通道内人员的详细追踪信息
 */
struct ChannelTrackedPerson {
    ChannelTrackState state{ChannelTrackState::Outside};
    int confirmFrames{0};       // 状态确认帧计数
    int entrySide{-1};          // 进入侧：0=A（顶端），1=B（底端），-1=未确定
    int exitSide{-1};           // 离开侧：0=A（顶端），1=B（底端），-1=未确定
    QDateTime enterTime;        // 进入时间
    int visitorId{-1};          // 关联的访客ID（全局稳定身份）
    cv::Mat enterImage;         // 进入时的图像
    int regionId{-1};           // 所在区域ID
};

/**
 * @brief 垂直通道人员运动方向监测系统
 *
 * 针对“同一个人移动时被分配多个 ID、上车/下车被拆成多条”的问题，做了三点强化：
 * 1. 身份来自共享的 VisitorRegistry（跨轨迹唯一），不再各自编号；
 * 2. **按人续接穿越**：轨迹短暂丢失（换 ID）后重新匹配到同一人，会续接原有进入记录，
 *    最终只产生一次上车/下车；
 * 3. 穿越完成后按人做时间窗去抖，避免重复计数。
 */
class ZoneMonitor : public QObject
{
    Q_OBJECT
public:
    explicit ZoneMonitor(QObject *parent = nullptr);

    /// 设置共享身份注册表（跨轨迹稳定 ID）；未设置时退化为本地编号
    void setVisitorRegistry(VisitorRegistry *registry);

    void setReidExtractor(ReIDExtractor *extractor);
    void setReidMatchThreshold(float threshold);
    void setConfirmFrames(int frames);

    /// 设置通道区域（从 RegionManager 的多边形区域自动计算 A/B 端点）
    void setChannelZones(const QVector<DetectionRegion> &regions);

    void processFrame(QVector<TrackedPerson> &persons,
                      const cv::Mat &rawFrame,
                      RegionManager *regionManager,
                      int imgW, int imgH);

    int boardingCount() const { return m_boardingCount; }
    int alightingCount() const { return m_alightingCount; }
    QVector<ZoneEvent> events() const { return m_events; }

    void reset();

signals:
    void boardingEvent(const ZoneEvent &event);
    void alightingEvent(const ZoneEvent &event);
    void zoneCrossingEvent(const ZoneEvent &event);

private:
    /// 待完成的穿越（按 personId 保存，用于轨迹丢失后“续接”）
    struct PendingCrossing {
        int personId{-1};
        int regionId{-1};
        int entrySide{-1};
        QDateTime enterTime;
        cv::Mat enterImage;
        qint64 lastTouchMs{0};
    };

    VisitorRegistry *m_registry{nullptr};
    ReIDExtractor *m_reidExtractor{nullptr};
    float m_reidMatchThreshold{0.70f};
    int m_confirmFrames{3};

    QMap<int, ChannelTrackedPerson> m_channelTracks;  // trackId -> 通道状态
    QMap<int, int> m_trackToVisitor;                  // trackId -> personId
    QMap<int, PendingCrossing> m_pendingCrossings;    // personId -> 待完成穿越
    QMap<int, qint64> m_lastCrossingMsByPerson;       // personId -> 上次完成穿越时刻

    QVector<ChannelZone> m_zones;
    QVector<ZoneEvent> m_events;
    int m_boardingCount{0};
    int m_alightingCount{0};
    int m_nextVisitorId{1};                           // 无注册表时的回退编号

    // 参数
    static constexpr qint64 PENDING_GRACE_MS  = 8000;  // 穿越续接宽限
    static constexpr qint64 CROSSING_DEDUP_MS = 2500;  // 同人重复穿越去抖

    int determineSide(float normY, const ChannelZone &zone) const;
    int determineEntrySide(float prevNormY, float currNormY, const ChannelZone &zone) const;
    int determineExitSide(float prevNormY, float currNormY, const ChannelZone &zone) const;

    /// 解析稳定 personId（优先使用共享注册表）
    int resolvePersonId(const TrackedPerson &person, const cv::Mat &rawFrame,
                        const cv::Point2f &normPos, qint64 nowMs);

    /// 过期待完成穿越清理（并解除 inside 标记）
    void expirePendingCrossings(qint64 nowMs);
};

#endif // ZONEMONITOR_H
