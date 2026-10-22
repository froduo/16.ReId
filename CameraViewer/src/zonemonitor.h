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

/**
 * @brief 穿越事件类型
 *
 * 方向约定（基于虚拟“门线”）——【相机装反 / 视角反转后的新定义】：
 *  - 乘客中心点从门线「外侧(A/上侧)」跨到「内侧(B/下侧)」= **下车**（A→B，等价于图像中从上往下）；
 *  - 乘客中心点从门线「内侧(B/下侧)」跨到「外侧(A/上侧)」= **上车**（B→A，等价于图像中从下往上）。
 *  若区域的 entryDirection == BottomToTop，则上述语义整体取反。
 */
enum class ZoneEventType {
    Boarding,    // 上车
    Alighting    // 下车
};

/**
 * @brief DoorCounter v6 的四层容错恢复机制
 *
 * 乘客上下车时可能因遮挡、转身、漏检造成轨迹中断，仅靠“实时判断”会漏检。
 * 因此按以下四层依次兜底（值越小优先级越高、置信度越高）：
 *   1) Realtime 实时判断   ：持续跟踪中心点相对门线的侧别转换；
 *   2) Inertia  惯性滑行   ：目标在门线处短暂丢失时，按速度外推判断是否已跨越；
 *   3) Stitch   轨迹拼接   ：被遮挡等原因打断的轨迹，用速度/位置预测重新连接续接；
 *   4) Fallback 兜底策略   ：处理“在门线一侧出现、另一侧消失”的极端情况。
 */
enum class RecoveryLayer {
    Realtime = 0,   // L1 实时判断
    Inertia  = 1,   // L2 惯性滑行
    Stitch   = 2,   // L3 轨迹拼接
    Fallback = 3    // L4 兜底策略
};

inline QString recoveryLayerName(RecoveryLayer l)
{
    switch (l) {
    case RecoveryLayer::Realtime: return QStringLiteral("L1实时判断");
    case RecoveryLayer::Inertia:  return QStringLiteral("L2惯性滑行");
    case RecoveryLayer::Stitch:   return QStringLiteral("L3轨迹拼接");
    case RecoveryLayer::Fallback: return QStringLiteral("L4兜底策略");
    }
    return QStringLiteral("未知");
}

/**
 * @brief 门线穿越事件记录
 */
struct ZoneEvent {
    int visitorId;              // 访客ID（ReID）
    int trackId;                // 跟踪ID
    int regionId;               // 区域ID
    ZoneEventType type;         // 事件类型
    QDateTime enterTime;        // 开始保存（接近门线）时间
    QDateTime exitTime;         // 完成穿越时间
    int entrySide;              // 进入侧：0=A（外侧/上方），1=B（内侧/下方）
    int exitSide;               // 离开侧：0=A（外侧/上方），1=B（内侧/下方）
    cv::Mat enterImage;         // 进入时整帧图像
    cv::Mat exitImage;          // 离开时整帧图像
    // ===== 乘客身份比对用（上车登记 / 下车比对）=====
    cv::Mat enterFeature;       // 进入该乘客的 ReID 特征
    cv::Mat exitFeature;        // 离开该乘客的 ReID 特征
    cv::Rect enterBbox;         // 进入时人体框
    cv::Rect exitBbox;          // 离开时人体框
    // ===== 整段穿越序列（接近并跨过门线期间逐帧采样）=====
    QVector<cv::Mat> trackImages;    // 逐帧人体裁剪图
    QVector<cv::Rect> trackBoxes;    // 对应原图人体框
    QVector<cv::Mat> trackFeatures;  // 逐帧特征（用于“整段序列”比对）
    // ===== 整段穿越视频帧（缩放到 960 宽的整帧序列，用于合成上/下车视频段）=====
    QVector<cv::Mat> trackFullFrames;
    // ===== 判定依据诊断字段 =====
    float startNormY{-1.0f};        // 起点（接近门线前）中心点归一化 Y
    float endNormY{-1.0f};          // 终点中心点归一化 Y
    float zoneTopNormY{-1.0f};      // 等价区域上边界（门线两端最小 Y）
    float zoneBottomNormY{-1.0f};   // 等价区域下边界（门线两端最大 Y）
    float zoneCenterNormY{-1.0f};   // 门线在图像中的纵向位置（中点 Y）
    float moveDy{0.0f};             // 终点-起点 的归一化纵向位移（正=向下；默认向下=下车）
    int   trackFrames{0};           // 本次采样到的整段帧数
    qint64 durationMs{0};           // 接近→穿越耗时（毫秒）
    bool  suppressed{false};        // 是否被防误判规则抑制
    QString suppressReason;         // 抑制原因（空=未抑制）
    // ===== DoorCounter v6 新增诊断字段 =====
    int    recoveryLayer{0};        // 采用的恢复层（RecoveryLayer）
    QString recoveryName;           // 恢复层名称
    float  crossNormX{-1.0f};       // 穿越点归一化 X
    float  crossNormY{-1.0f};       // 穿越点归一化 Y
    float  sideEnter{0.0f};         // 进入时相对门线有符号距离（<0=A侧，>0=B侧）
    float  sideExit{0.0f};          // 离开时相对门线有符号距离
    float  confidence{1.0f};        // 判定置信度 0~1
    bool   coasted{false};          // 是否由“惯性滑行”补全
    bool   stitched{false};         // 是否由“轨迹拼接”补全
    bool   fallback{false};         // 是否由“兜底策略”判定
};

/**
 * @brief 虚拟门线几何（归一化坐标）
 *
 * 侧别定义：normal 为单位法线（按构造始终指向图像下方 / Y 增大方向）。
 *   signedSide(p) = (p - a) · normal
 *   > 0 → B 侧（内侧/车门内）；< 0 → A 侧（外侧/车门外）。
 * 默认 entryDirection==TopToBottom 时：A→B（从上往下）跨越 = 下车。
 */
struct DoorLine {
    int regionId{-1};
    cv::Point2f a{0.f, 0.f};
    cv::Point2f b{0.f, 0.f};
    cv::Point2f dir{1.f, 0.f};      // 单位方向向量（a→b）
    cv::Point2f normal{0.f, 1.f};   // 单位法线（指向 Y 增大方向）
    float length{0.f};
    EntryDirection entry{EntryDirection::TopToBottom};

    inline float signedSide(const cv::Point2f &p) const
    {
        return (p.x - a.x) * normal.x + (p.y - a.y) * normal.y;
    }
    /// 投影是否落在门线线段范围内（marginRatio 为线段长度的外扩比例）
    inline bool withinSpan(const cv::Point2f &p, float marginRatio = 0.0f) const
    {
        if (length <= 1e-6f) return false;
        const float t = (p.x - a.x) * dir.x + (p.y - a.y) * dir.y;  // 沿门线的投影距离
        const float m = length * marginRatio;
        return (t >= -m) && (t <= length + m);
    }
    /// 中点
    inline cv::Point2f center() const { return cv::Point2f((a.x + b.x) * 0.5f, (a.y + b.y) * 0.5f); }
};

/**
 * @brief 图像保存状态
 */
enum class CaptureState {
    Idle,        // 未接近门线（未保存 / 停止保存）
    Capturing    // 正在接近门线走廊（正在保存整段图像）
};

/**
 * @brief 相对门线的侧别
 */
enum class DoorSide {
    Unknown = 0,
    ASide   = -1,   // 外侧（默认：门线上方 / Y 较小）
    BSide   = +1    // 内侧（默认：门线下方 / Y 较大）
};

/**
 * @brief 门线内人员的追踪状态
 */
struct DoorTrackedPerson {
    CaptureState state{CaptureState::Idle};
    int regionId{-1};
    int visitorId{-1};
    int seenFrames{0};                       // 该 track 被真实检测到的累计帧数（建档预热用）

    // ===== 几何 / 运动（归一化）=====
    DoorSide firstSide{DoorSide::Unknown};   // 本次接近窗口首次确认的侧别
    DoorSide lastSide{DoorSide::Unknown};    // 最近确认的侧别
    DoorSide pendingSide{DoorSide::Unknown}; // 待确认的新侧别
    int sideConfirm{0};                      // 侧别切换确认帧计数
    cv::Point2f prevCenter{-1.0f, -1.0f};
    cv::Point2f lastCenter{-1.0f, -1.0f};
    cv::Point2f velocity{0.f, 0.f};          // 归一化单位/秒
    bool  hasPrevCenter{false};
    qint64 firstSeenMs{0};
    qint64 lastSeenMs{0};
    float enterSigned{0.0f};                 // 接近窗口起点相对门线的有符号距离
    float lastSigned{0.0f};
    float lastCrossX{-1.0f}, lastCrossY{-1.0f};

    // ===== 本次接近已结算（避免重复计数；需重新离开走廊后再武装）=====
    bool crossedEmitted{false};
    bool coasted{false};                     // 由 L2 补全
    bool stitched{false};                    // 由 L3 补全
    RecoveryLayer seedLayer{RecoveryLayer::Realtime};

    // ===== 整段穿越序列采样 =====
    QDateTime enterTime;
    cv::Mat enterImage;
    cv::Mat enterFeature;
    cv::Rect enterBbox;
    cv::Rect lastBbox;
    QVector<cv::Mat> trackImages;
    QVector<cv::Rect> trackBoxes;
    QVector<cv::Mat> trackFeatures;
    QVector<cv::Mat> trackFullFrames;
    int sampleTick{0};
};

/**
 * @brief 垂直通道人员运动方向监测系统（DoorCounter v6）
 *
 * 核心：以「虚拟门线」为判定基准（而非整块区域）。
 *   1. 识别并跟踪乘客中心点；
 *   2. 持续计算中心点相对门线的**有符号距离**，得到侧别 A/B；
 *   3. 侧别发生翻转即视为一次“跨越门线”，方向由 A→B / B→A 决定上/下车；
 *   4. 使用四层恢复机制（实时判断、惯性滑行、轨迹拼接、兜底策略）保证鲁棒性；
 *   5. 携带整段图像与 ReID 特征，供下车时与上车乘客比对确认身份。
 */
class ZoneMonitor : public QObject
{
    Q_OBJECT
public:
    explicit ZoneMonitor(QObject *parent = nullptr);

    void setReidExtractor(ReIDExtractor *extractor);
    void setReidMatchThreshold(float threshold);

    /// 「离开接近走廊」确认所需连续帧数（防抖动）
    void setConfirmFrames(int frames);
    /// 判定“有效穿越”所需的最小纵向位移（归一化，0~1）
    void setMinDisplacement(float d);
    /// 「开始保存」所需的连续确认帧数
    void setStartConfirmFrames(int frames);
    /// 「开始保存」所需的人体框-走廊最小重叠面积（归一化，兼容旧接口）
    void setMinOverlapArea(float area);
    /// 「同人反向穿越」冷却窗口（毫秒）
    void setReverseCooldownMs(qint64 ms);
    /// 「同人同方向重复穿越」冷却窗口（毫秒）：抑制门线附近逗留/抖动造成的重复上下车
    void setSameDirCooldownMs(qint64 ms);
    /// 边界容差（归一化，兼容旧接口）
    void setEdgeMargin(float margin);

    // ===== DoorCounter v6 参数 =====
    /// 门线死区（归一化垂直距离）：在有符号距离 |d|<deadband 内不改变侧别，抑制抖动
    void setLineDeadband(float d);
    /// 接近走廊半宽（归一化）：用于决定何时开始/停止采样整段图像
    void setCorridorHalfWidth(float w);
    /// 侧别切换确认帧数（默认 2）
    void setSideConfirmFrames(int frames);
    /// L2 惯性滑行：目标丢失后允许按速度外推判定跨越的最长时间（毫秒）
    void setInertiaMaxMs(qint64 ms);
    /// L3 轨迹拼接：轨迹断裂后允许拼接续接的时间窗（毫秒）
    void setStitchWindowMs(qint64 ms);

    /**
     * @brief 启动采样/建档前所需的「轨迹预热帧数」（默认 3）
     *
     * 单帧/两帧的偶发误检（背景斑块、反光）不应被当作乘客建立访客档案，
     * 因此要求同一 track 被连续真实检测到 N 帧后才允许开始接近采样与建档。
     */
    void setMinTrackFrames(int frames);

    /**
     * @brief 设置检测区域（每个区域派生出一条虚拟门线）
     */
    void setChannelZones(const QVector<DetectionRegion> &regions);

    /**
     * @brief 处理一帧的跟踪结果
     */
    void processFrame(QVector<TrackedPerson> &persons,
                      const cv::Mat &rawFrame,
                      RegionManager *regionManager,
                      int imgW, int imgH);

    /**
     * @brief 清空「活跃穿越追踪状态」（m_tracks/m_trackToVisitor/m_anchors），
     *        用于视频判定跳帧回退重检时避免时间倒序污染穿越起点判定。
     *        保留累计计数（boarding/alighting）、已登记 visitor 特征与防误判冷却。
     */
    void resetTracks();

    int boardingCount() const { return m_boardingCount; }
    int alightingCount() const { return m_alightingCount; }
    /// 恢复累计计数（程序重启后不清零）
    void setCounters(int boarding, int alighting);
    QVector<ZoneEvent> events() const { return m_events; }

    /// 重置所有计数和事件记录
    void reset();

signals:
    void boardingEvent(const ZoneEvent &event);
    void alightingEvent(const ZoneEvent &event);
    void zoneCrossingEvent(const ZoneEvent &event);

private:
    // ===== 配置 =====
    ReIDExtractor *m_reidExtractor{nullptr};
    float m_reidMatchThreshold{0.70f};
    int m_confirmFrames{3};
    float m_minDisplacement{0.02f};
    int m_startConfirmFrames{1};
    float m_minOverlapArea{0.0012f};
    qint64 m_reverseCooldownMs{8000};
    qint64 m_sameDirCooldownMs{30000};
    float m_edgeMargin{0.0f};

    // ===== DoorCounter v6 参数 =====
    float  m_lineDeadband{0.006f};
    float  m_corridorHalf{0.16f};
    int    m_sideConfirmFrames{2};
    qint64 m_inertiaMaxMs{700};
    qint64 m_stitchWindowMs{1500};
    float  m_stitchRadius{0.14f};
    int    m_minTrackFrames{3};      // 建档/采样前的轨迹预热帧数

    // ===== 状态 =====
    QVector<DoorLine> m_lines;
    QMap<int, DoorTrackedPerson> m_tracks;      // trackId -> 追踪状态
    QMap<int, int> m_trackToVisitor;            // trackId -> visitorId
    QMap<int, cv::Mat> m_visitorFeatures;       // visitorId -> ReID 特征

    QVector<ZoneEvent> m_events;
    int m_boardingCount{0};
    int m_alightingCount{0};
    int m_nextVisitorId{1};

    // 防误判：记录每位访客最近一次「有效穿越」的时间与方向（+1=上车，-1=下车）
    QMap<int, qint64> m_lastCrossingMsByVisitor;
    QMap<int, int>    m_lastCrossingTypeByVisitor;

    /// L3 轨迹拼接锚点：轨迹在门线附近断裂时记录其“外侧起点 + 速度预测”，
    /// 供随后出现的新轨迹在同一时间窗/空间邻域内续接，拼成一次完整穿越。
    struct StitchAnchor {
        int regionId{-1};
        int visitorId{-1};
        cv::Point2f predicted{-1.0f, -1.0f};   // 断裂时刻按速度外推的位置
        cv::Point2f velocity{0.f, 0.f};
        int dirSign{0};                        // +1=向下，-1=向上
        DoorSide fromSide{DoorSide::Unknown};  // 断裂前已确认的侧别
        qint64 atMs{0};
        qint64 enterMs{0};
    };
    QVector<StitchAnchor> m_anchors;

    // ===== 内部辅助 =====
    void buildDoorLines(const QVector<DetectionRegion> &regions);
    int  bestLineIndex(const cv::Point2f &p, float maxPerp, float spanMargin) const;
    static DoorSide sideOf(float signedSide, float deadband);
    static int     sideCode(DoorSide s);   // 0=A, 1=B, -1=未知

    void sampleCapture(DoorTrackedPerson &t, const TrackedPerson &p,
                       const cv::Mat &rawFrame, int maxSamples);
    void startCapture(DoorTrackedPerson &t, const TrackedPerson &p,
                      const DoorLine &line, const cv::Mat &rawFrame,
                      const cv::Point2f &pert, float signedSide, qint64 nowMs);
    void stopCapture(DoorTrackedPerson &t);

    /// 结算并发出一次门线穿越事件
    void emitCrossing(DoorTrackedPerson &t, int trackId, const DoorLine &line,
                      DoorSide from, DoorSide to, RecoveryLayer layer,
                      const cv::Point2f &crossPoint, const cv::Mat &rawFrame,
                      qint64 nowMs, bool coasted, bool stitched);

    /// L2 惯性滑行：目标丢失时按速度外推判断是否已跨越
    bool tryCoastCrossing(DoorTrackedPerson &t, int trackId, const DoorLine &line,
                          const cv::Mat &rawFrame, qint64 nowMs, float dtSec);

    /// 轨迹移除时的最终结算（L2/L3/L4）
    void finalizeLost(DoorTrackedPerson &t, int trackId, const cv::Mat &rawFrame, qint64 nowMs);

    /// L3：新轨迹创建时尝试从锚点续接
    bool tryStitchInherit(DoorTrackedPerson &t, const DoorLine &line,
                          const cv::Point2f &pos, qint64 nowMs);

    void pruneAnchors(qint64 nowMs);

    void recordCrossing(int visitorId, int type, qint64 nowMs);
    bool isReverseInCooldown(int visitorId, int type, qint64 nowMs,
                             qint64 *outLastMs = nullptr, int *outLastType = nullptr) const;

    int matchOrCreateVisitor(const TrackedPerson &person, const cv::Mat &rawFrame);
    cv::Mat extractFeature(const TrackedPerson &person, const cv::Mat &rawFrame) const;
};

#endif // ZONEMONITOR_H
