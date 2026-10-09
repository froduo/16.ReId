#ifndef FLOWAGGREGATOR_H
#define FLOWAGGREGATOR_H

/**
 * @file flowaggregator.h
 * @brief 客流时段聚合器 + 人物上下车精准识别追踪
 *
 * 把上层（ZoneMonitor / MainWindow）产生的进出、上车、下车、路过、折返事件，
 * 按「一分钟一个时段」聚合成符合《动恰3DV6 AI HTTP 新对接协议》的报文，
 * 并维护每位访客（ReID 身份）的上下车时间线，用于验证「精准识别某一个人的上车/下车」。
 *
 * 关键不变量（协议 2.5，构造时即保证）：
 *   raw_stats.count == final_stats.count + final_stats.repeat + non_customer.count
 *   各维度分桶之和 == 所属块 count（无检测数据时全部落入 not_measured，仍计入总数）
 */

#include <QObject>
#include <QMap>
#include <QSet>
#include <QVector>
#include <QDateTime>
#include <QStringList>
#include <QTimer>
#include <opencv2/opencv.hpp>

#include "dongqia_protocol.h"

/// 人物属性（无检测时保持 not_measured）
///
/// 注意：此处定义为文件作用域而非嵌套类，以规避 GCC 对「嵌套类默认成员初始化器
/// 出现在外层类成员函数默认实参中」的限制（default member initializer required
/// before the end of its enclosing class）。
struct PersonAttributes {
    QString heightCategory{DongqiaFlow::HeightCategory::NotMeasured};
    QString ageRange{DongqiaFlow::AgeRange::NotMeasured};
    QString gender{DongqiaFlow::Gender::NotMeasured};
};

class FlowAggregator : public QObject
{
    Q_OBJECT
public:
    /// 单个人的上下车/进出时间线（用于验证精准识别）
    struct PersonTimeline {
        int visitorId{-1};
        QDateTime firstEnter;
        QDateTime lastExit;
        int enterCount{0};
        int leaveCount{0};
        int boardingCount{0};
        int alightingCount{0};
        int totalDwellSeconds{0};
        QStringList eventLog;   // "yyyy-MM-dd HH:mm:ss.zzz type"
    };

    explicit FlowAggregator(QObject *parent = nullptr);

    void setMasterSn(const QString &sn);
    void setDeviceSns(const QStringList &sns);
    void setStatBasis(const QString &basis);     // "enter" / "leave"
    void setDayCutHour(int hour);                // 日切时间，默认为 0
    void setTimezone(const QString &name, int utcOffsetSeconds, bool isDst);

    // ===== 事件输入 =====

    /// 进店（基准方向）。kind=Customer 时参与 final/raw；否则计入 non_customer
    void onEnter(int visitorId, const QDateTime &t,
                 DongqiaFlow::PersonKind kind = DongqiaFlow::PersonKind::Customer,
                 float reidSimilarity = -1.0f,
                 const PersonAttributes &attrs = PersonAttributes(),
                 const cv::Mat &snapshot = cv::Mat(),
                 int regionId = -1);

    /// 离店
    void onLeave(int visitorId, const QDateTime &t,
                 DongqiaFlow::PersonKind kind = DongqiaFlow::PersonKind::Customer,
                 const cv::Mat &snapshot = cv::Mat(),
                 int regionId = -1);

    /// 上车（B→A）：底端进入、顶端离开
    void onBoarding(int visitorId, const QDateTime &enterTime, const QDateTime &exitTime,
                    float reidSimilarity = -1.0f,
                    const cv::Mat &enterSnapshot = cv::Mat(),
                    const cv::Mat &exitSnapshot = cv::Mat(),
                    int regionId = -1);

    /// 下车（A→B）：顶端进入、底端离开
    void onAlighting(int visitorId, const QDateTime &enterTime, const QDateTime &exitTime,
                     float reidSimilarity = -1.0f,
                     const cv::Mat &enterSnapshot = cv::Mat(),
                     const cv::Mat &exitSnapshot = cv::Mat(),
                     int regionId = -1);

    /// 路过（在门口区域出现但没进店）
    void onPassBy(const QDateTime &t);
    /// 折返（跨过门线后又退回去）
    void onTurnBack(const QDateTime &t);

    // ===== 查询 / 验证 =====

    /// 某位访客的上下车时间线（精准识别验证）
    PersonTimeline timelineFor(int visitorId) const;
    /// 已知访客 ID 列表
    QVector<int> knownVisitors() const;
    /// 生成可读摘要，例如：“V3 上车 2 次 / 下车 2 次，首次 09:10:03，最近 09:40:22”
    QString summarizePerson(int visitorId) const;

    /// 当前时段聚合快照（不修改 revision，用于调试/测试）
    DongqiaFlow::IntervalAggregate snapshotCurrentInterval() const;

    void reset();

signals:
    /// 一个时段（分钟）聚合完成，可上行
    void intervalReady(const DongqiaFlow::IntervalAggregate &agg);
    /// 单人事件就绪（上车/下车/进出），可上行
    void personEventReady(const DongqiaFlow::PersonFlowEvent &evt);
    /// 每识别到一位访客的上下车事件（用于 UI 精准识别验证）
    void personTracked(int visitorId, const QString &eventType, const QDateTime &t);

private slots:
    void onTick();

private:
    struct MinuteBucket {
        qint64 startTsS{0};
        qint64 revision{0};
        DongqiaFlow::EventCounts ec;

        int rawCount{0};
        QMap<QString, int> rawHeight;
        QMap<QString, int> rawAgeGender;   // key: "age|gender"

        int finalCount{0};
        int finalRepeat{0};
        QMap<QString, int> finalHeight;
        QMap<QString, int> finalAgeGender;

        QMap<QString, int> nonCustomerType; // staff/rider/courier

        int dwellCount{0};
        QMap<QString, int> dwellRange;
    };

    struct EnterInfo {
        qint64 enterSec{0};
        qint64 minuteStart{0};
        DongqiaFlow::PersonKind kind{DongqiaFlow::PersonKind::Customer};
        bool countedAsNew{false};
    };

    MinuteBucket &bucketFor(qint64 minuteStart);
    qint64 currentMinuteStart() const;
    QString batchDateOf(const QDateTime &t) const;
    DongqiaFlow::IntervalAggregate buildAggregate(const MinuteBucket &b, qint64 revision) const;
    void emitBucket(MinuteBucket &b, bool bumpRevision);
    PersonTimeline &timelineRef(int visitorId);
    void appendTimelineEvent(int visitorId, const QString &type, const QDateTime &t);

    // 有序维度列表（行序固定，对齐协议 2.4）
    static QVector<DongqiaFlow::BucketRow> orderedBucketRows(
        const QMap<QString, int> &map, const QStringList &order);
    static QVector<DongqiaFlow::AgeGenderRow> orderedAgeGenderRows(
        const QMap<QString, int> &map);

    // ---- 配置 ----
    QString m_masterSn{QStringLiteral("CAM-A01")};
    QStringList m_deviceSns;
    QString m_statBasis{QStringLiteral("enter")};
    int m_dayCutHour{0};
    QString m_tzName{QStringLiteral("Asia/Shanghai")};
    int m_utcOffsetSeconds{28800};
    bool m_isDst{false};

    // ---- 运行状态 ----
    QMap<qint64, MinuteBucket> m_buckets;      // minuteStart -> bucket
    QMap<int, EnterInfo> m_enterInfo;          // visitorId -> 最近一次 enter
    QSet<int> m_seenToday;                      // 当天已出现过（用于回头客判定）
    QMap<int, PersonTimeline> m_timelines;     // visitorId -> 时间线
    QMap<int, qint64> m_lastCrossingMs;        // personId -> 上次上下车事件时刻（去抖）
    QString m_currentBatchDate;
    qint64 m_lastEmittedMinute{-1};
    QTimer *m_tick{nullptr};
    int m_tickCounter{0};
    static constexpr int TICK_MS = 1000;
    static constexpr int EMIT_INTERVAL_TICKS = 10; // 进行中的分钟每 ~10s 上报一次修订版
    static constexpr qint64 CROSSING_DEDUP_MS = 2500; // 同人重复上下车去抖窗口
};

#endif // FLOWAGGREGATOR_H
