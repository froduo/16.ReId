#include "flowaggregator.h"
#include "logger.h"

using namespace DongqiaFlow;

namespace {
// 固定行序（对齐协议 2.4）
const QStringList kHeightOrder{
    HeightCategory::AboveThreshold, HeightCategory::BelowThreshold,
    HeightCategory::Unknown, HeightCategory::NotMeasured};

const QStringList kAgeOrder{
    AgeRange::A0_9, AgeRange::A10_16, AgeRange::A17_30, AgeRange::A31_45,
    AgeRange::A46_60, AgeRange::A61Plus, AgeRange::Unknown, AgeRange::NotMeasured};

const QStringList kGenderOrder{
    Gender::Male, Gender::Female, Gender::Unknown, Gender::NotMeasured};

const QStringList kPersonTypeOrder{
    PersonType::Staff, PersonType::Rider, PersonType::Courier};

const QStringList kDwellOrder{
    DwellRange::D0_1, DwellRange::D1_5, DwellRange::D5_15,
    DwellRange::D15_30, DwellRange::D30Plus};

const QString kAgeGenderSep = QStringLiteral("|");

enum class CrossingDir { Boarding, Alighting };
}

FlowAggregator::FlowAggregator(QObject *parent)
    : QObject(parent)
{
    m_currentBatchDate = batchDateOf(QDateTime::currentDateTime());
    m_tick = new QTimer(this);
    m_tick->setInterval(TICK_MS);
    connect(m_tick, &QTimer::timeout, this, &FlowAggregator::onTick);
    m_tick->start();
}

// ------------------------------------------------------------
// 配置
// ------------------------------------------------------------

void FlowAggregator::setMasterSn(const QString &sn)
{
    m_masterSn = sn;
    if (m_deviceSns.isEmpty()) m_deviceSns = QStringList{sn};
}

void FlowAggregator::setDeviceSns(const QStringList &sns) { m_deviceSns = sns; }

void FlowAggregator::setStatBasis(const QString &basis)
{
    m_statBasis = (basis == QStringLiteral("leave")) ? QStringLiteral("leave")
                                                     : QStringLiteral("enter");
}

void FlowAggregator::setDayCutHour(int hour) { m_dayCutHour = qBound(0, hour, 23); }

void FlowAggregator::setTimezone(const QString &name, int utcOffsetSeconds, bool isDst)
{
    m_tzName = name;
    m_utcOffsetSeconds = utcOffsetSeconds;
    m_isDst = isDst;
}

// ------------------------------------------------------------
// 工具
// ------------------------------------------------------------

QString FlowAggregator::batchDateOf(const QDateTime &t) const
{
    // 营业日：日切时间之后归属新的一天；可跨午夜
    const QDateTime shifted = t.addSecs(-static_cast<qint64>(m_dayCutHour) * 3600);
    return shifted.date().toString(QStringLiteral("yyyy-MM-dd"));
}

qint64 FlowAggregator::currentMinuteStart() const
{
    const qint64 sec = QDateTime::currentSecsSinceEpoch();
    return sec - (sec % 60);
}

FlowAggregator::MinuteBucket &FlowAggregator::bucketFor(qint64 minuteStart)
{
    auto it = m_buckets.find(minuteStart);
    if (it == m_buckets.end()) {
        MinuteBucket b;
        b.startTsS = minuteStart;
        it = m_buckets.insert(minuteStart, b);
    }
    return it.value();
}

FlowAggregator::PersonTimeline &FlowAggregator::timelineRef(int visitorId)
{
    auto it = m_timelines.find(visitorId);
    if (it == m_timelines.end()) {
        PersonTimeline t;
        t.visitorId = visitorId;
        it = m_timelines.insert(visitorId, t);
    }
    return it.value();
}

void FlowAggregator::appendTimelineEvent(int visitorId, const QString &type, const QDateTime &t)
{
    PersonTimeline &tl = timelineRef(visitorId);
    tl.eventLog.append(QString("%1 %2").arg(t.toString("yyyy-MM-dd HH:mm:ss.zzz"), type));
}

// ------------------------------------------------------------
// 事件输入
// ------------------------------------------------------------

void FlowAggregator::onEnter(int visitorId, const QDateTime &t, PersonKind kind,
                             float reidSimilarity, const PersonAttributes &attrs,
                             const cv::Mat &snapshot, int regionId)
{
    const QString bd = batchDateOf(t);
    if (bd != m_currentBatchDate) {
        LOG_INFO(QString("[FlowAggregator] 营业日切换 %1 -> %2，重置当天去重集合")
                     .arg(m_currentBatchDate, bd));
        m_seenToday.clear();
        m_enterInfo.clear();
        m_buckets.clear();
        m_currentBatchDate = bd;
    }

    const qint64 sec = toUnixSec(t);
    const qint64 ms = sec - (sec % 60);
    MinuteBucket &b = bucketFor(ms);
    b.revision = QDateTime::currentSecsSinceEpoch();

    b.ec.enter++;
    b.rawCount++;
    b.rawHeight[attrs.heightCategory]++;
    b.rawAgeGender[attrs.ageRange + kAgeGenderSep + attrs.gender]++;

    const bool isNew = !m_seenToday.contains(visitorId);
    if (kind == PersonKind::Customer) {
        if (isNew) {
            b.finalCount++;
            b.finalHeight[attrs.heightCategory]++;
            b.finalAgeGender[attrs.ageRange + kAgeGenderSep + attrs.gender]++;
            m_seenToday.insert(visitorId);
        } else {
            b.finalRepeat++;
        }
    } else {
        b.nonCustomerType[personKindToProtocol(kind)]++;
    }

    EnterInfo info;
    info.enterSec = sec;
    info.minuteStart = ms;
    info.kind = kind;
    info.countedAsNew = isNew;
    m_enterInfo[visitorId] = info;

    // 时间线
    PersonTimeline &tl = timelineRef(visitorId);
    if (!tl.firstEnter.isValid()) tl.firstEnter = t;
    tl.enterCount++;
    appendTimelineEvent(visitorId, QStringLiteral("enter"), t);
    emit personTracked(visitorId, QStringLiteral("enter"), t);

    // 单人事件上行业务
    PersonFlowEvent evt;
    evt.masterSn = m_masterSn;
    evt.batchDate = m_currentBatchDate;
    evt.statBasis = m_statBasis;
    evt.visitorId = visitorId;
    evt.kind = kind;
    evt.eventType = QStringLiteral("enter");
    evt.direction = QString();
    evt.enterTsS = sec;
    evt.exitTsS = 0;
    evt.eventTsS = sec;
    evt.dwellSeconds = 0;
    evt.reidSimilarity = reidSimilarity;
    evt.regionId = regionId;
    evt.snapshotBase64 = matToJpegBase64(snapshot);
    emit personEventReady(evt);

    LOG_INFO(QString("[FlowAggregator] enter V%1 kind=%2 new=%3 bucket=%4 raw=%5 final=%6 repeat=%7 nonC=%8")
                 .arg(visitorId).arg(personKindToProtocol(kind))
                 .arg(isNew ? 1 : 0).arg(ms).arg(b.rawCount)
                 .arg(b.finalCount).arg(b.finalRepeat)
                 .arg([&]{int s=0; for(int v:b.nonCustomerType) s+=v; return s;}()));

    emitBucket(b, true);
}

void FlowAggregator::onLeave(int visitorId, const QDateTime &t, PersonKind kind,
                             const cv::Mat &snapshot, int regionId)
{
    const qint64 sec = toUnixSec(t);
    const qint64 ms = sec - (sec % 60);

    int dwellSeconds = 0;
    qint64 enterSecForEvent = 0;
    if (m_enterInfo.contains(visitorId)) {
        const EnterInfo info = m_enterInfo.value(visitorId);
        enterSecForEvent = info.enterSec;
        dwellSeconds = static_cast<int>(qMax<qint64>(0, sec - info.enterSec));
        if (info.kind == PersonKind::Customer) {
            // 驻留分布按「进店时间」归入时段（协议 2.3⑤）
            MinuteBucket &eb = bucketFor(info.minuteStart);
            eb.dwellCount++;
            eb.dwellRange[dwellBucketOfSeconds(dwellSeconds)]++;
            eb.revision = QDateTime::currentSecsSinceEpoch();
            emitBucket(eb, true);
        }
        m_enterInfo.remove(visitorId);
    }

    MinuteBucket &b = bucketFor(ms);
    b.ec.leave++;
    b.revision = QDateTime::currentSecsSinceEpoch();

    PersonTimeline &tl = timelineRef(visitorId);
    tl.lastExit = t;
    tl.leaveCount++;
    tl.totalDwellSeconds += dwellSeconds;
    appendTimelineEvent(visitorId, QStringLiteral("leave"), t);
    emit personTracked(visitorId, QStringLiteral("leave"), t);

    PersonFlowEvent evt;
    evt.masterSn = m_masterSn;
    evt.batchDate = m_currentBatchDate;
    evt.statBasis = m_statBasis;
    evt.visitorId = visitorId;
    evt.kind = kind;
    evt.eventType = QStringLiteral("leave");
    evt.direction = QString();
    evt.enterTsS = enterSecForEvent;
    evt.exitTsS = sec;
    evt.eventTsS = sec;
    evt.dwellSeconds = dwellSeconds;
    evt.reidSimilarity = -1.0f;
    evt.regionId = regionId;
    evt.snapshotBase64 = matToJpegBase64(snapshot);
    emit personEventReady(evt);

    LOG_INFO(QString("[FlowAggregator] leave V%1 dwell=%2s bucket=%3")
                 .arg(visitorId).arg(dwellSeconds).arg(ms));
    emitBucket(b, true);
}

void FlowAggregator::onBoarding(int visitorId, const QDateTime &enterTime, const QDateTime &exitTime,
                                float reidSimilarity, const cv::Mat &enterSnapshot,
                                const cv::Mat &exitSnapshot, int regionId)
{
    // 同一个人短时间内被拆成多次上报 → 去抖，只记一次（跨轨迹 ID 抖动兜底）
    const qint64 nowMs = QDateTime::currentMSecsSinceEpoch();
    if (visitorId > 0 && (nowMs - m_lastCrossingMs.value(visitorId, 0)) < CROSSING_DEDUP_MS) {
        LOG_INFO(QString("[FlowAggregator] skip duplicate boarding V%1").arg(visitorId));
        return;
    }
    if (visitorId > 0) m_lastCrossingMs[visitorId] = nowMs;

    const QString bd = batchDateOf(enterTime);
    if (bd != m_currentBatchDate) {
        m_seenToday.clear();
        m_enterInfo.clear();
        m_buckets.clear();
        m_currentBatchDate = bd;
    }

    const qint64 enterSec = toUnixSec(enterTime);
    const qint64 exitSec = toUnixSec(exitTime);
    const qint64 enterMs = enterSec - (enterSec % 60);
    const qint64 exitMs = exitSec - (exitSec % 60);
    const int dwellSeconds = static_cast<int>(qMax<qint64>(0, exitSec - enterSec));

    // 上车(B→A)：event_counts.enter++
    MinuteBucket &be = bucketFor(enterMs);
    be.revision = QDateTime::currentSecsSinceEpoch();
    be.ec.enter++;

    // 基准方向判定：基准为 enter 时，上车是可靠观测
    const bool isPrimary = (m_statBasis == QStringLiteral("enter"));
    if (isPrimary) {
        be.rawCount++;
        be.rawHeight[HeightCategory::NotMeasured]++;
        be.rawAgeGender[AgeRange::NotMeasured + kAgeGenderSep + Gender::NotMeasured]++;

        const bool isNew = !m_seenToday.contains(visitorId);
        if (isNew) {
            be.finalCount++;
            be.finalHeight[HeightCategory::NotMeasured]++;
            be.finalAgeGender[AgeRange::NotMeasured + kAgeGenderSep + Gender::NotMeasured]++;
            m_seenToday.insert(visitorId);
        } else {
            be.finalRepeat++;
        }

        // 驻留（已完成进+离）
        be.dwellCount++;
        be.dwellRange[dwellBucketOfSeconds(dwellSeconds)]++;
    }

    // 时间线
    PersonTimeline &tl = timelineRef(visitorId);
    if (!tl.firstEnter.isValid()) tl.firstEnter = enterTime;
    tl.lastExit = exitTime;
    tl.boardingCount++;
    tl.enterCount++;
    tl.leaveCount++;
    tl.totalDwellSeconds += dwellSeconds;
    appendTimelineEvent(visitorId, QStringLiteral("boarding(B->A)"), exitTime);

    PersonFlowEvent evt;
    evt.masterSn = m_masterSn;
    evt.batchDate = m_currentBatchDate;
    evt.statBasis = m_statBasis;
    evt.visitorId = visitorId;
    evt.kind = PersonKind::Customer;
    evt.eventType = QStringLiteral("boarding");
    evt.direction = QStringLiteral("B->A");
    evt.enterTsS = enterSec;
    evt.exitTsS = exitSec;
    evt.eventTsS = exitSec;
    evt.dwellSeconds = dwellSeconds;
    evt.reidSimilarity = reidSimilarity;
    evt.regionId = regionId;
    evt.snapshotBase64 = matToJpegBase64(exitSnapshot.empty() ? enterSnapshot : exitSnapshot);
    emit personEventReady(evt);
    emit personTracked(visitorId, QStringLiteral("boarding"), exitTime);

    LOG_INFO(QString("[FlowAggregator] ★boarding V%1 B→A enter=%2 exit=%3 dwell=%4s primary=%5")
                 .arg(visitorId).arg(enterSec).arg(exitSec).arg(dwellSeconds).arg(isPrimary ? 1 : 0));
    emitBucket(be, true);

    Q_UNUSED(exitMs);
}

void FlowAggregator::onAlighting(int visitorId, const QDateTime &enterTime, const QDateTime &exitTime,
                                 float reidSimilarity, const cv::Mat &enterSnapshot,
                                 const cv::Mat &exitSnapshot, int regionId)
{
    // 同人重复上报去抖
    const qint64 nowMs = QDateTime::currentMSecsSinceEpoch();
    if (visitorId > 0 && (nowMs - m_lastCrossingMs.value(visitorId, 0)) < CROSSING_DEDUP_MS) {
        LOG_INFO(QString("[FlowAggregator] skip duplicate alighting V%1").arg(visitorId));
        return;
    }
    if (visitorId > 0) m_lastCrossingMs[visitorId] = nowMs;

    const QString bd = batchDateOf(exitTime);
    if (bd != m_currentBatchDate) {
        m_seenToday.clear();
        m_enterInfo.clear();
        m_buckets.clear();
        m_currentBatchDate = bd;
    }

    const qint64 enterSec = toUnixSec(enterTime);
    const qint64 exitSec = toUnixSec(exitTime);
    const qint64 exitMs = exitSec - (exitSec % 60);
    const int dwellSeconds = static_cast<int>(qMax<qint64>(0, exitSec - enterSec));

    // 下车(A→B)：event_counts.leave++
    MinuteBucket &bl = bucketFor(exitMs);
    bl.revision = QDateTime::currentSecsSinceEpoch();
    bl.ec.leave++;

    // 若基准方向是 leave，则下车才是可靠观测
    const bool isPrimary = (m_statBasis == QStringLiteral("leave"));
    if (isPrimary) {
        const qint64 enterMs = enterSec - (enterSec % 60);
        MinuteBucket &be = bucketFor(enterMs);
        be.revision = QDateTime::currentSecsSinceEpoch();
        be.rawCount++;
        be.rawHeight[HeightCategory::NotMeasured]++;
        be.rawAgeGender[AgeRange::NotMeasured + kAgeGenderSep + Gender::NotMeasured]++;
        const bool isNew = !m_seenToday.contains(visitorId);
        if (isNew) {
            be.finalCount++;
            be.finalHeight[HeightCategory::NotMeasured]++;
            be.finalAgeGender[AgeRange::NotMeasured + kAgeGenderSep + Gender::NotMeasured]++;
            m_seenToday.insert(visitorId);
        } else {
            be.finalRepeat++;
        }
        be.dwellCount++;
        be.dwellRange[dwellBucketOfSeconds(dwellSeconds)]++;
        emitBucket(be, true);
    }

    PersonTimeline &tl = timelineRef(visitorId);
    if (!tl.firstEnter.isValid()) tl.firstEnter = enterTime;
    tl.lastExit = exitTime;
    tl.alightingCount++;
    tl.enterCount++;
    tl.leaveCount++;
    tl.totalDwellSeconds += dwellSeconds;
    appendTimelineEvent(visitorId, QStringLiteral("alighting(A->B)"), exitTime);

    PersonFlowEvent evt;
    evt.masterSn = m_masterSn;
    evt.batchDate = m_currentBatchDate;
    evt.statBasis = m_statBasis;
    evt.visitorId = visitorId;
    evt.kind = PersonKind::Customer;
    evt.eventType = QStringLiteral("alighting");
    evt.direction = QStringLiteral("A->B");
    evt.enterTsS = enterSec;
    evt.exitTsS = exitSec;
    evt.eventTsS = exitSec;
    evt.dwellSeconds = dwellSeconds;
    evt.reidSimilarity = reidSimilarity;
    evt.regionId = regionId;
    evt.snapshotBase64 = matToJpegBase64(exitSnapshot.empty() ? enterSnapshot : exitSnapshot);
    emit personEventReady(evt);
    emit personTracked(visitorId, QStringLiteral("alighting"), exitTime);

    LOG_INFO(QString("[FlowAggregator] ★alighting V%1 A→B enter=%2 exit=%3 dwell=%4s primary=%5")
                 .arg(visitorId).arg(enterSec).arg(exitSec).arg(dwellSeconds).arg(isPrimary ? 1 : 0));
    emitBucket(bl, true);
}

void FlowAggregator::onPassBy(const QDateTime &t)
{
    const qint64 sec = toUnixSec(t);
    const qint64 ms = sec - (sec % 60);
    MinuteBucket &b = bucketFor(ms);
    b.revision = QDateTime::currentSecsSinceEpoch();
    b.ec.passBy++;
    emitBucket(b, true);
}

void FlowAggregator::onTurnBack(const QDateTime &t)
{
    const qint64 sec = toUnixSec(t);
    const qint64 ms = sec - (sec % 60);
    MinuteBucket &b = bucketFor(ms);
    b.revision = QDateTime::currentSecsSinceEpoch();
    b.ec.turnBack++;
    emitBucket(b, true);
}

// ------------------------------------------------------------
// 聚合与发送
// ------------------------------------------------------------

QVector<BucketRow> FlowAggregator::orderedBucketRows(const QMap<QString, int> &map,
                                                     const QStringList &order)
{
    QVector<BucketRow> rows;
    for (const QString &k : order) {
        if (map.contains(k) && map.value(k) > 0)
            rows.append(BucketRow(k, map.value(k)));
    }
    // 防御：补上不在预定义顺序里的键（按行读取，不假设组合齐全）
    for (auto it = map.constBegin(); it != map.constEnd(); ++it) {
        if (!order.contains(it.key()) && it.value() > 0)
            rows.append(BucketRow(it.key(), it.value()));
    }
    return rows;
}

QVector<AgeGenderRow> FlowAggregator::orderedAgeGenderRows(const QMap<QString, int> &map)
{
    QVector<AgeGenderRow> rows;
    for (const QString &age : kAgeOrder) {
        for (const QString &g : kGenderOrder) {
            const QString key = age + kAgeGenderSep + g;
            if (map.contains(key) && map.value(key) > 0)
                rows.append(AgeGenderRow(age, g, map.value(key)));
        }
    }
    return rows;
}

DongqiaFlow::IntervalAggregate FlowAggregator::buildAggregate(const MinuteBucket &b,
                                                              qint64 revision) const
{
    IntervalAggregate agg;
    agg.masterSn = m_masterSn;
    agg.deviceSns = m_deviceSns.isEmpty() ? QStringList{m_masterSn} : m_deviceSns;
    agg.tzName = m_tzName;
    agg.utcOffsetSeconds = m_utcOffsetSeconds;
    agg.isDst = m_isDst;
    agg.batchDate = m_currentBatchDate;
    agg.statBasis = m_statBasis;
    agg.revision = revision;
    agg.intervalS = 60;
    agg.startTsS = b.startTsS;
    agg.endTsS = b.startTsS + 60;

    agg.eventCounts = b.ec;

    agg.rawCount = b.rawCount;
    agg.rawHeight = orderedBucketRows(b.rawHeight, kHeightOrder);
    agg.rawAgeGender = orderedAgeGenderRows(b.rawAgeGender);

    agg.finalCount = b.finalCount;
    agg.finalRepeat = b.finalRepeat;
    agg.finalHeight = orderedBucketRows(b.finalHeight, kHeightOrder);
    agg.finalAgeGender = orderedAgeGenderRows(b.finalAgeGender);

    int nonC = 0;
    for (int v : b.nonCustomerType) nonC += v;
    agg.nonCustomerCount = nonC;
    agg.nonCustomerPersonType = orderedBucketRows(b.nonCustomerType, kPersonTypeOrder);

    agg.dwellCount = b.dwellCount;
    agg.dwellRange = orderedBucketRows(b.dwellRange, kDwellOrder);

    return agg;
}

void FlowAggregator::emitBucket(MinuteBucket &b, bool bumpRevision)
{
    if (bumpRevision) b.revision = QDateTime::currentSecsSinceEpoch();
    IntervalAggregate agg = buildAggregate(b, b.revision);
    emit intervalReady(agg);
}

void FlowAggregator::onTick()
{
    m_tickCounter++;
    const qint64 curMin = currentMinuteStart();

    // 分钟切换：把已完成的分钟定稿发出
    if (curMin != m_lastEmittedMinute) {
        for (auto it = m_buckets.begin(); it != m_buckets.end(); ++it) {
            if (it.key() < curMin)
                emitBucket(it.value(), true);
        }
        m_lastEmittedMinute = curMin;
        return;
    }

    // 进行中的分钟：周期性上报修订版（协议 4.1）
    if (m_tickCounter % EMIT_INTERVAL_TICKS == 0) {
        auto it = m_buckets.find(curMin);
        if (it != m_buckets.end())
            emitBucket(it.value(), true);
    }
}

// ------------------------------------------------------------
// 查询 / 验证
// ------------------------------------------------------------

FlowAggregator::PersonTimeline FlowAggregator::timelineFor(int visitorId) const
{
    return m_timelines.value(visitorId);
}

QVector<int> FlowAggregator::knownVisitors() const
{
    QVector<int> ids;
    for (auto it = m_timelines.constBegin(); it != m_timelines.constEnd(); ++it)
        ids.append(it.key());
    return ids;
}

QString FlowAggregator::summarizePerson(int visitorId) const
{
    if (!m_timelines.contains(visitorId))
        return QStringLiteral("V%1: 无记录").arg(visitorId);

    const PersonTimeline &t = m_timelines.value(visitorId);
    return QStringLiteral("V%1 上车%2次/下车%3次 | 进入%4/离开%5 | 首次 %6 | 最近 %7 | 累计驻留 %8s")
        .arg(visitorId)
        .arg(t.boardingCount).arg(t.alightingCount)
        .arg(t.enterCount).arg(t.leaveCount)
        .arg(t.firstEnter.isValid() ? t.firstEnter.toString("HH:mm:ss.zzz") : QStringLiteral("--"))
        .arg(t.lastExit.isValid() ? t.lastExit.toString("HH:mm:ss.zzz") : QStringLiteral("--"))
        .arg(t.totalDwellSeconds);
}

DongqiaFlow::IntervalAggregate FlowAggregator::snapshotCurrentInterval() const
{
    const qint64 ms = currentMinuteStart();
    auto it = m_buckets.constFind(ms);
    if (it == m_buckets.constEnd()) {
        MinuteBucket empty;
        empty.startTsS = ms;
        return buildAggregate(empty, QDateTime::currentSecsSinceEpoch());
    }
    return buildAggregate(it.value(), QDateTime::currentSecsSinceEpoch());
}

void FlowAggregator::reset()
{
    m_buckets.clear();
    m_enterInfo.clear();
    m_seenToday.clear();
    m_timelines.clear();
    m_lastCrossingMs.clear();
    m_lastEmittedMinute = -1;
    LOG_INFO("[FlowAggregator] reset");
}
