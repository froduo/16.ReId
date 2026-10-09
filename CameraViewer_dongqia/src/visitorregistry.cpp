#include "visitorregistry.h"
#include "reidextractor.h"
#include "logger.h"

#include <QDateTime>
#include <QMutexLocker>
#include <cmath>

namespace {
// 时空门限参数
constexpr qint64 kNearbyMs      = 3000;    // 视为“刚见过”的时间窗
constexpr float  kNearbyDist    = 0.15f;   // 视为“位置接近”的归一化距离
constexpr qint64 kLongGapMs     = 60000;   // 超过此时长按更严格阈值
constexpr float  kInsidePenalty = 0.15f;   // 该人当前已在场内（防两人被并成一个）
constexpr float  kNearbyBonus   = 0.10f;   // 刚见过且位置近（放宽）
constexpr float  kLongGapPenalty= 0.10f;   // 隔太久（收紧）
} // namespace

void VisitorRegistry::setBaseThreshold(float t)
{
    QMutexLocker locker(&m_mutex);
    m_baseThreshold = t;
}

float VisitorRegistry::baseThreshold() const
{
    QMutexLocker locker(&m_mutex);
    return m_baseThreshold;
}

void VisitorRegistry::setGalleryMaxAgeMs(qint64 ms)
{
    QMutexLocker locker(&m_mutex);
    m_galleryMaxAgeMs = ms;
}

void VisitorRegistry::ensureNextIdAtLeast(int minId)
{
    QMutexLocker locker(&m_mutex);
    if (minId > m_nextId) m_nextId = minId;
}

float VisitorRegistry::similarityLocked(const cv::Mat &a, const cv::Mat &b) const
{
    if (!m_extractor) return 0.0f;
    if (a.empty() || b.empty()) return 0.0f;
    return m_extractor->compare(a, b);
}

float VisitorRegistry::similarity(const cv::Mat &a, const cv::Mat &b) const
{
    QMutexLocker locker(&m_mutex);
    return similarityLocked(a, b);
}

void VisitorRegistry::updateFeatureLocked(Entry &e, const cv::Mat &f)
{
    if (f.empty()) return;

    cv::Mat fm = f.reshape(1, 1);
    if (fm.type() != CV_32F) fm.convertTo(fm, CV_32F);

    if (e.feature.empty() || e.feature.total() != fm.total()) {
        e.feature = fm.clone();
        e.featCount = 1;
        return;
    }

    // 运行平均：feature = (feature*count + f) / (count+1)
    cv::Mat acc;
    cv::addWeighted(e.feature, static_cast<double>(e.featCount),
                    fm, 1.0, 0.0, acc);
    acc /= static_cast<double>(e.featCount + 1);
    e.feature = acc;
    e.featCount++;
}

void VisitorRegistry::pruneLocked(qint64 nowMs)
{
    auto it = m_entries.begin();
    while (it != m_entries.end()) {
        const bool expired = (nowMs - it.value().lastSeenMs) > m_galleryMaxAgeMs;
        if (expired && !it.value().inside) {
            LOG_INFO(QString("[VisitorRegistry] prune V%1 (idle %2ms)")
                         .arg(it.key()).arg(nowMs - it.value().lastSeenMs));
            it = m_entries.erase(it);
        } else {
            ++it;
        }
    }
}

VisitorRegistry::MatchInfo VisitorRegistry::resolveEx(const cv::Mat &feature, qint64 nowMs,
                                                      const cv::Point2f &normPos)
{
    QMutexLocker locker(&m_mutex);

    pruneLocked(nowMs);

    MatchInfo info;
    int bestId = -1;
    float bestSim = -1.0f;

    if (!feature.empty()) {
        for (auto it = m_entries.begin(); it != m_entries.end(); ++it) {
            Entry &e = it.value();
            if (e.feature.empty()) continue;

            const float sim = similarityLocked(feature, e.feature);
            if (sim <= 0.0f) continue;

            // 自适应阈值
            float thr = m_baseThreshold;
            if (e.inside) thr += kInsidePenalty;

            const qint64 gap = nowMs - e.lastSeenMs;
            float dist = -1.0f;
            if (e.lastNormPos.x >= 0.0f) {
                const float dx = normPos.x - e.lastNormPos.x;
                const float dy = normPos.y - e.lastNormPos.y;
                dist = std::sqrt(dx * dx + dy * dy);
            }
            if (gap < kNearbyMs && dist >= 0.0f && dist < kNearbyDist) {
                thr -= kNearbyBonus;
            } else if (gap > kLongGapMs) {
                thr += kLongGapPenalty;
            }
            if (thr < 0.1f) thr = 0.1f;
            if (thr > 0.98f) thr = 0.98f;

            if (sim >= thr && sim > bestSim) {
                bestSim = sim;
                bestId = it.key();
            }
        }
    }

    if (bestId > 0) {
        Entry &e = m_entries[bestId];
        updateFeatureLocked(e, feature);
        e.lastSeenMs = nowMs;
        e.lastNormPos = normPos;
        e.seenCount++;

        info.personId = bestId;
        info.similarity = bestSim;
        info.matched = true;
        info.created = false;

        LOG_INFO(QString("[VisitorRegistry] match V%1 sim=%2 (cnt=%3)")
                     .arg(bestId).arg(bestSim, 0, 'f', 3).arg(e.seenCount));
        return info;
    }

    // 新建身份
    Entry e;
    e.personId = m_nextId++;
    e.firstSeenMs = nowMs;
    e.lastSeenMs = nowMs;
    e.lastNormPos = normPos;
    e.seenCount = 1;
    updateFeatureLocked(e, feature);
    m_entries.insert(e.personId, e);

    info.personId = e.personId;
    info.similarity = 0.0f;
    info.matched = false;
    info.created = true;

    LOG_INFO(QString("[VisitorRegistry] new person V%1 (total=%2, hasFeat=%3)")
                 .arg(e.personId).arg(m_entries.size())
                 .arg(e.feature.empty() ? 0 : 1));
    return info;
}

int VisitorRegistry::resolve(const cv::Mat &feature, qint64 nowMs,
                             const cv::Point2f &normPos)
{
    return resolveEx(feature, nowMs, normPos).personId;
}

void VisitorRegistry::reinforce(int personId, const cv::Mat &feature, qint64 nowMs,
                                const cv::Point2f &normPos)
{
    QMutexLocker locker(&m_mutex);
    if (!m_entries.contains(personId)) return;
    Entry &e = m_entries[personId];
    if (!feature.empty()) updateFeatureLocked(e, feature);
    e.lastSeenMs = nowMs;
    e.lastNormPos = normPos;
    e.seenCount++;
}

bool VisitorRegistry::absorb(int keepId, int dropId)
{
    QMutexLocker locker(&m_mutex);
    if (keepId == dropId) return false;
    if (!m_entries.contains(keepId) || !m_entries.contains(dropId)) return false;

    Entry &keep = m_entries[keepId];
    const Entry drop = m_entries[dropId];

    // 特征归并：保持运行平均的语义（按观测次数加权）
    if (!drop.feature.empty()) {
        if (keep.feature.empty() || keep.feature.total() != drop.feature.total()) {
            keep.feature = drop.feature.clone();
            keep.featCount = drop.featCount;
        } else {
            const double total = keep.featCount + drop.featCount;
            if (total > 0.0) {
                cv::Mat acc;
                cv::addWeighted(keep.feature, static_cast<double>(keep.featCount),
                                drop.feature, static_cast<double>(drop.featCount),
                                0.0, acc);
                acc /= total;
                keep.feature = acc;
                keep.featCount = static_cast<int>(total);
            }
        }
    }

    keep.lastSeenMs = std::max(keep.lastSeenMs, drop.lastSeenMs);
    keep.firstSeenMs = (keep.firstSeenMs > 0)
                           ? std::min(keep.firstSeenMs, drop.firstSeenMs)
                           : drop.firstSeenMs;
    if (drop.lastSeenMs >= keep.lastSeenMs && drop.lastNormPos.x >= 0.0f)
        keep.lastNormPos = drop.lastNormPos;
    keep.inside = keep.inside || drop.inside;
    keep.seenCount += drop.seenCount;

    m_entries.remove(dropId);

    LOG_INFO(QString("[VisitorRegistry] absorb V%1 -> V%2 (featCount=%3, seen=%4)")
                 .arg(dropId).arg(keepId).arg(keep.featCount).arg(keep.seenCount));
    return true;
}

void VisitorRegistry::markInside(int personId, bool inside)
{
    QMutexLocker locker(&m_mutex);
    if (!m_entries.contains(personId)) return;
    m_entries[personId].inside = inside;
    if (inside) m_entries[personId].lastSeenMs = QDateTime::currentMSecsSinceEpoch();
}

bool VisitorRegistry::isInside(int personId) const
{
    QMutexLocker locker(&m_mutex);
    return m_entries.contains(personId) && m_entries.value(personId).inside;
}

bool VisitorRegistry::contains(int personId) const
{
    QMutexLocker locker(&m_mutex);
    return m_entries.contains(personId);
}

VisitorRegistry::Entry VisitorRegistry::entry(int personId) const
{
    QMutexLocker locker(&m_mutex);
    return m_entries.value(personId);
}

cv::Mat VisitorRegistry::feature(int personId) const
{
    QMutexLocker locker(&m_mutex);
    return m_entries.contains(personId) ? m_entries.value(personId).feature.clone() : cv::Mat();
}

cv::Point2f VisitorRegistry::lastPos(int personId) const
{
    QMutexLocker locker(&m_mutex);
    return m_entries.contains(personId) ? m_entries.value(personId).lastNormPos
                                        : cv::Point2f(-1.0f, -1.0f);
}

qint64 VisitorRegistry::lastSeenMs(int personId) const
{
    QMutexLocker locker(&m_mutex);
    return m_entries.contains(personId) ? m_entries.value(personId).lastSeenMs : 0;
}

QVector<int> VisitorRegistry::ids() const
{
    QMutexLocker locker(&m_mutex);
    QVector<int> v;
    v.reserve(m_entries.size());
    for (auto it = m_entries.constBegin(); it != m_entries.constEnd(); ++it)
        v.append(it.key());
    return v;
}

int VisitorRegistry::count() const
{
    QMutexLocker locker(&m_mutex);
    return m_entries.size();
}

void VisitorRegistry::reset()
{
    QMutexLocker locker(&m_mutex);
    m_entries.clear();
    m_nextId = 1;
    LOG_INFO("[VisitorRegistry] reset");
}
