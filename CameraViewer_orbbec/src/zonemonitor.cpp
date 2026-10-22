#include "zonemonitor.h"
#include "logger.h"

#include <algorithm>
#include <cmath>
#include <QSet>

// ===== 采样/视频参数 =====
static constexpr int SAMPLE_INTERVAL = 2;        // 采样节拍（每 N 帧采一帧）
static constexpr int MAX_FEATURE_SAMPLES = 30;   // 每段最多保留的特征/裁剪帧数
static constexpr int VIDEO_MAX_WIDTH = 960;      // 合成视频的整帧最大宽度

namespace {

/// 求线段 (p0,p1) 与门线 (a,b) 的归一化交点（参数式），失败返回 false
bool segIntersect(const cv::Point2f &p0, const cv::Point2f &p1,
                  const cv::Point2f &a, const cv::Point2f &b,
                  cv::Point2f &out)
{
    const float r_x = p1.x - p0.x, r_y = p1.y - p0.y;
    const float s_x = b.x - a.x,  s_y = b.y - a.y;
    const float denom = r_x * s_y - r_y * s_x;
    if (std::fabs(denom) < 1e-9f) return false;
    const float t = ((a.x - p0.x) * s_y - (a.y - p0.y) * s_x) / denom;
    const float u = ((a.x - p0.x) * r_y - (a.y - p0.y) * r_x) / denom;
    if (t < -0.25f || t > 1.25f) return false;   // 允许少量外扩（轨迹抖动）
    if (u < -0.25f || u > 1.25f) return false;   // 交点在门线线段附近
    out = cv::Point2f(p0.x + t * r_x, p0.y + t * r_y);
    return true;
}

/// 点到门线的最近点（门线线段上）
cv::Point2f projectOnLine(const cv::Point2f &p, const DoorLine &l)
{
    if (l.length < 1e-6f) return l.a;
    float t = (p.x - l.a.x) * l.dir.x + (p.y - l.a.y) * l.dir.y;
    t = std::max(0.0f, std::min(l.length, t));
    return cv::Point2f(l.a.x + l.dir.x * t, l.a.y + l.dir.y * t);
}

} // namespace

ZoneMonitor::ZoneMonitor(QObject *parent)
    : QObject(parent)
{
}

// ===== 配置接口 =====
void ZoneMonitor::setReidExtractor(ReIDExtractor *extractor) { m_reidExtractor = extractor; }
void ZoneMonitor::setReidMatchThreshold(float threshold) { m_reidMatchThreshold = threshold; }
void ZoneMonitor::setConfirmFrames(int frames) { m_confirmFrames = qMax(1, frames); }
void ZoneMonitor::setMinDisplacement(float d) { m_minDisplacement = qMax(0.0f, d); }
void ZoneMonitor::setStartConfirmFrames(int frames) { m_startConfirmFrames = qMax(1, frames); }
void ZoneMonitor::setMinOverlapArea(float area) { m_minOverlapArea = qMax(0.0f, area); }
void ZoneMonitor::setReverseCooldownMs(qint64 ms) { m_reverseCooldownMs = qMax<qint64>(0, ms); }
void ZoneMonitor::setSameDirCooldownMs(qint64 ms) { m_sameDirCooldownMs = qMax<qint64>(0, ms); }
void ZoneMonitor::setEdgeMargin(float margin) { m_edgeMargin = qMax(0.0f, margin); }

void ZoneMonitor::setLineDeadband(float d) { m_lineDeadband = qMax(0.0f, d); }
void ZoneMonitor::setCorridorHalfWidth(float w) { m_corridorHalf = qMax(0.01f, w); }
void ZoneMonitor::setSideConfirmFrames(int frames) { m_sideConfirmFrames = qMax(1, frames); }
void ZoneMonitor::setInertiaMaxMs(qint64 ms) { m_inertiaMaxMs = qMax<qint64>(0, ms); }
void ZoneMonitor::setStitchWindowMs(qint64 ms) { m_stitchWindowMs = qMax<qint64>(0, ms); }
void ZoneMonitor::setMinTrackFrames(int frames) { m_minTrackFrames = qMax(1, frames); }

// ===== 门线构建 =====
void ZoneMonitor::buildDoorLines(const QVector<DetectionRegion> &regions)
{
    QVector<DoorLine> lines;

    for (const auto &r : regions) {
        QPointF pa, pb;
        if (!RegionManager::regionDoorLine(r, pa, pb)) continue;

        DoorLine l;
        l.regionId = r.id;
        l.entry = r.entryDirection;
        l.a = cv::Point2f(static_cast<float>(pa.x()), static_cast<float>(pa.y()));
        l.b = cv::Point2f(static_cast<float>(pb.x()), static_cast<float>(pb.y()));

        // 规范端点顺序（a.x <= b.x）以获得稳定的方向/法线
        if (l.a.x > l.b.x) std::swap(l.a, l.b);

        const cv::Point2f d = l.b - l.a;
        const float len = std::sqrt(d.x * d.x + d.y * d.y);
        if (len < 1e-6f) continue;
        l.dir = cv::Point2f(d.x / len, d.y / len);

        // 法线取向：始终指向 Y 增大方向（图像下方）→ signedSide>0 表示“内侧/下方”
        cv::Point2f n(l.dir.y, -l.dir.x);
        if (n.y < 0.0f) n = cv::Point2f(-n.x, -n.y);
        if (std::fabs(n.y) < 1e-6f && n.x < 0.0f) n = cv::Point2f(-n.x, -n.y);
        l.normal = n;
        l.length = len;

        lines.append(l);
    }

    // 内容未变化则跳过（避免逐帧 clear/重建产生的重复日志与开销）
    bool same = (lines.size() == m_lines.size());
    if (same) {
        for (int i = 0; i < lines.size(); ++i) {
            const DoorLine &a = lines[i];
            const DoorLine &b = m_lines[i];
            if (a.regionId != b.regionId
                || std::fabs(a.a.x - b.a.x) > 1e-6f || std::fabs(a.a.y - b.a.y) > 1e-6f
                || std::fabs(a.b.x - b.b.x) > 1e-6f || std::fabs(a.b.y - b.b.y) > 1e-6f) {
                same = false;
                break;
            }
        }
    }
    if (same && !lines.isEmpty()) return;

    m_lines = lines;

    if (m_lines.isEmpty()) {
        LOG_WARN("[ZoneMonitor] 未配置上下车分界线（门线数=0）：无法进行上/下车穿越判定，"
                 "请先在主界面绘制上下车分界线");
        return;
    }

    for (const auto &l : m_lines) {
        LOG_INFO(QString("[ZoneMonitor] DoorLine set: regionId=%1, A=(%2,%3) B=(%4,%5), "
                         "len=%6, normal=(%7,%8), dir=%9")
                     .arg(l.regionId)
                     .arg(l.a.x, 0, 'f', 4).arg(l.a.y, 0, 'f', 4)
                     .arg(l.b.x, 0, 'f', 4).arg(l.b.y, 0, 'f', 4)
                     .arg(l.length, 0, 'f', 4)
                     .arg(l.normal.x, 0, 'f', 3).arg(l.normal.y, 0, 'f', 3)
                     .arg(l.entry == EntryDirection::TopToBottom
                              ? "TopToBottom(从上往下A→B=下车)"
                              : "BottomToTop(从下往上B→A=下车)"));
    }
}

void ZoneMonitor::setChannelZones(const QVector<DetectionRegion> &regions)
{
    buildDoorLines(regions);
}

int ZoneMonitor::bestLineIndex(const cv::Point2f &p, float maxPerp, float spanMargin) const
{
    int best = -1;
    float bestPerp = maxPerp;
    for (int i = 0; i < m_lines.size(); ++i) {
        const DoorLine &l = m_lines[i];
        if (!l.withinSpan(p, spanMargin)) continue;
        const float perp = std::fabs(l.signedSide(p));
        if (perp < bestPerp) {
            bestPerp = perp;
            best = i;
        }
    }
    return best;
}

DoorSide ZoneMonitor::sideOf(float signedSide, float deadband)
{
    if (signedSide > deadband) return DoorSide::BSide;
    if (signedSide < -deadband) return DoorSide::ASide;
    return DoorSide::Unknown;
}

int ZoneMonitor::sideCode(DoorSide s)
{
    switch (s) {
    case DoorSide::ASide: return 0;
    case DoorSide::BSide: return 1;
    default: return -1;
    }
}

// ===== ReID =====
cv::Mat ZoneMonitor::extractFeature(const TrackedPerson &person, const cv::Mat &rawFrame) const
{
    if (!person.reidFeature.empty()) return person.reidFeature.clone();
    if (!rawFrame.empty() && m_reidExtractor && !person.bbox.empty()) {
        return m_reidExtractor->extract(rawFrame, person.bbox);
    }
    return cv::Mat();
}

int ZoneMonitor::matchOrCreateVisitor(const TrackedPerson &person, const cv::Mat &rawFrame)
{
    if (m_trackToVisitor.contains(person.trackId)) {
        return m_trackToVisitor[person.trackId];
    }

    cv::Mat reidFeature = extractFeature(person, rawFrame);

    int matchedVisitorId = -1;
    float bestSimilarity = 0.0f;

    for (auto it = m_visitorFeatures.begin(); it != m_visitorFeatures.end(); ++it) {
        const int vid = it.key();
        const cv::Mat &storedFeature = it.value();

        // 跳过「在车内」的 visitor（最后穿越=上车）：车内的人不会作为新 track 出现在车外，
        // 避免视野中多人同时出现时被 ReID 误匹配成同一个 visitor（共用 id / 混用视频）。
        if (m_lastCrossingTypeByVisitor.value(vid, 0) > 0) continue;

        // 仅与当前不在任何接近窗口内的访客匹配（已离开）
        bool isInside = false;
        for (auto t = m_tracks.begin(); t != m_tracks.end(); ++t) {
            if (t.value().visitorId == vid && t.value().state == CaptureState::Capturing) {
                isInside = true;
                break;
            }
        }
        if (!isInside && !storedFeature.empty() && !reidFeature.empty()) {
            const float similarity = m_reidExtractor->compare(reidFeature, storedFeature);
            if (similarity > bestSimilarity && similarity >= m_reidMatchThreshold) {
                bestSimilarity = similarity;
                matchedVisitorId = vid;
            }
        }
    }

    if (matchedVisitorId > 0) {
        m_trackToVisitor[person.trackId] = matchedVisitorId;
        if (!reidFeature.empty()) m_visitorFeatures[matchedVisitorId] = reidFeature.clone();
        LOG_INFO(QString("[ZoneMonitor] ReID RE-ENTRY: Visitor V%1 matched (similarity=%2) with track %3")
                     .arg(matchedVisitorId).arg(bestSimilarity, 0, 'f', 2).arg(person.trackId));
        return matchedVisitorId;
    }

    const int newVisitorId = m_nextVisitorId++;
    m_trackToVisitor[person.trackId] = newVisitorId;
    if (!reidFeature.empty()) m_visitorFeatures[newVisitorId] = reidFeature.clone();
    LOG_INFO(QString("[ZoneMonitor] New visitor V%1 created (track %2)")
                 .arg(newVisitorId).arg(person.trackId));
    return newVisitorId;
}

// ===== 采样 =====
void ZoneMonitor::sampleCapture(DoorTrackedPerson &t, const TrackedPerson &p,
                                const cv::Mat &rawFrame, int maxSamples)
{
    if (rawFrame.empty()) return;

    const cv::Rect safeBox = p.bbox & cv::Rect(0, 0, rawFrame.cols, rawFrame.rows);
    if (safeBox.empty()) return;

    const int padX = static_cast<int>(safeBox.width * 0.15);
    const int padY = static_cast<int>(safeBox.height * 0.15);
    const int x1 = std::max(0, safeBox.x - padX);
    const int y1 = std::max(0, safeBox.y - padY);
    const int x2 = std::min(rawFrame.cols, safeBox.x + safeBox.width + padX);
    const int y2 = std::min(rawFrame.rows, safeBox.y + safeBox.height + padY);
    if (x2 <= x1 || y2 <= y1) return;

    cv::Mat crop = rawFrame(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
    t.trackImages.append(crop);
    t.trackBoxes.append(safeBox);
    t.trackFeatures.append(extractFeature(p, rawFrame));

    // 整帧序列（缩放到 960 宽），用于合成上/下车视频段
    cv::Mat full = rawFrame.clone();
    if (full.cols > VIDEO_MAX_WIDTH) {
        const double scale = static_cast<double>(VIDEO_MAX_WIDTH) / full.cols;
        cv::resize(full, full, cv::Size(), scale, scale, cv::INTER_AREA);
    }
    t.trackFullFrames.append(full);

    while (t.trackImages.size() > maxSamples) {
        t.trackImages.removeFirst();
        t.trackBoxes.removeFirst();
        t.trackFeatures.removeFirst();
    }
    while (t.trackFullFrames.size() > maxSamples) {
        t.trackFullFrames.removeFirst();
    }
}

void ZoneMonitor::startCapture(DoorTrackedPerson &t, const TrackedPerson &p,
                               const DoorLine &line, const cv::Mat &rawFrame,
                               const cv::Point2f &pert, float signedSide, qint64 nowMs)
{
    Q_UNUSED(pert);
    t.state = CaptureState::Capturing;
    t.regionId = line.regionId;
    t.sampleTick = 0;
    t.trackImages.clear();
    t.trackBoxes.clear();
    t.trackFeatures.clear();
    t.trackFullFrames.clear();
    t.enterImage = rawFrame.empty() ? cv::Mat() : rawFrame.clone();
    t.enterFeature = extractFeature(p, rawFrame);
    t.enterBbox = p.bbox;
    t.lastBbox = p.bbox;
    t.enterTime = QDateTime::currentDateTime();
    t.enterSigned = signedSide;
    t.crossedEmitted = false;
    t.coasted = false;
    t.stitched = false;
    t.seedLayer = RecoveryLayer::Realtime;

    if (t.visitorId <= 0) t.visitorId = matchOrCreateVisitor(p, rawFrame);

    sampleCapture(t, p, rawFrame, MAX_FEATURE_SAMPLES);

    LOG_INFO(QString("[ZoneMonitor] START 接近门线: track=%1 V%2 region=%3 signed=%4 (开始整段采样)")
                 .arg(p.trackId).arg(t.visitorId).arg(line.regionId).arg(signedSide, 0, 'f', 4));
    Q_UNUSED(nowMs);
}

void ZoneMonitor::stopCapture(DoorTrackedPerson &t)
{
    if (t.state == CaptureState::Capturing) {
        t.state = CaptureState::Idle;
    }
    t.trackImages.clear();
    t.trackBoxes.clear();
    t.trackFeatures.clear();
    t.trackFullFrames.clear();
    t.sampleTick = 0;
}

// ===== 事件结算 =====
void ZoneMonitor::emitCrossing(DoorTrackedPerson &t, int trackId, const DoorLine &line,
                               DoorSide from, DoorSide to, RecoveryLayer layer,
                               const cv::Point2f &crossPoint, const cv::Mat &rawFrame,
                               qint64 nowMs, bool coasted, bool stitched)
{
    const bool aToB = (from == DoorSide::ASide && to == DoorSide::BSide);
    const bool bToA = (from == DoorSide::BSide && to == DoorSide::ASide);
    if (!aToB && !bToA) return;

    RecoveryLayer eff = layer;
    if (stitched && eff == RecoveryLayer::Realtime) eff = RecoveryLayer::Stitch;
    if (coasted && eff == RecoveryLayer::Realtime)  eff = RecoveryLayer::Inertia;

    // 方向语义（上下车定义）——【相机装反 / 视角反转后的新定义】：
    //   A(顶/上侧) → B(底/下侧)：从上往下穿越分界线 = 下车；
    //   B(底/下侧) → A(顶/上侧)：从下往上穿越分界线 = 上车。
    //   entryDirection==TopToBottom（默认）：A→B=下车，B→A=上车；
    //   entryDirection==BottomToTop：整体取反（A→B=上车，B→A=下车）。
    const bool boarding = (line.entry == EntryDirection::TopToBottom) ? bToA : aToB;

    const float sideExit = line.signedSide(crossPoint);
    const float dyMove = sideExit - t.enterSigned;

    // 位移门槛（仅实时判定时严格要求，避免“擦线抖动”）
    if (eff == RecoveryLayer::Realtime && std::fabs(dyMove) < m_minDisplacement) {
        LOG_INFO(QString("[ZoneMonitor] 忽略微小跨越(擦线抖动): V%1 track=%2 region=%3 "
                         "from=%4 to=%5 dy=%6 < 门槛%7")
                     .arg(t.visitorId).arg(trackId).arg(line.regionId)
                     .arg(sideCode(from)).arg(sideCode(to))
                     .arg(dyMove, 0, 'f', 4).arg(m_minDisplacement, 0, 'f', 4));
        return;
    }

    // 防误判：同人反向穿越冷却窗口（不计数）
    // 注意：同方向连续穿越（如门线附近逗留导致的连续两次上车）不在此抑制，
    // 每次真实穿越都应当计数，person 编号在 MainWindow 事件处理中每次 +1。
    if (t.visitorId > 0) {
        const int type = boarding ? +1 : -1;
        qint64 lastMs = 0; int lastType = 0;
        if (isReverseInCooldown(t.visitorId, type, nowMs, &lastMs, &lastType)) {
            LOG_WARN(QString("[ZoneMonitor] SUPPRESS 反向穿越(疑似抖动/误判): V%1 track=%2 region=%3 "
                             "本次=%4 startSigned=%5 exitSigned=%6 dy=%7 | 上次=%8 间隔=%9ms < 冷却=%10ms")
                         .arg(t.visitorId).arg(trackId).arg(line.regionId)
                         .arg(boarding ? "上车候选" : "下车候选")
                         .arg(t.enterSigned, 0, 'f', 4).arg(sideExit, 0, 'f', 4).arg(dyMove, 0, 'f', 4)
                         .arg(lastType > 0 ? "上车" : "下车")
                         .arg(nowMs - lastMs).arg(m_reverseCooldownMs));
            t.crossedEmitted = true;
            stopCapture(t);
            return;
        }
    }

    const float centerY = line.center().y;
    const float topY = std::min(line.a.y, line.b.y);
    const float bottomY = std::max(line.a.y, line.b.y);

    ZoneEvent evt;
    evt.visitorId   = t.visitorId;
    evt.trackId     = trackId;
    evt.regionId    = line.regionId;
    evt.type        = boarding ? ZoneEventType::Boarding : ZoneEventType::Alighting;
    evt.enterTime   = t.enterTime.isValid() ? t.enterTime : QDateTime::currentDateTime();
    evt.exitTime    = QDateTime::currentDateTime();
    evt.entrySide   = sideCode(from);
    evt.exitSide    = sideCode(to);
    evt.enterImage  = t.enterImage.clone();
    evt.exitImage   = rawFrame.empty() ? cv::Mat() : rawFrame.clone();
    evt.enterBbox   = t.enterBbox;
    evt.exitBbox    = t.lastBbox;
    evt.enterFeature = t.enterFeature.clone();
    evt.exitFeature  = ReIDExtractor::averageFeatures(t.trackFeatures);
    if (evt.exitFeature.empty() && !t.trackFeatures.isEmpty())
        evt.exitFeature = t.trackFeatures.last().clone();
    evt.trackImages     = t.trackImages;
    evt.trackBoxes      = t.trackBoxes;
    evt.trackFeatures   = t.trackFeatures;
    evt.trackFullFrames = t.trackFullFrames;
    // 诊断字段
    evt.startNormY      = centerY + t.enterSigned;
    evt.endNormY        = centerY + sideExit;
    evt.zoneTopNormY    = topY;
    evt.zoneBottomNormY = bottomY;
    evt.zoneCenterNormY = centerY;
    evt.moveDy          = dyMove;
    evt.trackFrames     = static_cast<int>(t.trackImages.size());
    evt.durationMs      = evt.enterTime.isValid()
                            ? evt.enterTime.msecsTo(evt.exitTime) : 0;
    // DoorCounter v6
    evt.recoveryLayer = static_cast<int>(eff);
    evt.recoveryName  = recoveryLayerName(eff);
    evt.crossNormX    = crossPoint.x;
    evt.crossNormY    = crossPoint.y;
    evt.sideEnter     = t.enterSigned;
    evt.sideExit      = sideExit;
    evt.coasted       = coasted;
    evt.stitched      = stitched;
    evt.fallback      = (eff == RecoveryLayer::Fallback);
    switch (eff) {
    case RecoveryLayer::Realtime: evt.confidence = 1.00f; break;
    case RecoveryLayer::Inertia:  evt.confidence = 0.80f; break;
    case RecoveryLayer::Stitch:   evt.confidence = 0.75f; break;
    case RecoveryLayer::Fallback: evt.confidence = 0.60f; break;
    }

    LOG_INFO(QString("[ZoneMonitor] CROSSING DECISION [%1 conf=%2]: V%3 track=%4 region=%5 | "
                     "门线 A=(%6,%7) B=(%8,%9) | 侧别 %10→%11 | startSigned=%12 exitSigned=%13 dy=%14 | "
                     "cross=(%15,%16) frames=%17 durationMs=%18 → %19")
                 .arg(evt.recoveryName).arg(evt.confidence, 0, 'f', 2)
                 .arg(evt.visitorId).arg(trackId).arg(evt.regionId)
                 .arg(line.a.x, 0, 'f', 4).arg(line.a.y, 0, 'f', 4)
                 .arg(line.b.x, 0, 'f', 4).arg(line.b.y, 0, 'f', 4)
                 .arg(from == DoorSide::ASide ? "A" : "B")
                 .arg(to == DoorSide::ASide ? "A" : "B")
                 .arg(t.enterSigned, 0, 'f', 4).arg(sideExit, 0, 'f', 4).arg(dyMove, 0, 'f', 4)
                 .arg(crossPoint.x, 0, 'f', 4).arg(crossPoint.y, 0, 'f', 4)
                 .arg(evt.trackFrames).arg(evt.durationMs)
                 .arg(boarding ? "BOARDING(上车)" : "ALIGHTING(下车)"));

    if (boarding) {
        m_boardingCount++;
        LOG_INFO(QString("[ZoneMonitor] ★ BOARDING (上车): V%1 region=%2 [%3] frames=%4")
                     .arg(evt.visitorId).arg(evt.regionId).arg(evt.recoveryName).arg(evt.trackFrames));
        m_events.append(evt);
        emit boardingEvent(evt);
        emit zoneCrossingEvent(evt);
    } else {
        m_alightingCount++;
        LOG_INFO(QString("[ZoneMonitor] ★ ALIGHTING (下车): V%1 region=%2 [%3] frames=%4")
                     .arg(evt.visitorId).arg(evt.regionId).arg(evt.recoveryName).arg(evt.trackFrames));
        m_events.append(evt);
        emit alightingEvent(evt);
        emit zoneCrossingEvent(evt);
    }

    recordCrossing(t.visitorId, boarding ? +1 : -1, nowMs);

    // 结算：本次接近窗口结束（需重新离开走廊后才再次武装）
    t.crossedEmitted = true;
    t.firstSide = to;
    t.lastSide  = to;
    t.pendingSide = DoorSide::Unknown;
    t.sideConfirm = 0;
    stopCapture(t);
    t.enterTime = QDateTime();
}

bool ZoneMonitor::tryCoastCrossing(DoorTrackedPerson &t, int trackId, const DoorLine &line,
                                   const cv::Mat &rawFrame, qint64 nowMs, float dtSec)
{
    if (t.crossedEmitted) return false;
    if (t.lastSide == DoorSide::Unknown) return false;
    if (dtSec <= 0.0f || dtSec > m_inertiaMaxMs / 1000.0f * 1.5f) return false;

    const cv::Point2f pred(t.lastCenter.x + t.velocity.x * dtSec,
                           t.lastCenter.y + t.velocity.y * dtSec);
    const float predSigned = line.signedSide(pred);
    const DoorSide predSide = sideOf(predSigned, m_lineDeadband);
    if (predSide == DoorSide::Unknown || predSide == t.lastSide) return false;
    if (!line.withinSpan(pred, 0.30f)) return false;

    // 速度方向必须朝向门线另一侧
    const float vn = t.velocity.x * line.normal.x + t.velocity.y * line.normal.y;
    if (t.lastSide == DoorSide::ASide && vn <= 0.0f) return false;
    if (t.lastSide == DoorSide::BSide && vn >= 0.0f) return false;

    cv::Point2f crossPt;
    if (!segIntersect(t.lastCenter, pred, line.a, line.b, crossPt))
        crossPt = projectOnLine(pred, line);

    LOG_WARN(QString("[ZoneMonitor] L2 惯性滑行: V%1 track=%2 region=%3 丢失%4ms 按速度外推跨越")
                 .arg(t.visitorId).arg(trackId).arg(line.regionId)
                 .arg(static_cast<qint64>(dtSec * 1000)));
    emitCrossing(t, trackId, line, t.lastSide, predSide, RecoveryLayer::Inertia,
                 crossPt, rawFrame, nowMs, /*coasted*/ true, /*stitched*/ false);
    return true;
}

void ZoneMonitor::finalizeLost(DoorTrackedPerson &t, int trackId,
                               const cv::Mat &rawFrame, qint64 nowMs)
{
    if (t.crossedEmitted) return;

    const DoorLine *line = nullptr;
    for (const auto &l : m_lines) {
        if (l.regionId == t.regionId) { line = &l; break; }
    }
    if (!line) return;
    if (t.lastSeenMs <= 0) return;

    const float dtSec = std::min<qint64>(m_inertiaMaxMs, nowMs - t.lastSeenMs) / 1000.0f;

    // ===== L2 惯性滑行 =====
    if (dtSec > 0.0f && tryCoastCrossing(t, trackId, *line, rawFrame, nowMs, dtSec)) return;

    // ===== L4 兜底策略：一侧出现、另一侧消失 =====
    if (t.firstSide != DoorSide::Unknown && t.lastSide != DoorSide::Unknown
        && t.firstSide != t.lastSide) {
        cv::Point2f crossPt = projectOnLine(t.lastCenter, *line);
        LOG_WARN(QString("[ZoneMonitor] L4 兜底策略: V%1 track=%2 region=%3 "
                         "firstSide=%4 lastSide=%5（未经实时确认的跨越，按兜底计入）")
                     .arg(t.visitorId).arg(trackId).arg(t.regionId)
                     .arg(t.firstSide == DoorSide::ASide ? "A" : "B")
                     .arg(t.lastSide == DoorSide::ASide ? "A" : "B"));
        emitCrossing(t, trackId, *line, t.firstSide, t.lastSide, RecoveryLayer::Fallback,
                     crossPt, rawFrame, nowMs, /*coasted*/ false, /*stitched*/ false);
        return;
    }

    // ===== L3 轨迹拼接：半程（同侧未跨完）→ 记录锚点 =====
    if (t.firstSide != DoorSide::Unknown && t.firstSide == t.lastSide) {
        StitchAnchor an;
        an.regionId = t.regionId;
        an.visitorId = t.visitorId;
        const qint64 dt2 = std::min<qint64>(m_inertiaMaxMs * 2, nowMs - t.lastSeenMs);
        const float dt2s = dt2 / 1000.0f;
        an.predicted = cv::Point2f(t.lastCenter.x + t.velocity.x * dt2s,
                                   t.lastCenter.y + t.velocity.y * dt2s);
        an.velocity = t.velocity;
        const float vn = t.velocity.x * line->normal.x + t.velocity.y * line->normal.y;
        an.dirSign = vn > 1e-4f ? +1 : (vn < -1e-4f ? -1 : 0);
        an.fromSide = t.firstSide;
        an.atMs = nowMs;
        an.enterMs = t.enterTime.isValid() ? t.enterTime.toMSecsSinceEpoch() : nowMs;
        m_anchors.append(an);
        LOG_INFO(QString("[ZoneMonitor] L3 记录轨迹拼接锚点: V%1 region=%2 fromSide=%3 "
                         "pred=(%4,%5) dir=%6 (窗口%7ms 内可续接)")
                     .arg(t.visitorId).arg(t.regionId)
                     .arg(t.firstSide == DoorSide::ASide ? "A" : "B")
                     .arg(an.predicted.x, 0, 'f', 4).arg(an.predicted.y, 0, 'f', 4)
                     .arg(an.dirSign).arg(m_stitchWindowMs));
    }
}

bool ZoneMonitor::tryStitchInherit(DoorTrackedPerson &t, const DoorLine &line,
                                   const cv::Point2f &pos, qint64 nowMs)
{
    if (m_anchors.isEmpty()) return false;

    int best = -1;
    float bestD = m_stitchRadius;
    for (int i = 0; i < m_anchors.size(); ++i) {
        const StitchAnchor &an = m_anchors[i];
        if (an.regionId != line.regionId) continue;
        if (nowMs - an.atMs > m_stitchWindowMs) continue;

        const cv::Point2f dp = pos - an.predicted;
        const float d = std::sqrt(dp.x * dp.x + dp.y * dp.y);
        if (d > bestD) continue;

        // 方向一致性：新轨迹的门线法向速度应与锚点方向一致
        if (an.dirSign != 0) {
            const float vn = t.velocity.x * line.normal.x + t.velocity.y * line.normal.y;
            if ((an.dirSign > 0 && vn < 0.0f) || (an.dirSign < 0 && vn > 0.0f)) continue;
        }
        bestD = d;
        best = i;
    }
    if (best < 0) return false;

    const StitchAnchor an = m_anchors.takeAt(best);
    t.firstSide = an.fromSide;
    t.stitched = true;
    t.seedLayer = RecoveryLayer::Stitch;
    if (an.enterMs > 0) t.enterTime = QDateTime::fromMSecsSinceEpoch(an.enterMs);

    LOG_INFO(QString("[ZoneMonitor] L3 轨迹拼接续接: trackV%1 region=%2 fromSide=%3 "
                     "(沿用断裂轨迹的外侧起点, 距锚点 %4ms)")
                 .arg(t.visitorId).arg(line.regionId)
                 .arg(an.fromSide == DoorSide::ASide ? "A" : "B")
                 .arg(nowMs - an.atMs));
    return true;
}

void ZoneMonitor::pruneAnchors(qint64 nowMs)
{
    for (int i = m_anchors.size() - 1; i >= 0; --i) {
        if (nowMs - m_anchors[i].atMs >= m_stitchWindowMs) {
            LOG_INFO(QString("[ZoneMonitor] L3 锚点过期: V%1 region=%2 (未等到续接)")
                         .arg(m_anchors[i].visitorId).arg(m_anchors[i].regionId));
            m_anchors.removeAt(i);
        }
    }
}

// ===== 主处理 =====
void ZoneMonitor::processFrame(QVector<TrackedPerson> &persons,
                               const cv::Mat &rawFrame,
                               RegionManager *regionManager,
                               int imgW, int imgH)
{
    if (!regionManager || imgW <= 0 || imgH <= 0) return;

    const qint64 nowMs = QDateTime::currentMSecsSinceEpoch();
    pruneAnchors(nowMs);

    // 刷新门线定义（动态响应区域变化）
    const auto regions = regionManager->regions();
    if (m_lines.size() != regions.size()) {
        buildDoorLines(regions);
    } else {
        // 区域内容可能变化（例如编辑顶点）→ 逐帧轻量校验
        buildDoorLines(regions);
    }

    const float assocPerp = std::max(m_corridorHalf * 2.0f, 0.22f);

    // 收集当前活跃 trackId
    QSet<int> activeTrackIds;
    for (const auto &p : persons) activeTrackIds.insert(p.trackId);

    for (int i = 0; i < persons.size(); ++i) {
        TrackedPerson &p = persons[i];

        // ===== 轨迹丢失帧（卡尔曼外推，detected=false）=====
        if (!p.detected) {
            if (m_tracks.contains(p.trackId)) {
                DoorTrackedPerson &t = m_tracks[p.trackId];
                if (t.regionId >= 0 && !t.crossedEmitted && t.lastSeenMs > 0) {
                    for (const auto &l : m_lines) {
                        if (l.regionId != t.regionId) continue;
                        const float dtSec = std::min<qint64>(m_inertiaMaxMs,
                                                             nowMs - t.lastSeenMs) / 1000.0f;
                        if (dtSec > 0.0f)
                            tryCoastCrossing(t, p.trackId, l, rawFrame, nowMs, dtSec);
                        break;
                    }
                }
                p.isInsideRegion = (t.state == CaptureState::Capturing);
            }
            continue;
        }

        const float normCx = p.center.x / static_cast<float>(imgW);
        const float normCy = p.center.y / static_cast<float>(imgH);
        const cv::Point2f pos(normCx, normCy);

        if (!m_tracks.contains(p.trackId)) m_tracks.insert(p.trackId, DoorTrackedPerson());
        DoorTrackedPerson &t = m_tracks[p.trackId];
        t.seenFrames++;   // 该 track 被真实检测到的帧数（建档预热用）

        // ===== 运动估计（归一化单位/秒，EMA 平滑）=====
        const qint64 dtMs = (t.lastSeenMs > 0) ? (nowMs - t.lastSeenMs) : 0;
        if (t.hasPrevCenter && dtMs > 0 && dtMs < 1000) {
            const float dt = dtMs / 1000.0f;
            const cv::Point2f inst((pos.x - t.lastCenter.x) / dt, (pos.y - t.lastCenter.y) / dt);
            t.velocity = cv::Point2f(0.6f * t.velocity.x + 0.4f * inst.x,
                                     0.6f * t.velocity.y + 0.4f * inst.y);
        }
        if (t.firstSeenMs == 0) t.firstSeenMs = nowMs;
        t.prevCenter = t.lastCenter;
        t.lastCenter = pos;
        t.hasPrevCenter = true;
        t.lastSeenMs = nowMs;

        // ===== 关联最近门线 =====
        const int li = bestLineIndex(pos, assocPerp, 0.40f);

        if (li < 0) {
            // 远离所有门线：重新武装（允许下一次真实穿越被计数），并停止采样
            if (t.state == CaptureState::Capturing) stopCapture(t);
            t.crossedEmitted = false;
            t.firstSide = DoorSide::Unknown;
            t.stitched = false;
            t.seedLayer = RecoveryLayer::Realtime;
            p.isInsideRegion = false;
            continue;
        }

        const DoorLine &line = m_lines[li];
        const float signedSide = line.signedSide(pos);
        t.lastSigned = signedSide;
        const DoorSide cand = sideOf(signedSide, m_lineDeadband);
        const bool inCorridor = std::fabs(signedSide) <= m_corridorHalf
                                && line.withinSpan(pos, 0.20f);

        // ===== L1 实时判断：侧别维护与翻转 =====
        if (cand != DoorSide::Unknown) {
            if (t.lastSide == DoorSide::Unknown) {
                t.lastSide = cand;
                t.sideConfirm = 1;
                if (t.firstSide == DoorSide::Unknown) t.firstSide = cand;
            } else if (cand == t.lastSide) {
                t.pendingSide = cand;
                t.sideConfirm = 0;
            } else {
                // 候选翻转 → 连续确认
                if (t.pendingSide != cand) {
                    t.pendingSide = cand;
                    t.sideConfirm = 1;
                } else {
                    t.sideConfirm++;
                }
                if (t.sideConfirm >= m_sideConfirmFrames) {
                    if (line.withinSpan(pos, 0.15f) && !t.crossedEmitted) {
                        cv::Point2f crossPt;
                        if (!segIntersect(t.prevCenter, pos, line.a, line.b, crossPt))
                            crossPt = projectOnLine(pos, line);
                        t.lastCrossX = crossPt.x;
                        t.lastCrossY = crossPt.y;
                        emitCrossing(t, p.trackId, line, t.lastSide, cand,
                                     RecoveryLayer::Realtime, crossPt, rawFrame,
                                     nowMs, t.coasted, t.stitched);
                    }
                    t.lastSide = cand;
                    t.pendingSide = DoorSide::Unknown;
                    t.sideConfirm = 0;
                }
            }
        }

        // ===== 整段采样窗口（接近走廊）=====
        if (!t.crossedEmitted) {
            if (t.state == CaptureState::Idle && inCorridor) {
                // 轨迹预热：不足 m_minTrackFrames 帧不建档/不采样，
                // 避免空场景的单帧/两帧偶发误检被当作乘客（出现 V1）。
                if (t.seenFrames < m_minTrackFrames) {
                    // 预热中，跳过本帧（等后续帧确认）
                } else {
                    startCapture(t, p, line, rawFrame, pos, signedSide, nowMs);
                    // 记录进入侧（用于诊断）；若未能确定则尝试 L3 拼接续接
                    if (t.firstSide == DoorSide::Unknown) {
                        if (!tryStitchInherit(t, line, pos, nowMs))
                            t.firstSide = cand;   // 可能仍为 Unknown，将在后续帧补齐
                    }
                }
            } else if (t.state == CaptureState::Capturing) {
                if (std::fabs(signedSide) <= m_corridorHalf * 1.4f) {
                    t.lastBbox = p.bbox;
                    if ((t.sampleTick++ % SAMPLE_INTERVAL) == 0)
                        sampleCapture(t, p, rawFrame, MAX_FEATURE_SAMPLES);
                } else {
                    stopCapture(t);   // 离开走廊，停止采样（未跨越则丢弃该段）
                }
            }
        } else if (std::fabs(signedSide) > m_corridorHalf * 1.6f) {
            // 已结算且远离门线 → 重新武装，等待下一次真实穿越
            t.crossedEmitted = false;
            t.firstSide = DoorSide::Unknown;
            t.lastSide = (cand != DoorSide::Unknown) ? cand : t.lastSide;
            t.stitched = false;
            t.seedLayer = RecoveryLayer::Realtime;
        }

        p.visitorId = t.visitorId;
        p.isInsideRegion = (t.state == CaptureState::Capturing);
    }

    // ===== 清理不活跃轨迹（立即结算：L2/L3/L4）=====
    QList<int> staleIds;
    for (auto it = m_tracks.begin(); it != m_tracks.end(); ++it) {
        if (!activeTrackIds.contains(it.key())) staleIds.append(it.key());
    }
    for (int id : staleIds) {
        finalizeLost(m_tracks[id], id, cv::Mat(), nowMs);
        m_tracks.remove(id);
    }
}

void ZoneMonitor::setCounters(int boarding, int alighting)
{
    m_boardingCount = boarding;
    m_alightingCount = alighting;
    LOG_INFO(QString("[ZoneMonitor] 恢复累计计数: 上车=%1 下车=%2")
                 .arg(boarding).arg(alighting));
}

void ZoneMonitor::reset()
{
    m_tracks.clear();
    m_trackToVisitor.clear();
    m_visitorFeatures.clear();
    m_events.clear();
    m_boardingCount = 0;
    m_alightingCount = 0;
    m_nextVisitorId = 1;
    m_lastCrossingMsByVisitor.clear();
    m_lastCrossingTypeByVisitor.clear();
    m_anchors.clear();
    LOG_INFO("[ZoneMonitor] All state reset");
}

void ZoneMonitor::recordCrossing(int visitorId, int type, qint64 nowMs)
{
    if (visitorId <= 0) return;
    m_lastCrossingMsByVisitor[visitorId] = nowMs;
    m_lastCrossingTypeByVisitor[visitorId] = type;
}

bool ZoneMonitor::isReverseInCooldown(int visitorId, int type, qint64 nowMs,
                                      qint64 *outLastMs, int *outLastType) const
{
    if (visitorId <= 0) return false;
    if (!m_lastCrossingMsByVisitor.contains(visitorId)) return false;

    const qint64 lastMs = m_lastCrossingMsByVisitor.value(visitorId);
    const int lastType = m_lastCrossingTypeByVisitor.value(visitorId, 0);
    if (outLastMs) *outLastMs = lastMs;
    if (outLastType) *outLastType = lastType;

    if (lastType == 0 || lastType == type) return false;
    return (nowMs - lastMs) < m_reverseCooldownMs;
}
