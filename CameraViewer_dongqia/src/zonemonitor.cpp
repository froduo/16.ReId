#include "zonemonitor.h"
#include "logger.h"

ZoneMonitor::ZoneMonitor(QObject *parent)
    : QObject(parent)
{
}

void ZoneMonitor::setVisitorRegistry(VisitorRegistry *registry)
{
    m_registry = registry;
}

void ZoneMonitor::setReidExtractor(ReIDExtractor *extractor)
{
    m_reidExtractor = extractor;
}

void ZoneMonitor::setReidMatchThreshold(float threshold)
{
    m_reidMatchThreshold = threshold;
    if (m_registry) m_registry->setBaseThreshold(threshold);
}

void ZoneMonitor::setConfirmFrames(int frames)
{
    m_confirmFrames = qMax(1, frames);
}

void ZoneMonitor::setChannelZones(const QVector<DetectionRegion> &regions)
{
    m_zones.clear();

    for (const auto &region : regions) {
        if (region.polygon.size() < 3) continue;

        ChannelZone zone;
        zone.regionId = region.id;

        float minY = 1.0f, maxY = 0.0f;
        float sumX = 0.0f;
        int count = 0;

        for (const QPointF &pt : region.polygon) {
            if (pt.y() < minY) minY = pt.y();
            if (pt.y() > maxY) maxY = pt.y();
            sumX += pt.x();
            count++;
        }

        zone.topY = minY;       // A端（顶端，归一化Y最小）
        zone.bottomY = maxY;    // B端（底端，归一化Y最大）
        zone.centerX = (count > 0) ? sumX / count : 0.5f;

        m_zones.append(zone);

        LOG_INFO(QString("[ZoneMonitor] Channel zone set: regionId=%1, A(topY)=%2, B(bottomY)=%3")
                     .arg(zone.regionId)
                     .arg(zone.topY, 0, 'f', 4)
                     .arg(zone.bottomY, 0, 'f', 4));
    }
}

int ZoneMonitor::determineSide(float normY, const ChannelZone &zone) const
{
    float midY = (zone.topY + zone.bottomY) / 2.0f;
    return (normY < midY) ? 0 : 1;  // 0=A（顶端），1=B（底端）
}

int ZoneMonitor::determineEntrySide(float prevNormY, float currNormY, const ChannelZone &zone) const
{
    if (prevNormY < zone.topY) {
        return 0;  // 从A端（顶端）进入
    } else if (prevNormY > zone.bottomY) {
        return 1;  // 从B端（底端）进入
    }

    float dy = currNormY - prevNormY;
    if (dy > 0) {
        return 0;  // 向下移动 → 从上方（A端）进入
    } else if (dy < 0) {
        return 1;  // 向上移动 → 从下方（B端）进入
    }

    return determineSide(prevNormY, zone);
}

int ZoneMonitor::determineExitSide(float prevNormY, float currNormY, const ChannelZone &zone) const
{
    if (currNormY < zone.topY) {
        return 0;  // 从A端（顶端）离开
    } else if (currNormY > zone.bottomY) {
        return 1;  // 从B端（底端）离开
    }

    float dy = currNormY - prevNormY;
    if (dy < 0) {
        return 0;  // 向上移动 → 从A端（顶端）离开
    } else if (dy > 0) {
        return 1;  // 向下移动 → 从B端（底端）离开
    }

    return determineSide(currNormY, zone);
}

int ZoneMonitor::resolvePersonId(const TrackedPerson &person, const cv::Mat &rawFrame,
                                 const cv::Point2f &normPos, qint64 nowMs)
{
    // (1) 跟踪器已解析出稳定身份：直接采用
    //     这样「跟踪 ID == 上下车 ID == 显示 ID」，同一个人不会因轨迹重建而换号
    if (person.visitorId > 0) {
        const int pid = person.visitorId;
        m_trackToVisitor[person.trackId] = pid;
        if (m_registry) {
            cv::Mat f = person.reidFeature;
            if (f.empty() && !rawFrame.empty() && m_reidExtractor)
                f = m_reidExtractor->extract(rawFrame, person.bbox);
            m_registry->reinforce(pid, f, nowMs, normPos);
        }
        return pid;
    }

    // (2) 本轨迹已有映射：复用映射，并用当前特征强化画廊
    if (m_trackToVisitor.contains(person.trackId)) {
        const int pid = m_trackToVisitor.value(person.trackId);
        if (pid > 0 && m_registry) {
            cv::Mat f = person.reidFeature;
            if (f.empty() && !rawFrame.empty() && m_reidExtractor)
                f = m_reidExtractor->extract(rawFrame, person.bbox);
            m_registry->reinforce(pid, f, nowMs, normPos);
        }
        return pid;
    }

    cv::Mat feature = person.reidFeature;
    if (feature.empty() && !rawFrame.empty() && m_reidExtractor)
        feature = m_reidExtractor->extract(rawFrame, person.bbox);

    // (3) 注册表解析（跟踪器尚未给出身份时的兜底）
    if (m_registry) {
        const VisitorRegistry::MatchInfo mi = m_registry->resolveEx(feature, nowMs, normPos);
        if (mi.personId > 0) m_trackToVisitor[person.trackId] = mi.personId;
        return mi.personId;
    }

    // (4) 回退：无共享注册表时的本地编号
    const int newId = m_nextVisitorId++;
    m_trackToVisitor[person.trackId] = newId;
    return newId;
}

void ZoneMonitor::expirePendingCrossings(qint64 nowMs)
{
    QList<int> expired;
    for (auto it = m_pendingCrossings.constBegin(); it != m_pendingCrossings.constEnd(); ++it) {
        if (nowMs - it.value().lastTouchMs > PENDING_GRACE_MS)
            expired.append(it.key());
    }
    for (int pid : expired) {
        LOG_INFO(QString("[ZoneMonitor] pending crossing V%1 expired (no resume)")
                     .arg(pid));
        if (m_registry) m_registry->markInside(pid, false);
        m_pendingCrossings.remove(pid);
    }
}

void ZoneMonitor::processFrame(QVector<TrackedPerson> &persons,
                                const cv::Mat &rawFrame,
                                RegionManager *regionManager,
                                int imgW, int imgH)
{
    if (!regionManager || imgW <= 0 || imgH <= 0) return;

    const qint64 nowMs = QDateTime::currentMSecsSinceEpoch();

    // 刷新通道区域定义（动态响应区域变化）
    auto regions = regionManager->regions();
    if (m_zones.isEmpty() || m_zones.size() != regions.size()) {
        setChannelZones(regions);
    }

    expirePendingCrossings(nowMs);

    // 收集当前活跃的 trackId
    QSet<int> activeTrackIds;
    for (const auto &p : persons) {
        activeTrackIds.insert(p.trackId);
    }

    // 清理不活跃的通道追踪状态（注意：不清 m_pendingCrossings，用于跨轨迹“续接”）
    QList<int> staleIds;
    for (auto it = m_channelTracks.begin(); it != m_channelTracks.end(); ++it) {
        if (!activeTrackIds.contains(it.key())) {
            staleIds.append(it.key());
        }
    }
    for (int id : staleIds) {
        m_channelTracks.remove(id);
        // 轨迹消失，解除 track 映射（身份仍保留在注册表/待完成穿越里）
        m_trackToVisitor.remove(id);
    }

    for (int i = 0; i < persons.size(); ++i) {
        TrackedPerson &p = persons[i];

        const float normCx = p.center.x / imgW;
        const float normCy = p.center.y / imgH;
        const float prevNormCy = p.prevCenter.y / imgH;
        const cv::Point2f normPos(normCx, normCy);

        int currentRegionId = regionManager->pointInWhichRegion(QPointF(normCx, normCy));
        const bool nowInRegion = (currentRegionId > 0);

        const ChannelZone *currentZone = nullptr;
        for (const auto &z : m_zones) {
            if (z.regionId == currentRegionId) { currentZone = &z; break; }
        }

        if (!m_channelTracks.contains(p.trackId)) {
            m_channelTracks[p.trackId] = ChannelTrackedPerson();
        }
        ChannelTrackedPerson &ctp = m_channelTracks[p.trackId];

        switch (ctp.state) {
        case ChannelTrackState::Outside: {
            if (nowInRegion && currentZone) {
                // 解析稳定身份（跨轨迹唯一）
                const int pid = resolvePersonId(p, rawFrame, normPos, nowMs);
                ctp.visitorId = pid;
                ctp.regionId = currentRegionId;
                p.visitorId = pid;

                // 若该人此前已有“未完成穿越”且仍在宽限期内 → 续接，而不是新建一次穿越
                if (pid > 0 && m_pendingCrossings.contains(pid)) {
                    PendingCrossing &pc = m_pendingCrossings[pid];
                    if (pc.regionId == currentRegionId &&
                        (nowMs - pc.lastTouchMs) <= PENDING_GRACE_MS) {
                        ctp.state = ChannelTrackState::Inside;
                        ctp.confirmFrames = m_confirmFrames;
                        ctp.entrySide = pc.entrySide;
                        ctp.enterTime = pc.enterTime;
                        ctp.enterImage = pc.enterImage;
                        pc.lastTouchMs = nowMs;
                        if (m_registry) m_registry->markInside(pid, true);
                        LOG_INFO(QString("[ZoneMonitor] RESUME crossing V%1 track=%2 region=%3 side=%4 enter=%5")
                                     .arg(pid).arg(p.trackId).arg(currentRegionId)
                                     .arg(pc.entrySide == 0 ? "A(top)" : "B(bottom)")
                                     .arg(pc.enterTime.toString("HH:mm:ss.zzz")));
                        break;
                    }
                }

                // 新的穿越：开始进入确认
                ctp.state = ChannelTrackState::Entering;
                ctp.confirmFrames = 1;
                ctp.entrySide = determineEntrySide(prevNormCy, normCy, *currentZone);
                LOG_INFO(QString("[ZoneMonitor] Track %1 START ENTERING region %2 (V%3, entrySide=%4)")
                             .arg(p.trackId).arg(currentRegionId).arg(pid)
                             .arg(ctp.entrySide == 0 ? "A(top)" : "B(bottom)"));
            }
            break;
        }
        case ChannelTrackState::Entering: {
            if (nowInRegion) {
                ctp.confirmFrames++;
                if (ctp.confirmFrames >= m_confirmFrames) {
                    ctp.state = ChannelTrackState::Inside;
                    ctp.enterTime = QDateTime::currentDateTime();
                    if (ctp.visitorId <= 0)
                        ctp.visitorId = resolvePersonId(p, rawFrame, normPos, nowMs);
                    ctp.enterImage = rawFrame.empty() ? cv::Mat() : rawFrame.clone();
                    p.visitorId = ctp.visitorId;

                    // 登记待完成穿越（按 personId，便于轨迹丢失后续接）
                    if (ctp.visitorId > 0) {
                        PendingCrossing pc;
                        pc.personId = ctp.visitorId;
                        pc.regionId = ctp.regionId;
                        pc.entrySide = ctp.entrySide;
                        pc.enterTime = ctp.enterTime;
                        pc.enterImage = ctp.enterImage;
                        pc.lastTouchMs = nowMs;
                        m_pendingCrossings[pc.personId] = pc;
                        if (m_registry) m_registry->markInside(pc.personId, true);
                    }

                    LOG_INFO(QString("[ZoneMonitor] V%1 CONFIRMED ENTERED region %2 (track=%3, side=%4)")
                                 .arg(ctp.visitorId).arg(ctp.regionId).arg(p.trackId)
                                 .arg(ctp.entrySide == 0 ? "A(top)" : "B(bottom)"));
                }
            } else {
                ctp.state = ChannelTrackState::Outside;
                ctp.confirmFrames = 0;
                ctp.entrySide = -1;
                ctp.regionId = -1;
                ctp.visitorId = -1;
            }
            break;
        }
        case ChannelTrackState::Inside: {
            if (!nowInRegion) {
                ctp.state = ChannelTrackState::Exiting;
                ctp.confirmFrames = 1;

                const ChannelZone *prevZone = nullptr;
                for (const auto &z : m_zones) {
                    if (z.regionId == ctp.regionId) { prevZone = &z; break; }
                }
                if (prevZone) ctp.exitSide = determineExitSide(prevNormCy, normCy, *prevZone);
            } else if (ctp.visitorId > 0 && m_pendingCrossings.contains(ctp.visitorId)) {
                m_pendingCrossings[ctp.visitorId].lastTouchMs = nowMs;  // 保持有效
            }
            break;
        }
        case ChannelTrackState::Exiting: {
            if (!nowInRegion) {
                ctp.confirmFrames++;
                if (ctp.confirmFrames >= m_confirmFrames) {
                    const int pid = ctp.visitorId;
                    const bool hasPending = (pid > 0 && m_pendingCrossings.contains(pid));
                    const PendingCrossing pc = hasPending ? m_pendingCrossings.value(pid)
                                                          : PendingCrossing();

                    const int entrySide = hasPending ? pc.entrySide : ctp.entrySide;
                    const int regionId  = hasPending ? pc.regionId  : ctp.regionId;
                    const QDateTime enterTime = hasPending ? pc.enterTime : ctp.enterTime;
                    const cv::Mat enterImage  = hasPending ? pc.enterImage : ctp.enterImage;
                    const int exitSide = ctp.exitSide;
                    const QDateTime exitTime = QDateTime::currentDateTime();

                    const qint64 lastMs = m_lastCrossingMsByPerson.value(pid, 0);
                    const bool duplicate = (pid > 0 && (nowMs - lastMs) < CROSSING_DEDUP_MS);

                    if (duplicate) {
                        LOG_INFO(QString("[ZoneMonitor] skip duplicate crossing V%1 (%2ms since last)")
                                     .arg(pid).arg(nowMs - lastMs));
                    } else if (pid > 0 && entrySide >= 0 && exitSide >= 0) {
                        ZoneEvent evt;
                        evt.visitorId = pid;
                        evt.trackId = p.trackId;
                        evt.regionId = regionId;
                        evt.enterTime = enterTime;
                        evt.exitTime = exitTime;
                        evt.entrySide = entrySide;
                        evt.exitSide = exitSide;
                        evt.enterImage = enterImage;
                        evt.exitImage = rawFrame.empty() ? cv::Mat() : rawFrame.clone();

                        if (entrySide == 1 && exitSide == 0) {
                            evt.type = ZoneEventType::Boarding;
                            m_boardingCount++;
                            LOG_INFO(QString("[ZoneMonitor] ★ BOARDING (上车) V%1 B→A region=%2 enter=%3 exit=%4")
                                         .arg(pid).arg(regionId)
                                         .arg(enterTime.toString("HH:mm:ss.zzz"))
                                         .arg(exitTime.toString("HH:mm:ss.zzz")));
                            m_events.append(evt);
                            emit boardingEvent(evt);
                            emit zoneCrossingEvent(evt);
                        } else if (entrySide == 0 && exitSide == 1) {
                            evt.type = ZoneEventType::Alighting;
                            m_alightingCount++;
                            LOG_INFO(QString("[ZoneMonitor] ★ ALIGHTING (下车) V%1 A→B region=%2 enter=%3 exit=%4")
                                         .arg(pid).arg(regionId)
                                         .arg(enterTime.toString("HH:mm:ss.zzz"))
                                         .arg(exitTime.toString("HH:mm:ss.zzz")));
                            m_events.append(evt);
                            emit alightingEvent(evt);
                            emit zoneCrossingEvent(evt);
                        } else {
                            // 同侧进出 = 折返，仅记穿越事件
                            LOG_INFO(QString("[ZoneMonitor] same-side (turn back) V%1 side=%2→%3 region=%4")
                                         .arg(pid)
                                         .arg(entrySide == 0 ? "A" : "B")
                                         .arg(exitSide == 0 ? "A" : "B").arg(regionId));
                            m_events.append(evt);
                            emit zoneCrossingEvent(evt);
                        }
                    }

                    if (pid > 0) {
                        m_lastCrossingMsByPerson[pid] = nowMs;
                        m_pendingCrossings.remove(pid);
                        if (m_registry) m_registry->markInside(pid, false);
                    }

                    ctp.state = ChannelTrackState::Outside;
                    ctp.confirmFrames = 0;
                    ctp.entrySide = -1;
                    ctp.exitSide = -1;
                    ctp.regionId = -1;
                    ctp.enterTime = QDateTime();
                    ctp.enterImage = cv::Mat();
                    ctp.visitorId = -1;
                }
            } else {
                ctp.state = ChannelTrackState::Inside;
                ctp.confirmFrames = 0;
                ctp.exitSide = -1;
            }
            break;
        }
        }

        // 同步状态到 TrackedPerson（供绘制和UI使用）
        p.isInsideRegion = (ctp.state == ChannelTrackState::Inside);
    }
}

void ZoneMonitor::reset()
{
    m_channelTracks.clear();
    m_trackToVisitor.clear();
    m_pendingCrossings.clear();
    m_lastCrossingMsByPerson.clear();
    m_events.clear();
    m_boardingCount = 0;
    m_alightingCount = 0;
    m_nextVisitorId = 1;
    LOG_INFO("[ZoneMonitor] All state reset");
}
