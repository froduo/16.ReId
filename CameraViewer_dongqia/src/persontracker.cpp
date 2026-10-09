#include "persontracker.h"
#include "reidextractor.h"
#include "visitorregistry.h"
#include "logger.h"
#include <algorithm>
#include <cmath>

// ===== KalmanTrack 实现 =====

KalmanTrack::KalmanTrack()
{
}

void KalmanTrack::init(const cv::Rect &bbox)
{
    // 状态: [cx, cy, w, h, vcx, vcy, vw, vh]
    kf = cv::KalmanFilter(8, 4, 0);

    // 状态转移矩阵 F
    // cx' = cx + vcx*dt, cy' = cy + vcy*dt, w' = w + vw*dt, h' = h + vh*dt
    kf.transitionMatrix = (cv::Mat_<float>(8, 8) <<
        1, 0, 0, 0, 1, 0, 0, 0,
        0, 1, 0, 0, 0, 1, 0, 0,
        0, 0, 1, 0, 0, 0, 1, 0,
        0, 0, 0, 1, 0, 0, 0, 1,
        0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 1);

    // 观测矩阵 H
    kf.measurementMatrix = cv::Mat::zeros(4, 8, CV_32F);
    kf.measurementMatrix.at<float>(0, 0) = 1;
    kf.measurementMatrix.at<float>(1, 1) = 1;
    kf.measurementMatrix.at<float>(2, 2) = 1;
    kf.measurementMatrix.at<float>(3, 3) = 1;

    // 过程噪声协方差 Q
    cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-2));
    // 速度分量的过程噪声更大（允许加速度变化）
    kf.processNoiseCov.at<float>(4, 4) = 1e-1;
    kf.processNoiseCov.at<float>(5, 5) = 1e-1;
    kf.processNoiseCov.at<float>(6, 6) = 1e-1;
    kf.processNoiseCov.at<float>(7, 7) = 1e-1;

    // 观测噪声协方差 R
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));

    // 初始后验协方差 P
    cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1));

    // 初始化状态
    float cx = bbox.x + bbox.width / 2.0f;
    float cy = bbox.y + bbox.height / 2.0f;
    kf.statePost = (cv::Mat_<float>(8, 1) <<
        cx, cy, static_cast<float>(bbox.width), static_cast<float>(bbox.height),
        0, 0, 0, 0);

    initialized = true;
}

void KalmanTrack::predict()
{
    if (!initialized) return;
    kf.predict();
}

void KalmanTrack::update(const cv::Rect &bbox)
{
    if (!initialized) {
        init(bbox);
        return;
    }

    float cx = bbox.x + bbox.width / 2.0f;
    float cy = bbox.y + bbox.height / 2.0f;
    cv::Mat measurement = (cv::Mat_<float>(4, 1) <<
        cx, cy, static_cast<float>(bbox.width), static_cast<float>(bbox.height));

    kf.correct(measurement);
}

cv::Rect KalmanTrack::getPredictedBbox() const
{
    if (!initialized) return cv::Rect();

    cv::Mat state = kf.statePost;
    float cx = state.at<float>(0);
    float cy = state.at<float>(1);
    float w = state.at<float>(2);
    float h = state.at<float>(3);

    // 加上速度预测
    float vcx = state.at<float>(4);
    float vcy = state.at<float>(5);
    float vw = state.at<float>(6);
    float vh = state.at<float>(7);

    float predCx = cx + vcx;
    float predCy = cy + vcy;
    float predW = std::max(10.0f, w + vw);
    float predH = std::max(10.0f, h + vh);

    int x = static_cast<int>(predCx - predW / 2.0f);
    int y = static_cast<int>(predCy - predH / 2.0f);

    return cv::Rect(x, y, static_cast<int>(predW), static_cast<int>(predH));
}

cv::Point2f KalmanTrack::predictedCenter() const
{
    if (!initialized) return cv::Point2f(0.0f, 0.0f);
    const cv::Rect p = getPredictedBbox();
    return cv::Point2f(p.x + p.width * 0.5f, p.y + p.height * 0.5f);
}

cv::Point2f KalmanTrack::velocity() const
{
    if (!initialized) return cv::Point2f(0.0f, 0.0f);
    return cv::Point2f(kf.statePost.at<float>(4), kf.statePost.at<float>(5));
}


// ===== PersonTracker 实现 =====

PersonTracker::PersonTracker(float iouThreshold, int maxMissed)
    : m_nextId(1)
    , m_iouThreshold(iouThreshold)
    , m_maxMissed(maxMissed)
{
}

float PersonTracker::computeIoU(const cv::Rect &a, const cv::Rect &b)
{
    cv::Rect intersection = a & b;
    if (intersection.empty())
        return 0.0f;

    float interArea = static_cast<float>(intersection.area());
    float unionArea = static_cast<float>(a.area() + b.area()) - interArea;

    return (unionArea > 0) ? (interArea / unionArea) : 0.0f;
}

cv::Point2f PersonTracker::centerOf(const cv::Rect &r)
{
    return cv::Point2f(r.x + r.width * 0.5f, r.y + r.height * 0.5f);
}

float PersonTracker::centerDistance(const cv::Rect &a, const cv::Rect &b)
{
    const cv::Point2f ca = centerOf(a);
    const cv::Point2f cb = centerOf(b);
    const float dx = ca.x - cb.x;
    const float dy = ca.y - cb.y;
    return std::sqrt(dx * dx + dy * dy);
}

cv::Point2f PersonTracker::normalizedPos(const cv::Rect &bbox, const cv::Mat &frame)
{
    if (frame.empty() || frame.cols <= 0 || frame.rows <= 0)
        return cv::Point2f(0.5f, 0.5f);
    const cv::Point2f c = centerOf(bbox);
    return cv::Point2f(c.x / frame.cols, c.y / frame.rows);
}

cv::Mat PersonTracker::computeAverageReid(const QVector<cv::Mat> &features) const
{
    if (features.isEmpty()) return cv::Mat();

    // 过滤空特征
    QVector<cv::Mat> validFeatures;
    for (const auto &f : features) {
        if (!f.empty()) validFeatures.append(f);
    }
    if (validFeatures.isEmpty()) return cv::Mat();

    // 计算逐元素平均
    cv::Mat sum = validFeatures[0].clone();
    sum.convertTo(sum, CV_64F);
    for (int i = 1; i < validFeatures.size(); ++i) {
        cv::Mat tmp;
        validFeatures[i].convertTo(tmp, CV_64F);
        sum += tmp;
    }
    sum /= static_cast<double>(validFeatures.size());
    sum.convertTo(sum, CV_32F);

    // L2 归一化
    cv::normalize(sum, sum, 1.0, 0.0, cv::NORM_L2);

    return sum;
}

cv::Mat PersonTracker::robustFeature(const TrackedPerson &track) const
{
    cv::Mat f = computeAverageReid(track.reidHistory);
    if (f.empty()) f = track.reidFeature;
    return f;
}

void PersonTracker::updateReidHistory(TrackedPerson &track, const cv::Mat &newFeature)
{
    if (newFeature.empty()) return;

    track.reidHistory.append(newFeature.clone());
    if (track.reidHistory.size() > TrackedPerson::REID_HISTORY_SIZE) {
        track.reidHistory.removeFirst();
    }
}

void PersonTracker::multiStageMatch(const QVector<cv::Rect> &detections,
                                     const cv::Mat &frame,
                                     ReIDExtractor *reidExtractor,
                                     QVector<bool> &detMatched,
                                     QVector<bool> &trackMatched,
                                     const QList<int> &trackIds)
{
    int numTracks = trackIds.size();
    int numDets = detections.size();

    // ===== 阶段1: IoU 匹配 =====
    struct MatchPair {
        int trackIdx;
        int detIdx;
        float score;  // 越高越好
        int stage;    // 匹配阶段
    };
    QVector<MatchPair> allPairs;

    for (int ti = 0; ti < numTracks; ti++) {
        if (trackMatched[ti]) continue;
        int id = trackIds[ti];
        const TrackedPerson &track = m_tracks[id];

        for (int di = 0; di < numDets; di++) {
            if (detMatched[di]) continue;

            float iou = computeIoU(track.bbox, detections[di]);
            if (iou >= m_iouThreshold) {
                allPairs.push_back({ti, di, iou, 1});
            }
        }
    }

    // ===== 阶段2: 卡尔曼预测 IoU 匹配 =====
    if (m_kalmanEnabled) {
        for (int ti = 0; ti < numTracks; ti++) {
            if (trackMatched[ti]) continue;
            int id = trackIds[ti];
            const TrackedPerson &track = m_tracks[id];

            if (track.kalman && track.kalman->initialized) {
                cv::Rect predictedBbox = track.kalman->getPredictedBbox();
                for (int di = 0; di < numDets; di++) {
                    if (detMatched[di]) continue;

                    float predIou = computeIoU(predictedBbox, detections[di]);
                    // 卡尔曼预测IoU阈值略低（因为是预测值）
                    float kalmanThreshold = m_iouThreshold * 0.5f;
                    if (predIou >= kalmanThreshold) {
                        // 检查是否已经在阶段1中有此配对
                        bool alreadyMatched = false;
                        for (const auto &mp : allPairs) {
                            if (mp.trackIdx == ti && mp.detIdx == di && mp.stage == 1) {
                                alreadyMatched = true;
                                break;
                            }
                        }
                        if (!alreadyMatched) {
                            // 使用预测IoU作为分数，但权重略低
                            allPairs.push_back({ti, di, predIou * 0.8f, 2});
                        }
                    }
                }
            }
        }
    }

    // ===== 阶段3: ReID 特征匹配 =====
    if (m_reidMatchingEnabled && reidExtractor && !frame.empty()) {
        // 只对尚未匹配的检测提取ReID特征
        QVector<cv::Mat> detFeatures(numDets);
        for (int di = 0; di < numDets; di++) {
            if (detMatched[di]) continue;
            detFeatures[di] = reidExtractor->extract(frame, detections[di]);
        }

        for (int ti = 0; ti < numTracks; ti++) {
            if (trackMatched[ti]) continue;
            int id = trackIds[ti];
            const TrackedPerson &track = m_tracks[id];

            // 使用平均ReID特征（更鲁棒）
            cv::Mat trackFeature = robustFeature(track);
            if (trackFeature.empty()) continue;

            for (int di = 0; di < numDets; di++) {
                if (detMatched[di]) continue;
                if (detFeatures[di].empty()) continue;

                float similarity = reidExtractor->compare(trackFeature, detFeatures[di]);
                if (similarity >= m_reidMatchThreshold) {
                    // 检查是否已有此配对
                    bool alreadyMatched = false;
                    for (const auto &mp : allPairs) {
                        if (mp.trackIdx == ti && mp.detIdx == di) {
                            alreadyMatched = true;
                            break;
                        }
                    }
                    if (!alreadyMatched) {
                        // ReID分数权重：相似度 * 0.6（低于IoU权重，避免误匹配）
                        allPairs.push_back({ti, di, similarity * 0.6f, 3});
                    }
                }
            }
        }
    }

    // ===== 贪心匹配（按分数降序）=====
    std::sort(allPairs.begin(), allPairs.end(),
              [](const MatchPair &a, const MatchPair &b) {
                  return a.score > b.score;
              });

    for (const MatchPair &mp : allPairs) {
        if (!trackMatched[mp.trackIdx] && !detMatched[mp.detIdx]) {
            // 匹配成功：更新轨迹
            int id = trackIds[mp.trackIdx];
            TrackedPerson &track = m_tracks[id];
            track.prevCenter = track.center;
            track.bbox = detections[mp.detIdx];
            track.center = centerOf(track.bbox);
            track.wasInsideRegion = track.isInsideRegion;
            track.age++;
            track.missedFrames = 0;
            track.lastSeen = QDateTime::currentDateTime();

            // 更新卡尔曼滤波器
            if (track.kalman) {
                track.kalman->update(track.bbox);
            }

            // 更新ReID特征
            if (m_reidMatchingEnabled && reidExtractor && !frame.empty()) {
                cv::Mat newFeature = reidExtractor->extract(frame, track.bbox);
                if (!newFeature.empty()) {
                    track.reidFeature = newFeature.clone();
                    updateReidHistory(track, newFeature);
                }
            }

            trackMatched[mp.trackIdx] = true;
            detMatched[mp.detIdx] = true;

            if (mp.stage > 1) {
                LOG_INFO(QString("[Tracker] Track %1 matched via stage %2 (score=%3)")
                             .arg(id).arg(mp.stage).arg(mp.score, 0, 'f', 3));
            }
        }
    }
}

void PersonTracker::updateLostGallery()
{
    QDateTime now = QDateTime::currentDateTime();

    // 将超过保留时间的轨迹从画廊中移除
    auto it = m_lostGallery.begin();
    while (it != m_lostGallery.end()) {
        if (it->lostTime.msecsTo(now) > m_galleryMaxAgeMs) {
            it = m_lostGallery.erase(it);
        } else {
            ++it;
        }
    }
}

bool PersonTracker::tryReidFromGallery(const cv::Rect &detBbox,
                                        const cv::Mat &detFeature,
                                        ReIDExtractor *reidExtractor,
                                        int *trackIdOut, int *visitorIdOut, float *simOut)
{
    if (m_lostGallery.isEmpty()) return false;
    if (detFeature.empty() || !reidExtractor) return false;

    int bestIdx = -1;
    float bestSimilarity = 0.0f;

    for (int i = 0; i < m_lostGallery.size(); ++i) {
        const LostTrack &lt = m_lostGallery[i];
        if (lt.reidFeature.empty()) continue;

        float similarity = reidExtractor->compare(detFeature, lt.reidFeature);

        // 额外检查：位置距离合理性（防止远处的人被错误匹配）
        float cx1 = detBbox.x + detBbox.width / 2.0f;
        float cy1 = detBbox.y + detBbox.height / 2.0f;
        float cx2 = lt.lastCenter.x;
        float cy2 = lt.lastCenter.y;
        float dist = std::sqrt((cx1 - cx2) * (cx1 - cx2) + (cy1 - cy2) * (cy1 - cy2));

        // 最大允许距离（基于bbox尺寸）
        float maxDist = std::max(detBbox.width, detBbox.height) * 10.0f;

        if (similarity > bestSimilarity && similarity >= m_reidMatchThreshold && dist < maxDist) {
            bestSimilarity = similarity;
            bestIdx = i;
        }
    }

    if (bestIdx >= 0) {
        const LostTrack lt = m_lostGallery[bestIdx];
        if (trackIdOut) *trackIdOut = lt.trackId;
        if (visitorIdOut) *visitorIdOut = lt.visitorId;
        if (simOut) *simOut = bestSimilarity;

        LOG_INFO(QString("[Tracker] Gallery ReID match: recovered track %1 (V%2, similarity=%3)")
                     .arg(lt.trackId).arg(lt.visitorId).arg(bestSimilarity, 0, 'f', 3));
        m_lostGallery.remove(bestIdx);
        return true;
    }

    return false;
}

// ===== 身份管理 =====

int PersonTracker::resolveIdentity(const cv::Mat &feature, const cv::Point2f &normPos,
                                   float *simOut, bool *createdOut)
{
    if (simOut) *simOut = 0.0f;
    if (createdOut) *createdOut = false;
    if (!m_registry) return -1;

    const VisitorRegistry::MatchInfo mi =
        m_registry->resolveEx(feature, QDateTime::currentMSecsSinceEpoch(), normPos);
    if (simOut) *simOut = mi.similarity;
    if (createdOut) *createdOut = mi.created;
    return mi.personId;
}

void PersonTracker::bindIdentity(int trackId, int personId)
{
    if (trackId <= 0 || personId <= 0) return;

    const int prevTrack = m_personToTrack.value(personId, -1);
    if (prevTrack > 0 && prevTrack != trackId) {
        // 身份被其它轨迹占用：解除旧绑定（交由去重环节合并）
        m_trackToPerson.remove(prevTrack);
    }
    m_personToTrack[personId] = trackId;
    m_trackToPerson[trackId] = personId;
}

void PersonTracker::unbindIdentity(int trackId)
{
    const int pid = m_trackToPerson.value(trackId, -1);
    if (pid > 0 && m_personToTrack.value(pid, -1) == trackId)
        m_personToTrack.remove(pid);
    m_trackToPerson.remove(trackId);
}

TrackedPerson PersonTracker::createTrack(const cv::Rect &det, const cv::Mat &frame,
                                         ReIDExtractor *reidExtractor)
{
    TrackedPerson person;
    person.trackId = m_nextId++;
    person.bbox = det;
    person.center = centerOf(det);
    person.prevCenter = person.center;
    person.age = 1;
    person.missedFrames = 0;
    person.isInsideRegion = false;
    person.wasInsideRegion = false;
    person.firstSeen = QDateTime::currentDateTime();
    person.lastSeen = person.firstSeen;
    person.hasEntered = false;
    person.hasExited = false;
    person.visitorId = -1;

    // 初始化卡尔曼滤波器
    if (m_kalmanEnabled) {
        person.kalman = std::make_shared<KalmanTrack>();
        person.kalman->init(det);
    }

    // 提取ReID特征
    if (m_reidMatchingEnabled && reidExtractor && !frame.empty()) {
        person.reidFeature = reidExtractor->extract(frame, det);
        updateReidHistory(person, person.reidFeature);
    }

    return person;
}

void PersonTracker::absorbDetection(int trackId, const cv::Rect &det, const cv::Mat &feature)
{
    if (!m_tracks.contains(trackId)) return;

    TrackedPerson &t = m_tracks[trackId];
    t.prevCenter = t.center;
    t.bbox = det;
    t.center = centerOf(det);
    t.age++;
    t.missedFrames = 0;
    t.wasInsideRegion = t.isInsideRegion;
    t.lastSeen = QDateTime::currentDateTime();

    if (t.kalman) t.kalman->update(t.bbox);

    if (!feature.empty()) {
        if (t.reidFeature.empty()) t.reidFeature = feature.clone();
        updateReidHistory(t, feature);
    }
}

void PersonTracker::absorbTrack(int keepTrackId, int dropTrackId)
{
    if (keepTrackId == dropTrackId) return;
    if (!m_tracks.contains(keepTrackId) || !m_tracks.contains(dropTrackId)) return;

    TrackedPerson keep = m_tracks.value(keepTrackId);
    const TrackedPerson drop = m_tracks.value(dropTrackId);

    // 外观历史合并（保留更多观测，提升后续匹配质量）
    for (const cv::Mat &f : drop.reidHistory) {
        if (!f.empty()) updateReidHistory(keep, f);
    }
    if (keep.reidFeature.empty() && !drop.reidFeature.empty())
        keep.reidFeature = drop.reidFeature.clone();

    keep.age = std::max(keep.age, drop.age);
    keep.missedFrames = 0;
    keep.firstSeen = (keep.firstSeen.isValid() && drop.firstSeen.isValid())
                         ? std::min(keep.firstSeen, drop.firstSeen)
                         : (keep.firstSeen.isValid() ? keep.firstSeen : drop.firstSeen);

    // 身份统一：优先保留已有稳定身份
    int pid = (keep.visitorId > 0) ? keep.visitorId : drop.visitorId;
    keep.visitorId = pid;

    // 穿越状态：保留信息更完整的一条
    if (drop.hasEntered && !keep.hasEntered) {
        keep.hasEntered = true;
        keep.enterTime = drop.enterTime;
        if (keep.enterImage.empty() && !drop.enterImage.empty())
            keep.enterImage = drop.enterImage.clone();
    }
    if (drop.isInsideRegion) keep.isInsideRegion = true;
    if (keep.crossState == RegionCrossState::Outside &&
        drop.crossState != RegionCrossState::Outside) {
        keep.crossState = drop.crossState;
        keep.crossConfirmFrames = drop.crossConfirmFrames;
    }
    keep.mergedTracks += 1 + drop.mergedTracks;

    m_tracks[keepTrackId] = keep;

    // 绑定关系收敛到 keep
    unbindIdentity(dropTrackId);
    if (pid > 0) bindIdentity(keepTrackId, pid);

    m_tracks.remove(dropTrackId);

    // 画廊中若残留同一身份的条目，一并清理，避免重复恢复
    if (pid > 0) {
        for (int i = m_lostGallery.size() - 1; i >= 0; --i) {
            if (m_lostGallery[i].visitorId == pid || m_lostGallery[i].trackId == dropTrackId)
                m_lostGallery.remove(i);
        }
    }

    // 若被合并的轨迹在注册表中另有身份，合并身份画廊
    if (m_registry && keep.visitorId > 0 && drop.visitorId > 0
        && keep.visitorId != drop.visitorId) {
        m_registry->absorb(keep.visitorId, drop.visitorId);
    }

    m_mergeCount++;
}

void PersonTracker::dedupTracks(ReIDExtractor *reidExtractor)
{
    if (m_tracks.size() < 2) return;

    // ===== 第一遍：稳定身份完全相同的轨迹必然是同一个人 → 直接合并 =====
    QMap<int, QList<int>> byPerson;
    const QList<int> allIds = m_tracks.keys();
    for (int id : allIds) {
        const int pid = m_tracks.value(id).visitorId;
        if (pid > 0) byPerson[pid].append(id);
    }
    for (auto it = byPerson.constBegin(); it != byPerson.constEnd(); ++it) {
        const QList<int> group = it.value();
        if (group.size() < 2) continue;

        int keep = group.first();
        for (int id : group) {
            if (!m_tracks.contains(id) || !m_tracks.contains(keep)) continue;
            if (m_tracks.value(id).age > m_tracks.value(keep).age) keep = id;
        }
        if (!m_tracks.contains(keep)) continue;

        for (int id : group) {
            if (id == keep || !m_tracks.contains(id)) continue;
            absorbTrack(keep, id);
            LOG_INFO(QString("[Tracker] DEDUP(identity): track %1 merged into %2 (V%3)")
                         .arg(id).arg(keep).arg(it.key()));
        }
    }

    if (m_tracks.size() < 2 || !reidExtractor) return;

    // ===== 第二遍：外观 + 卡尔曼预测位置（时空门限）=====
    const QList<int> ids = m_tracks.keys();
    for (int i = 0; i < ids.size(); ++i) {
        const int a = ids[i];
        if (!m_tracks.contains(a)) continue;

        for (int j = i + 1; j < ids.size(); ++j) {
            const int b = ids[j];
            if (!m_tracks.contains(b)) continue;

            const TrackedPerson ta = m_tracks.value(a);
            const TrackedPerson tb = m_tracks.value(b);

            // 空间门限：优先使用卡尔曼预测框（跟随运动方向）
            cv::Rect pa = ta.bbox;
            cv::Rect pb = tb.bbox;
            if (m_kalmanEnabled) {
                if (ta.kalman && ta.kalman->initialized) pa = ta.kalman->getPredictedBbox();
                if (tb.kalman && tb.kalman->initialized) pb = tb.kalman->getPredictedBbox();
            }
            const float maxDim = static_cast<float>(
                std::max(std::max(pa.width, pa.height), std::max(pb.width, pb.height)));
            if (maxDim <= 0.0f) continue;

            // ===== 空间证据（两路）=====
            // (1) IoU：检测分裂/同一人被检出两个高度重叠的框 → 极强证据
            // (2) 卡尔曼预测中心距：快速移动或短暂跟丢后重建 → 中等证据
            const float iou = computeIoU(pa, pb);
            const float dist = centerDistance(pa, pb);
            const bool overlapStrong = (iou >= 0.45f);
            const bool nearStrong = (dist <= maxDim * 0.9f);
            if (!overlapStrong && !nearStrong) continue;

            // ===== 外观证据 =====
            const cv::Mat fa = robustFeature(ta);
            const cv::Mat fb = robustFeature(tb);
            if (fa.empty() || fb.empty()) continue;

            const float sim = reidExtractor->compare(fa, fb);
            // 重叠很强时（同一人的重复框）放宽外观要求；仅「位置邻近」时用严格阈值
            const float required = overlapStrong ? (m_mergeSimilarity - 0.12f)
                                                 : m_mergeSimilarity;
            if (sim < required) continue;

            // 年龄保护：两条轨迹都已成熟时，必须外观几乎一致才允许合并
            if (std::min(ta.age, tb.age) > m_mergeMaxYoungAge && sim < 0.94f) continue;

            // 保留更稳定的那条（存活更久；相同则保留 trackId 较小的）
            int keep = a;
            int drop = b;
            if (tb.age > ta.age || (tb.age == ta.age && b < a)) {
                keep = b;
                drop = a;
            }

            absorbTrack(keep, drop);
            LOG_INFO(QString("[Tracker] DEDUP: track %1 merged into %2 "
                             "(sim=%3, iou=%4, dist=%5)")
                         .arg(drop).arg(keep).arg(sim, 0, 'f', 3)
                         .arg(iou, 0, 'f', 2).arg(dist, 0, 'f', 1));

            if (drop == a) break;   // a 已被并入 b，跳出内层
        }
    }
}

void PersonTracker::retireTrack(int trackId)
{
    if (!m_tracks.contains(trackId)) return;

    const TrackedPerson t = m_tracks.value(trackId);

    LostTrack lt;
    lt.trackId = t.trackId;
    lt.reidFeature = robustFeature(t);
    lt.lastBbox = t.bbox;
    lt.lastCenter = t.center;
    lt.visitorId = t.visitorId;
    lt.lostTime = QDateTime::currentDateTime();
    lt.age = t.age;

    // 同一身份只保留最新一条画廊记录，避免重复恢复产生新 ID
    if (lt.visitorId > 0) {
        for (int i = m_lostGallery.size() - 1; i >= 0; --i) {
            if (m_lostGallery[i].visitorId == lt.visitorId)
                m_lostGallery.remove(i);
        }
    }
    m_lostGallery.append(lt);

    unbindIdentity(trackId);
    m_tracks.remove(trackId);
}

QVector<TrackedPerson> PersonTracker::update(const QVector<cv::Rect> &detections,
                                              const cv::Mat &frame,
                                              ReIDExtractor *reidExtractor)
{
    QVector<TrackedPerson> updatedTracks;
    const qint64 nowMs = QDateTime::currentMSecsSinceEpoch();

    // 更新丢失画廊
    updateLostGallery();

    // 对所有现有轨迹执行卡尔曼预测
    if (m_kalmanEnabled) {
        for (auto it = m_tracks.begin(); it != m_tracks.end(); ++it) {
            if (it.value().kalman) {
                it.value().kalman->predict();
            }
        }
    }

    if (detections.isEmpty()) {
        // 无检测：所有轨迹增加丢失计数
        const QList<int> trackIds = m_tracks.keys();
        for (int id : trackIds) {
            if (!m_tracks.contains(id)) continue;
            TrackedPerson &track = m_tracks[id];
            track.missedFrames++;

            // 卡尔曼继续预测更新（保持速度估计）
            if (track.kalman && track.kalman->initialized) {
                track.prevCenter = track.center;
                const cv::Rect pred = track.kalman->getPredictedBbox();
                track.bbox = pred;
                track.center = centerOf(pred);
            }

            if (track.missedFrames > m_maxMissed) retireTrack(id);
        }
    } else {
        QList<int> trackIds = m_tracks.keys();
        QVector<bool> detMatched(detections.size(), false);
        QVector<bool> trackMatched(trackIds.size(), false);

        // 多阶段匹配（IoU → 卡尔曼预测 IoU → ReID）
        multiStageMatch(detections, frame, reidExtractor,
                        detMatched, trackMatched, trackIds);

        // ===== 未匹配检测：先做身份去重，再决定是否新建轨迹 =====
        for (int di = 0; di < detections.size(); ++di) {
            if (detMatched[di]) continue;

            cv::Mat detFeature;
            if (m_reidMatchingEnabled && reidExtractor && !frame.empty())
                detFeature = reidExtractor->extract(frame, detections[di]);

            const cv::Point2f normPos = normalizedPos(detections[di], frame);

            // (a) 画廊重识别：短暂跟丢后恢复原轨迹（身份连续）
            int galleryTrackId = -1;
            int galleryPersonId = -1;
            float gallerySim = 0.0f;
            tryReidFromGallery(detections[di], detFeature, reidExtractor,
                               &galleryTrackId, &galleryPersonId, &gallerySim);

            // (b) 注册表身份解析：命中既有身份则复用，绝不轻易新编号
            float idSim = 0.0f;
            bool idCreated = false;
            int personId = resolveIdentity(detFeature, normPos, &idSim, &idCreated);
            if (galleryPersonId > 0 && (personId <= 0 || idCreated)) {
                personId = galleryPersonId;
                idCreated = false;
                idSim = std::max(idSim, gallerySim);
            }

            // (c) 该身份已被某活跃轨迹占用 → 同一人的重复检测，直接并入
            if (personId > 0) {
                const int existing = m_personToTrack.value(personId, -1);
                if (existing > 0 && m_tracks.contains(existing) && existing != galleryTrackId) {
                    absorbDetection(existing, detections[di], detFeature);
                    detMatched[di] = true;
                    LOG_INFO(QString("[Tracker] DEDUP: det merged into track %1 "
                                     "(V%2, sim=%3)")
                                 .arg(existing).arg(personId).arg(idSim, 0, 'f', 3));
                    continue;
                }
            }

            // (d) 新建轨迹
            TrackedPerson person = createTrack(detections[di], frame, reidExtractor);
            if (galleryTrackId > 0) {
                person.trackId = galleryTrackId;
                if (galleryPersonId > 0) person.visitorId = galleryPersonId;
            }
            if (personId > 0) person.visitorId = personId;

            const int newTrackId = person.trackId;
            m_tracks.insert(newTrackId, person);
            if (person.visitorId > 0) bindIdentity(newTrackId, person.visitorId);
            detMatched[di] = true;

            LOG_INFO(QString("[Tracker] new track %1 (V%2, created=%3, sim=%4)")
                         .arg(newTrackId).arg(person.visitorId)
                         .arg(idCreated ? 1 : 0).arg(idSim, 0, 'f', 3));
        }

        // ===== 未匹配轨迹：丢失计数 / 移入画廊 =====
        for (int ti = 0; ti < trackIds.size(); ++ti) {
            if (trackMatched[ti]) continue;
            const int id = trackIds[ti];
            if (!m_tracks.contains(id)) continue;
            TrackedPerson &track = m_tracks[id];
            track.missedFrames++;

            // 使用卡尔曼预测位置（在丢失期间保持平滑移动）
            if (track.kalman && track.kalman->initialized) {
                track.prevCenter = track.center;
                const cv::Rect pred = track.kalman->getPredictedBbox();
                track.bbox = pred;
                track.center = centerOf(pred);
            }

            if (track.missedFrames > m_maxMissed) retireTrack(id);
        }

        // ===== 稳定身份维护 + 外观画廊强化 =====
        const QList<int> activeIds = m_tracks.keys();
        for (int id : activeIds) {
            if (!m_tracks.contains(id)) continue;
            TrackedPerson &t = m_tracks[id];

            const cv::Mat feat = robustFeature(t);
            if (feat.empty()) continue;

            const cv::Point2f np = normalizedPos(t.bbox, frame);
            if (t.visitorId <= 0) {
                float s = 0.0f;
                const int pid = resolveIdentity(feat, np, &s);
                if (pid > 0) {
                    t.visitorId = pid;
                    bindIdentity(id, pid);
                    LOG_INFO(QString("[Tracker] track %1 bound to V%2 (sim=%3)")
                                 .arg(id).arg(pid).arg(s, 0, 'f', 3));
                }
            } else if (m_registry && (t.age % 10 == 0)) {
                // 节流：每 10 帧强化一次画廊即可，避免每帧加锁与自相关累积
                m_registry->reinforce(t.visitorId, feat, nowMs, np);
            }
        }

        // ===== 去重：把同一人的重复轨迹合并为一条 =====
        dedupTracks(reidExtractor);
    }

    // 收集所有活跃轨迹
    for (auto it = m_tracks.constBegin(); it != m_tracks.constEnd(); ++it) {
        updatedTracks.push_back(it.value());
    }

    return updatedTracks;
}

QVector<TrackedPerson> PersonTracker::getActiveTracks() const
{
    QVector<TrackedPerson> active;
    for (auto it = m_tracks.constBegin(); it != m_tracks.constEnd(); ++it) {
        active.push_back(it.value());
    }
    return active;
}

TrackedPerson *PersonTracker::getTrack(int trackId)
{
    if (m_tracks.contains(trackId))
        return &m_tracks[trackId];
    return nullptr;
}
