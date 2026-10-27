#include "persontracker.h"
#include "reidextractor.h"
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

cv::Mat PersonTracker::computeAverageReid(const QVector<cv::Mat> &features) const
{
    if (features.isEmpty()) return cv::Mat();

    // 过滤空特征
    QVector<cv::Mat> validFeatures;
    for (const auto &f : features) {
        if (!f.empty()) validFeatures.append(f);
    }
    if (validFeatures.isEmpty()) return cv::Mat();

    // ===== 多帧时序融合（EMA）：越新的帧权重越高 =====
    // 相比等权平均，EMA 既抑制偶发噪声/误检帧，又更贴近目标“最近状态”，
    // 对俯视场景下的转身、遮挡中断更鲁棒。内部已做尺寸/通道一致性保护
    // （避免 arithm_op 尺寸不匹配导致崩溃）。
    cv::Mat fused = ReIDExtractor::temporalFuse(validFeatures, 0.4f);
    if (fused.empty()) return cv::Mat();

    // 非完整 bundle（旧维度特征）时退化为整体 L2 归一化
    if (fused.cols != ReIDExtractor::featureDim())
        cv::normalize(fused, fused, 1.0, 0.0, cv::NORM_L2);

    return fused;
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
            cv::Mat trackFeature = computeAverageReid(track.reidHistory);
            if (trackFeature.empty()) {
                trackFeature = track.reidFeature;
            }
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
                  // 优先IoU匹配（stage小的优先），同stage按分数降序
                  if (a.stage != b.stage && std::abs(a.score - b.score) > 0.1f)
                      return a.score > b.score;
                  return a.score > b.score;
              });

    for (const MatchPair &mp : allPairs) {
        if (!trackMatched[mp.trackIdx] && !detMatched[mp.detIdx]) {
            // 匹配成功：更新轨迹
            int id = trackIds[mp.trackIdx];
            TrackedPerson &track = m_tracks[id];
            track.prevCenter = track.center;
            track.bbox = detections[mp.detIdx];
            track.center = cv::Point2f(track.bbox.x + track.bbox.width / 2.0f,
                                       track.bbox.y + track.bbox.height / 2.0f);
            track.wasInsideRegion = track.isInsideRegion;
            track.age++;
            track.missedFrames = 0;
            track.detected = true;              // 本帧匹配到真实检测
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

int PersonTracker::tryReidFromGallery(const cv::Rect &detBbox,
                                       const cv::Mat &detFeature,
                                       ReIDExtractor *reidExtractor)
{
    if (m_lostGallery.isEmpty()) return -1;
    if (detFeature.empty() || !reidExtractor) return -1;

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
        int recoveredId = m_lostGallery[bestIdx].trackId;
        LOG_INFO(QString("[Tracker] Gallery ReID match: recovered track %1 (similarity=%2)")
                     .arg(recoveredId).arg(bestSimilarity, 0, 'f', 3));
        m_lostGallery.remove(bestIdx);
        return recoveredId;
    }

    return -1;
}

QVector<TrackedPerson> PersonTracker::update(const QVector<cv::Rect> &detections,
                                              const cv::Mat &frame,
                                              ReIDExtractor *reidExtractor)
{
    QMutexLocker locker(&m_mutex);   // 与主线程的 getTrack/状态同步串行化，避免 data race
    QVector<TrackedPerson> updatedTracks;

    // 每帧先假设所有轨迹都「未被真实检测」；匹配成功/新建轨迹再置回 true。
    // 这样下游（ZoneMonitor）可忽略卡尔曼预测外推的 bbox，避免误判。
    for (auto it = m_tracks.begin(); it != m_tracks.end(); ++it) {
        it.value().detected = false;
    }

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
        QList<int> trackIds = m_tracks.keys();
        for (int id : trackIds) {
            TrackedPerson &track = m_tracks[id];
            track.missedFrames++;
            // 卡尔曼继续预测更新（保持速度估计）
            if (track.kalman && track.kalman->initialized) {
                // 预测位置作为当前bbox（用于后续匹配）
                track.prevCenter = track.center;
                cv::Rect pred = track.kalman->getPredictedBbox();
                track.bbox = pred;
                track.center = cv::Point2f(pred.x + pred.width / 2.0f,
                                           pred.y + pred.height / 2.0f);
            }
            if (track.missedFrames > m_maxMissed) {
                // 移入画廊
                LostTrack lt;
                lt.trackId = track.trackId;
                lt.reidFeature = computeAverageReid(track.reidHistory);
                if (lt.reidFeature.empty()) lt.reidFeature = track.reidFeature;
                lt.lastBbox = track.bbox;
                lt.lastCenter = track.center;
                lt.visitorId = track.visitorId;
                lt.lostTime = QDateTime::currentDateTime();
                lt.age = track.age;
                m_lostGallery.append(lt);
                m_tracks.remove(id);
            }
        }
    } else if (m_tracks.isEmpty()) {
        // 无现有轨迹：为每个检测创建新轨迹（尝试从画廊恢复）
        for (const cv::Rect &det : detections) {
            TrackedPerson person;
            person.trackId = m_nextId++;
            person.bbox = det;
            person.center = cv::Point2f(det.x + det.width / 2.0f,
                                        det.y + det.height / 2.0f);
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

            // 尝试从画廊恢复
            if (m_reidMatchingEnabled && reidExtractor && !frame.empty()) {
                int recoveredId = tryReidFromGallery(det, person.reidFeature, reidExtractor);
                if (recoveredId > 0) {
                    person.trackId = recoveredId;
                    // 恢复visitorId等身份信息
                    for (const auto &lt : m_lostGallery) {
                        if (lt.trackId == recoveredId) {
                            person.visitorId = lt.visitorId;
                            break;
                        }
                    }
                }
            }

            m_tracks.insert(person.trackId, person);
        }
    } else {
        // 有现有轨迹和检测：多阶段匹配
        QList<int> trackIds = m_tracks.keys();
        QVector<bool> detMatched(detections.size(), false);
        QVector<bool> trackMatched(trackIds.size(), false);

        // 多阶段匹配
        multiStageMatch(detections, frame, reidExtractor,
                        detMatched, trackMatched, trackIds);

        // 未匹配的检测：创建新轨迹（尝试从画廊恢复）
        for (int di = 0; di < detections.size(); di++) {
            if (!detMatched[di]) {
                TrackedPerson person;
                person.trackId = m_nextId++;
                person.bbox = detections[di];
                person.center = cv::Point2f(detections[di].x + detections[di].width / 2.0f,
                                            detections[di].y + detections[di].height / 2.0f);
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
                    person.kalman->init(detections[di]);
                }

                // 提取ReID特征
                if (m_reidMatchingEnabled && reidExtractor && !frame.empty()) {
                    person.reidFeature = reidExtractor->extract(frame, detections[di]);
                    updateReidHistory(person, person.reidFeature);
                }

                // 尝试从画廊恢复身份
                if (m_reidMatchingEnabled && !person.reidFeature.empty()) {
                    int recoveredId = tryReidFromGallery(detections[di], person.reidFeature, reidExtractor);
                    if (recoveredId > 0) {
                        person.trackId = recoveredId;
                    }
                }

                m_tracks.insert(person.trackId, person);
            }
        }

        // 未匹配的轨迹：增加丢失计数
        for (int ti = 0; ti < trackIds.size(); ti++) {
            if (!trackMatched[ti]) {
                int id = trackIds[ti];
                TrackedPerson &track = m_tracks[id];
                track.missedFrames++;

                // 使用卡尔曼预测位置（在丢失期间保持平滑移动）
                if (track.kalman && track.kalman->initialized) {
                    track.prevCenter = track.center;
                    cv::Rect pred = track.kalman->getPredictedBbox();
                    track.bbox = pred;
                    track.center = cv::Point2f(pred.x + pred.width / 2.0f,
                                               pred.y + pred.height / 2.0f);
                }

                if (track.missedFrames > m_maxMissed) {
                    // 移入画廊
                    LostTrack lt;
                    lt.trackId = track.trackId;
                    lt.reidFeature = computeAverageReid(track.reidHistory);
                    if (lt.reidFeature.empty()) lt.reidFeature = track.reidFeature;
                    lt.lastBbox = track.bbox;
                    lt.lastCenter = track.center;
                    lt.visitorId = track.visitorId;
                    lt.lostTime = QDateTime::currentDateTime();
                    lt.age = track.age;
                    m_lostGallery.append(lt);
                    m_tracks.remove(id);
                }
            }
        }
    }

    // 收集所有活跃轨迹
    for (auto it = m_tracks.begin(); it != m_tracks.end(); ++it) {
        updatedTracks.push_back(it.value());
    }

    return updatedTracks;
}

QVector<TrackedPerson> PersonTracker::getActiveTracks() const
{
    QVector<TrackedPerson> active;
    for (auto it = m_tracks.begin(); it != m_tracks.end(); ++it) {
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
