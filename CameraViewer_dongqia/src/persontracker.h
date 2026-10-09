#ifndef PERSONTRACKER_H
#define PERSONTRACKER_H

#include <QVector>
#include <QMap>
#include <QSet>
#include <QDateTime>
#include <QElapsedTimer>
#include <opencv2/opencv.hpp>
#include <memory>

// 前向声明
class ReIDExtractor;
class VisitorRegistry;

// 区域穿越状态机
enum class RegionCrossState {
    Outside,    // 在区域外
    Entering,   // 正在进入（已穿过边界，等待确认）
    Inside,     // 已确认在区域内
    Exiting     // 正在离开（已穿出边界，等待确认）
};

/**
 * @brief 卡尔曼滤波器封装
 *
 * 状态向量: [cx, cy, w, h, vcx, vcy, vw, vh]
 * 观测向量: [cx, cy, w, h]
 */
struct KalmanTrack {
    cv::KalmanFilter kf;
    bool initialized{false};

    KalmanTrack();
    void init(const cv::Rect &bbox);
    void predict();
    void update(const cv::Rect &bbox);
    cv::Rect getPredictedBbox() const;
    /// 预测中心点（用于时空门限判定）
    cv::Point2f predictedCenter() const;
    /// 当前速度估计 (vcx, vcy)
    cv::Point2f velocity() const;
};

struct TrackedPerson {
    int trackId;             // 轨迹 ID（易变：跟丢后重建会变）
    /**
     * @brief 稳定身份 ID（= VisitorRegistry::personId），跨轨迹唯一
     *
     * 同一个人无论中途被分配了多少个 trackId，visitorId 始终不变，
     * 因此显示 ID、上下车 ID、进出记录 ID 全部以 visitorId 为准。
     */
    int visitorId;
    cv::Rect bbox;
    cv::Point2f center;
    cv::Point2f prevCenter;  // 前一帧中心点（用于方向检测）
    int age;             // 存活帧数
    int missedFrames;    // 丢失帧数
    bool isInsideRegion; // 当前帧是否在检测区域内（点在多边形内）
    bool wasInsideRegion; // 前一帧是否在检测区域内
    cv::Mat reidFeature; // ReID 特征向量
    QDateTime firstSeen;
    QDateTime lastSeen;
    QDateTime enterTime; // 进入区域时间
    QDateTime exitTime;  // 离开区域时间
    bool hasEntered;     // 是否已确认进入过区域
    bool hasExited;      // 是否已确认离开过区域
    cv::Mat enterImage;  // 进入时的图像
    cv::Mat exitImage;   // 离开时的图像
    int mergedTracks{0}; // 被并入本轨迹的重复轨迹数量（去重统计）

    // ===== 状态机：防抖动进入/离开检测 =====
    RegionCrossState crossState{RegionCrossState::Outside};
    int crossConfirmFrames{0};  // 穿越确认帧计数
    int crossDirectionSign{0};  // 穿越方向（+1=正向, -1=反向）

    // ===== 卡尔曼滤波 =====
    std::shared_ptr<KalmanTrack> kalman;  // 卡尔曼滤波器

    // ===== ReID 特征历史（多帧平均以提高鲁棒性）=====
    QVector<cv::Mat> reidHistory;  // 最近N帧的ReID特征
    static constexpr int REID_HISTORY_SIZE = 5;
};

/**
 * @brief 丢失轨迹信息（用于重识别画廊）
 */
struct LostTrack {
    int trackId;
    cv::Mat reidFeature;          // 最终ReID特征（历史平均）
    cv::Rect lastBbox;            // 最后位置
    cv::Point2f lastCenter;       // 最后中心点
    int visitorId{-1};            // 关联的稳定身份 ID
    QDateTime lostTime;           // 丢失时间
    int age;                      // 存活帧数（丢失时）
};

/**
 * @brief 增强型人员跟踪器
 *
 * 核心改进：
 * 1. 卡曼滤波运动预测 - 即使检测丢失也能预测位置
 * 2. 多阶段匹配 - IoU → 预测IoU → ReID特征
 * 3. 丢失轨迹画廊 - 短暂丢失后可重识别恢复
 * 4. ReID特征历史平均 - 提高身份匹配鲁棒性
 * 5. **全局身份注册表（一人一 ID）** - 新检测先解析稳定身份，
 *    若该身份已被某活跃轨迹占用，则直接并入该轨迹（丢弃重复 ID）；
 *    并在每帧对活跃轨迹做「外观 + 卡尔曼预测位置」的重复轨迹合并。
 */
class PersonTracker
{
public:
    PersonTracker(float iouThreshold = 0.3f, int maxMissed = 30);

    QVector<TrackedPerson> update(const QVector<cv::Rect> &detections,
                                  const cv::Mat &frame = cv::Mat(),
                                  ReIDExtractor *reidExtractor = nullptr);
    QVector<TrackedPerson> getActiveTracks() const;
    TrackedPerson *getTrack(int trackId);

    // ===== 配置接口 =====
    void setIouThreshold(float threshold) { m_iouThreshold = threshold; }
    void setMaxMissed(int maxMissed) { m_maxMissed = maxMissed; }

    /**
     * @brief 设置ReID匹配阈值（用于丢失轨迹重识别）
     */
    void setReidMatchThreshold(float threshold) { m_reidMatchThreshold = threshold; }

    /**
     * @brief 设置丢失轨迹画廊的最大保留时间（毫秒）
     */
    void setGalleryMaxAge(int maxAgeMs) { m_galleryMaxAgeMs = maxAgeMs; }

    /**
     * @brief 设置是否启用卡尔曼预测匹配
     */
    void setKalmanEnabled(bool enabled) { m_kalmanEnabled = enabled; }

    /**
     * @brief 设置是否启用ReID辅助匹配
     */
    void setReidMatchingEnabled(bool enabled) { m_reidMatchingEnabled = enabled; }

    /**
     * @brief 设置共享的全局身份注册表（跨轨迹稳定 ID 的唯一来源）
     */
    void setVisitorRegistry(VisitorRegistry *registry) { m_registry = registry; }

    /**
     * @brief 重复轨迹合并的外观阈值（越高越保守，建议 0.82~0.92）
     */
    void setTrackMergeSimilarity(float s)
    {
        m_mergeSimilarity = (s < 0.5f) ? 0.5f : ((s > 0.99f) ? 0.99f : s);
    }

    /**
     * @brief 允许合并的“年轻轨迹”最大存活帧数（超过则要求极高相似度）
     */
    void setTrackMergeMaxYoungAge(int frames) { m_mergeMaxYoungAge = qMax(1, frames); }

    /**
     * @brief 获取当前活跃轨迹数量
     */
    int activeTrackCount() const { return m_tracks.size(); }

    /**
     * @brief 获取丢失画廊中的轨迹数量
     */
    int gallerySize() const { return m_lostGallery.size(); }

    /**
     * @brief 累计去重（重复轨迹合并）次数
     */
    int mergeCount() const { return m_mergeCount; }

private:
    QMap<int, TrackedPerson> m_tracks;
    int m_nextId;
    float m_iouThreshold;
    int m_maxMissed;

    // ===== ReID 匹配参数 =====
    float m_reidMatchThreshold{0.65f};
    bool m_reidMatchingEnabled{true};

    // ===== 卡尔曼预测参数 =====
    bool m_kalmanEnabled{true};

    // ===== 丢失轨迹画廊 =====
    QVector<LostTrack> m_lostGallery;
    int m_galleryMaxAgeMs{10000};  // 10秒

    // ===== 全局身份（一人一 ID）=====
    VisitorRegistry *m_registry{nullptr};
    QMap<int, int> m_personToTrack;    // personId -> 活跃 trackId
    QMap<int, int> m_trackToPerson;    // trackId -> personId

    // ===== 重复轨迹合并参数 =====
    float m_mergeSimilarity{0.85f};
    int m_mergeMaxYoungAge{20};
    int m_mergeCount{0};

    // ===== 核心匹配函数 =====
    float computeIoU(const cv::Rect &a, const cv::Rect &b);
    static cv::Point2f centerOf(const cv::Rect &r);
    static float centerDistance(const cv::Rect &a, const cv::Rect &b);

    /**
     * @brief 多阶段匹配算法
     *
     * 阶段1: IoU 匹配（基于当前位置）
     * 阶段2: 卡尔曼预测 IoU 匹配（基于预测位置）
     * 阶段3: ReID 特征匹配（基于外观特征）
     */
    void multiStageMatch(const QVector<cv::Rect> &detections,
                         const cv::Mat &frame,
                         ReIDExtractor *reidExtractor,
                         QVector<bool> &detMatched,
                         QVector<bool> &trackMatched,
                         const QList<int> &trackIds);

    /**
     * @brief 更新丢失轨迹画廊
     */
    void updateLostGallery();

    /**
     * @brief 从画廊中尝试重识别新检测
     * @param trackIdOut   命中的原轨迹 ID
     * @param visitorIdOut 命中的稳定身份 ID
     * @param simOut       命中相似度
     * @return 命中返回 true
     */
    bool tryReidFromGallery(const cv::Rect &detBbox,
                            const cv::Mat &detFeature,
                            ReIDExtractor *reidExtractor,
                            int *trackIdOut = nullptr,
                            int *visitorIdOut = nullptr,
                            float *simOut = nullptr);

    /**
     * @brief 计算ReID特征的平均值
     */
    cv::Mat computeAverageReid(const QVector<cv::Mat> &features) const;

    /**
     * @brief 更新轨迹的ReID特征历史
     */
    void updateReidHistory(TrackedPerson &track, const cv::Mat &newFeature);

    /**
     * @brief 轨迹的鲁棒外观特征（历史平均优先，回退单帧特征）
     */
    cv::Mat robustFeature(const TrackedPerson &track) const;

    // ===== 身份管理（一人一 ID）=====

    /// 用注册表解析稳定身份；返回 personId（<=0 表示不可用）
    int resolveIdentity(const cv::Mat &feature, const cv::Point2f &normPos,
                        float *simOut = nullptr, bool *createdOut = nullptr);

    /// 建立 trackId <-> personId 双向绑定
    void bindIdentity(int trackId, int personId);

    /// 解除某个轨迹的身份绑定
    void unbindIdentity(int trackId);

    /// 归一化位置（以画面尺寸为基准；画面为空时取中心）
    static cv::Point2f normalizedPos(const cv::Rect &bbox, const cv::Mat &frame);

    /// 由检测创建新轨迹（含卡尔曼初始化与ReID特征）
    TrackedPerson createTrack(const cv::Rect &det, const cv::Mat &frame,
                              ReIDExtractor *reidExtractor);

    /// 把检测并入既有轨迹（去重：同一人的重复框直接吃掉，不新建 ID）
    void absorbDetection(int trackId, const cv::Rect &det, const cv::Mat &feature);

    /// 把 dropTrack 合并进 keepTrack（同一人的两条轨迹合成一条）
    void absorbTrack(int keepTrackId, int dropTrackId);

    /// 合并重复轨迹（身份相同 → 直接合并；外观+时空门限 → 判定合并）
    void dedupTracks(ReIDExtractor *reidExtractor);

    /// 轨迹丢失 → 移入画廊并解除身份绑定
    void retireTrack(int trackId);
};

#endif // PERSONTRACKER_H
