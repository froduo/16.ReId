#ifndef PERSONTRACKER_H
#define PERSONTRACKER_H

#include <QVector>
#include <QMap>
#include <QDateTime>
#include <QElapsedTimer>
#include <opencv2/opencv.hpp>
#include <memory>

// 前向声明
class ReIDExtractor;

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
};

struct TrackedPerson {
    int trackId;
    cv::Rect bbox;
    cv::Point2f center;
    cv::Point2f prevCenter;  // 前一帧中心点（用于方向检测）
    int age;             // 存活帧数
    int missedFrames;    // 丢失帧数
    bool isInsideRegion; // 当前帧是否在检测区域内（点在多边形内）
    bool wasInsideRegion; // 前一帧是否在检测区域内
    /// 本帧是否被「真实检测」到（false=轨迹丢失、bbox 为卡尔曼预测外推）
    /// 下游（ZoneMonitor 等）应据此忽略预测框，避免外推位置造成误判
    bool detected{true};
    cv::Mat reidFeature; // ReID 特征向量
    QDateTime firstSeen;
    QDateTime lastSeen;
    QDateTime enterTime; // 进入区域时间
    QDateTime exitTime;  // 离开区域时间
    bool hasEntered;     // 是否已确认进入过区域
    bool hasExited;      // 是否已确认离开过区域
    cv::Mat enterImage;  // 进入时的图像
    cv::Mat exitImage;   // 离开时的图像
    int visitorId;       // ReID 身份 ID（跨 track 持久化）

    // ===== 状态机：防抖动进入/离开检测 =====
    RegionCrossState crossState{RegionCrossState::Outside};
    int crossConfirmFrames{0};  // 穿越确认帧计数
    int crossDirectionSign{0};  // 穿越方向（+1=正向, -1=反向）

    // ===== 卡尔曼滤波 =====
    std::shared_ptr<KalmanTrack> kalman;  // 卡尔曼滤波器

    // ===== ReID 特征历史（多帧平均以提高鲁棒性）=====
    QVector<cv::Mat> reidHistory;  // 最近N帧的ReID特征（用于多帧时序融合 EMA）
    static constexpr int REID_HISTORY_SIZE = 10;
};

/**
 * @brief 丢失轨迹信息（用于重识别画廊）
 */
struct LostTrack {
    int trackId;
    cv::Mat reidFeature;          // 最终ReID特征（历史平均）
    cv::Rect lastBbox;            // 最后位置
    cv::Point2f lastCenter;       // 最后中心点
    int visitorId{-1};            // 关联的访客ID
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
     * @brief 获取当前活跃轨迹数量
     */
    int activeTrackCount() const { return m_tracks.size(); }

    /**
     * @brief 获取丢失画廊中的轨迹数量
     */
    int gallerySize() const { return m_lostGallery.size(); }

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

    // ===== 核心匹配函数 =====
    float computeIoU(const cv::Rect &a, const cv::Rect &b);

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
     */
    int tryReidFromGallery(const cv::Rect &detBbox,
                           const cv::Mat &detFeature,
                           ReIDExtractor *reidExtractor);

    /**
     * @brief 计算ReID特征的平均值
     */
    cv::Mat computeAverageReid(const QVector<cv::Mat> &features) const;

    /**
     * @brief 更新轨迹的ReID特征历史
     */
    void updateReidHistory(TrackedPerson &track, const cv::Mat &newFeature);
};

#endif // PERSONTRACKER_H
