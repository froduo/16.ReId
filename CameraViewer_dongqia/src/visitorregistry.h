#ifndef VISITORREGISTRY_H
#define VISITORREGISTRY_H

/**
 * @file visitorregistry.h
 * @brief 全局访客身份注册表（跨轨迹稳定 ID）
 *
 * 解决的问题：同一个人移动过程中因跟丢/重检测被分配多个 ID，导致
 * 「上车 / 下车」被拆成多条记录。本注册表是**唯一身份来源**：
 *   - 维护每个人的外观特征画廊（多帧运行平均）+ 最后出现时间/位置
 *   - 新轨迹出现时按「外观相似度 + 时空门限」解析到既有 personId，命中则复用
 *   - 自适应阈值：刚见过且位置接近更宽容；隔得久、或该人当前已在场内更严格
 *   - 支持身份合并（absorb）：确认两个身份其实是同一人时归并，做到「一人一 ID」
 *
 * 被 PersonTracker（跟踪级）、ZoneMonitor（上下车）、MainWindow（显示）共用，
 * 从而保证 显示 ID == 跟踪 ID == 上下车 ID。
 *
 * 线程安全：跟踪在采集线程运行、上下车/显示在 UI 线程运行，因此所有公开
 * 接口以互斥锁保护内部画廊。
 */

#include <QMap>
#include <QVector>
#include <QPointF>
#include <QMutex>
#include <opencv2/opencv.hpp>

class ReIDExtractor;

class VisitorRegistry
{
public:
    struct Entry {
        int personId{-1};
        cv::Mat feature;                 // 运行平均外观特征（1×N, CV_32F）
        int featCount{0};
        qint64 firstSeenMs{0};
        qint64 lastSeenMs{0};
        cv::Point2f lastNormPos{-1.0f, -1.0f};  // 最后一次归一化位置
        bool inside{false};              // 当前是否处于监控通道/区域内
        int seenCount{0};
    };

    /// 身份解析结果（供跟踪器做「一人一 ID」判定）
    struct MatchInfo {
        int personId{-1};        // 命中或新建的身份 ID
        float similarity{0.0f};  // 命中时的外观相似度
        bool matched{false};     // 是否命中既有身份
        bool created{false};     // 是否新建身份
    };

    VisitorRegistry() = default;

    void setExtractor(ReIDExtractor *e) { m_extractor = e; }

    /// 基础匹配阈值（外观相似度，0~1）
    void setBaseThreshold(float t);
    float baseThreshold() const;

    /// 画廊保留时长（超时且不在场内的身份会被清理）
    void setGalleryMaxAgeMs(qint64 ms);

    /// 保证下一个分配的 ID 不小于指定值（用于加载历史记录后避免 ID 冲突）
    void ensureNextIdAtLeast(int minId);

    /**
     * @brief 解析身份：返回稳定的 personId（命中既有身份或新建）
     * @param feature 外观特征（可为空；为空时只会新建，不与既有身份匹配）
     * @param nowMs   当前毫秒时间戳
     * @param normPos 归一化位置（x,y ∈ [0,1]）
     */
    int resolve(const cv::Mat &feature, qint64 nowMs, const cv::Point2f &normPos);

    /**
     * @brief 解析身份并返回匹配详情（匹配度/是否新建），供去重判定使用
     */
    MatchInfo resolveEx(const cv::Mat &feature, qint64 nowMs, const cv::Point2f &normPos);

    /// 用新观测强化某人的外观画廊（不进行查找，仅更新已知身份）
    void reinforce(int personId, const cv::Mat &feature, qint64 nowMs,
                   const cv::Point2f &normPos);

    /**
     * @brief 身份合并：把 dropId 的观测量并入 keepId，并删除 dropId
     *
     * 用于「发现同一个人的两个身份」时归并，保证后续只产生一条记录。
     * @return 合并成功返回 true
     */
    bool absorb(int keepId, int dropId);

    void markInside(int personId, bool inside);
    bool isInside(int personId) const;
    bool contains(int personId) const;
    Entry entry(int personId) const;
    cv::Mat feature(int personId) const;
    cv::Point2f lastPos(int personId) const;
    qint64 lastSeenMs(int personId) const;
    QVector<int> ids() const;
    int count() const;
    void reset();

    /// 外观相似度（0~1），内部封装 ReIDExtractor::compare
    float similarity(const cv::Mat &a, const cv::Mat &b) const;

private:
    void updateFeatureLocked(Entry &e, const cv::Mat &f);
    void pruneLocked(qint64 nowMs);
    float similarityLocked(const cv::Mat &a, const cv::Mat &b) const;

    QMap<int, Entry> m_entries;
    ReIDExtractor *m_extractor{nullptr};
    float m_baseThreshold{0.70f};
    int m_nextId{1};
    qint64 m_galleryMaxAgeMs{300000};   // 5 分钟
    mutable QMutex m_mutex;
};

#endif // VISITORREGISTRY_H
