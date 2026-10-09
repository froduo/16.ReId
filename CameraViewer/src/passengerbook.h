#ifndef PASSENGERBOOK_H
#define PASSENGERBOOK_H

/**
 * @file passengerbook.h
 * @brief 乘客台账：记录「谁上了车、何时上车、上车时的样子」，下车时用图像比对确认身份
 *
 * 业务闭环（配合 ZoneMonitor 的上车/下车事件）：
 *   1. 发生一次「上车」（图像中从上往下 A→B）时，调用 registerBoarding()
 *      记录 乘客ID + 上车时间 + 上车抓拍图片 + 外观特征；
 *   2. 发生一次「下车」时，调用 matchAndRemove()，把下车抓拍的外观特征与
 *      台账里所有「已上车但尚未下车」的乘客逐一比对，取相似度最高且超过阈值者，
 *      从而判断「是哪一个乘客下车了」；
 *   3. 台账以 JSON（特征 Base64）持久化，进程重启后仍可继续比对。
 *
 * 说明：外观特征来自 ReIDExtractor（HSV 直方图），与保存的上下车图片一一对应，
 * 因此这里的“与上车图片比较”实质上是比较两张抓拍图对应的特征向量。
 */

#include <QString>
#include <QVector>
#include <QMap>
#include <QPair>
#include <QDateTime>
#include <opencv2/opencv.hpp>

#include "identitymatchworker.h"   // MatchTemplate（含时序融合模板）

class ReIDExtractor;

class PassengerBook
{
public:
    /// 一条「已上车」记录
    struct BoardingRecord {
        int passengerId{-1};            // 乘客 ID
        QString boardTime;              // 上车时间（ISO 8601）
        QString boardImagePath;         // 上车抓拍图片路径（乘客ID_上车_时间）
        QVector<cv::Mat> features;      // 上车时外观特征（可含进入/离开两帧）
        cv::Mat fused;                  // 时序融合后的模板特征（EMA，比对主用）
    };

    /// 下车比对结果
    struct MatchResult {
        int passengerId{-1};    // 判定下车的乘客 ID；-1 表示无法判定
        float similarity{0.0f}; // 与台账中的最高相似度
        bool matched{false};    // 是否达到阈值（true=图像比对命中）
    };

    PassengerBook();

    void setReidExtractor(ReIDExtractor *extractor) { m_reid = extractor; }
    void setMatchThreshold(float threshold);
    float matchThreshold() const { return m_threshold; }

    /// 台账持久化目录（空则不做磁盘持久化）
    void setStorageDir(const QString &dir);
    QString storageDir() const { return m_storageDir; }

    /**
     * @brief 登记一次上车
     * @param preferredId 期望使用的乘客 ID（<=0 表示自动分配）
     * @param features    上车时的外观特征（可为 1~2 个）
     * @param boardTime   上车时间
     * @param imagePath   上车抓拍图片路径
     * @return 最终使用的乘客 ID
     */
    int registerBoarding(int preferredId, const QVector<cv::Mat> &features,
                         const QDateTime &boardTime, const QString &imagePath);

    /// 补写/更新上车抓拍图片路径
    void updateBoardingImage(int passengerId, const QString &imagePath);

    /**
     * @brief 下车比对：在「已上车未下车」乘客中找最相似者，命中后移出台账
     * @param probeFeatures 下车时的外观特征
     * @param fallbackId    比对失败时回退使用的乘客 ID（可 <=0）
     */
    MatchResult matchAndRemove(const QVector<cv::Mat> &probeFeatures, int fallbackId);

    bool isOnboard(int passengerId) const { return m_records.contains(passengerId); }
    int onboardCount() const { return m_records.size(); }
    QVector<BoardingRecord> onboardRecords() const;

    /// 在车乘客模板快照：passengerId -> 特征序列（深拷贝，供后台线程比对）
    QVector<QPair<int, QVector<cv::Mat>>> onboardTemplates() const;

    /**
     * @brief 在车乘客模板快照（多模态 + 时序融合 + 原始帧）
     *
     * 供 IdentityMatchWorker 做「动态加权 + 匈牙利全局匹配」，
     * 每个模板携带 fused（融合特征，主匹配）与 frames（原始帧，best-avg 精修）。
     */
    QVector<MatchTemplate> onboardTemplateBundle() const;

    /**
     * @brief 在线追加一帧上车特征并更新该乘客的时序融合模板（EMA）
     *
     * 用于上车后短期内继续采样到该乘客时，持续细化模板、提升后续下车比对命中率。
     * 注意：不在此处落盘（避免高频 IO），由调用方在合适时机 save()。
     */
    void addBoardingFeature(int passengerId, const cv::Mat &feature);

    /// 移除指定乘客（下车身份确认后调用，内部会持久化）
    void removePassenger(int passengerId);

    /// 自动分配一个未被占用的乘客 ID
    int allocatePassengerId(int preferredId = -1);

    void clear();

    /// 持久化（写 passenger_book.json）
    void save() const;
    /// 从磁盘恢复
    void load();

private:
    /// 两组特征的最大相似度（逐对比较取最大）
    float similarityOf(const QVector<cv::Mat> &a, const QVector<cv::Mat> &b) const;
    /// 两段序列的相似度：对探针序列的每一帧取与模板序列的最大相似度，再求平均
    float sequenceSimilarity(const QVector<cv::Mat> &probe,
                             const QVector<cv::Mat> &templ) const;

    QMap<int, BoardingRecord> m_records;   // passengerId -> 上车记录（当前在车上）
    ReIDExtractor *m_reid{nullptr};
    float m_threshold{0.70f};
    int m_nextPassengerId{1};
    QString m_storageDir;
};

#endif // PASSENGERBOOK_H
