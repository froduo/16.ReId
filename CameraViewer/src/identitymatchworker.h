#ifndef IDENTITYMATCHWORKER_H
#define IDENTITYMATCHWORKER_H

#include <QObject>
#include <QVector>
#include <QPair>
#include <opencv2/opencv.hpp>
#include "reidextractor.h"

class DeepReID;

/**
 * @brief 匹配模板：一位在车乘客的外观模板
 *
 *  - fused  ：时序融合后的模板特征（主匹配用；为空时由 frames 现场融合）
 *  - frames ：原始模板帧（用于 best-avg 精修，可空）
 */
struct MatchTemplate {
    int passengerId{-1};
    cv::Mat fused;
    QVector<cv::Mat> frames;
};

/**
 * @brief 下车身份比对后台工作器（多模态 + 时序融合 + 动态加权 + 匈牙利全局匹配）
 *
 * 流程：
 *   1. **时序融合**：对每个探测的多帧特征做 EMA 融合（ReIDExtractor::temporalFuse），
 *      模板同样以融合特征参与匹配；
 *   2. **多模态动态加权**：按颜色/纹理/身形三分量分别评分，权重随「该分量在本批
 *      候选间的区分度（极差）」自适应放大——区分度高的分量说话，抑制无区分度分量；
 *   3. **匈牙利全局匹配**：当一批有多个探测（如多人同时下车）时，构建 探测×模板
 *      相似度矩阵并做全局最优指派（而非逐个贪心），避免互相抢占造成的次优配对；
 *   4. **精修**：对全局指派得到的配对，用「探测序列 × 模板序列」的 best-avg 相似度
 *      与融合相似度加权融合，得到最终判定分数。
 *
 * 线程安全：submit()/submitBatch() 可在任意线程调用，内部通过 Qt::QueuedConnection
 * 投递到本对象所属线程串行执行。台账增删仍由主线程负责。
 */
class IdentityMatchWorker : public QObject
{
    Q_OBJECT
public:
    explicit IdentityMatchWorker(QObject *parent = nullptr);

    /// 设置三类分量基础权重（颜色/纹理/身形）；实际赋值投递到本对象线程串行执行
    void setComponentWeights(float color, float texture, float shape);

    /// 是否启用「动态加权」（默认启用；深度后端下不适用）
    void setDynamicWeightingEnabled(bool enabled);

    /**
     * @brief 绑定深度 ReID 后端（OSNet 现场微调模型）
     *
     * 深度特征（512 维余弦）区分力远强于手工特征，绑定后 runBatch() 直接走
     * 余弦相似度 + 匈牙利全局指派，跳过手工分量动态加权。
     * 线程安全：通过队列连接投递到本对象所属线程执行。
     */
    void setDeepReid(DeepReID *deep, bool use);

    /**
     * @brief 提交单条探测的比对任务（兼容旧接口）
     */
    void submit(quint64 jobId,
                QVector<cv::Mat> probeFeatures,
                QVector<MatchTemplate> templates,
                float threshold);

    /**
     * @brief 批量提交（多探测同时参与全局匹配）
     * @param probes    (jobId, 该探测的特征序列) 列表
     * @param templates 在车乘客模板快照
     * @param threshold 判定阈值
     *
     * 每个探测完成后都会 emit 一次 finished(jobId, ...)，jobId 原样回传。
     */
    void submitBatch(QVector<QPair<quint64, QVector<cv::Mat>>> probes,
                     QVector<MatchTemplate> templates,
                     float threshold);

signals:
    /**
     * @brief 单个探测的比对完成
     * @param bestId     判定/候选乘客 ID（-1 表示无候选）
     * @param similarity 最终相似度（融合 + best-avg 精修后）
     * @param matched    是否达到阈值且被全局指派
     */
    void finished(quint64 jobId, int bestId, float similarity, bool matched);

private:
    /// 实际计算（在对象所属线程执行）
    void runBatch(QVector<QPair<quint64, QVector<cv::Mat>>> probes,
                  QVector<MatchTemplate> templates,
                  float threshold);

    /// 序列相似度：对探针每帧取与模板序列的最优匹配再平均（best-avg）
    static float bestAvgSimilarity(ReIDExtractor &reid,
                                   const QVector<cv::Mat> &probe,
                                   const QVector<cv::Mat> &templ);

    /// 专用比对器（仅本线程访问）
    ReIDExtractor m_reid;
    bool m_dynamicWeighting{true};
};

#endif // IDENTITYMATCHWORKER_H
