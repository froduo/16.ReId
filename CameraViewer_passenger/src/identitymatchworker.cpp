#include "identitymatchworker.h"
#include "hungarian.h"
#include "reiddeep.h"
#include "logger.h"

#include <QPair>
#include <QVector>
#include <algorithm>
#include <cmath>
#include <vector>

IdentityMatchWorker::IdentityMatchWorker(QObject *parent)
    : QObject(parent)
{
}

void IdentityMatchWorker::setComponentWeights(float color, float texture, float shape)
{
    // 投递到本对象所属线程执行，保证与 runBatch() 串行（线程安全）
    QMetaObject::invokeMethod(this,
        [this, color, texture, shape]() {
            m_reid.setComponentWeights(color, texture, shape);
        },
        Qt::QueuedConnection);
}

void IdentityMatchWorker::setDynamicWeightingEnabled(bool enabled)
{
    QMetaObject::invokeMethod(this, [this, enabled]() {
        m_dynamicWeighting = enabled;
    }, Qt::QueuedConnection);
}

void IdentityMatchWorker::setDeepReid(DeepReID *deep, bool use)
{
    QMetaObject::invokeMethod(this, [this, deep, use]() {
        m_reid.setDeepReid(deep, use);
    }, Qt::QueuedConnection);
}

float IdentityMatchWorker::bestAvgSimilarity(ReIDExtractor &reid,
                                             const QVector<cv::Mat> &probe,
                                             const QVector<cv::Mat> &templ)
{
    if (probe.isEmpty() || templ.isEmpty()) return 0.0f;

    float sum = 0.0f;
    int counted = 0;
    for (const cv::Mat &p : probe) {
        if (p.empty()) continue;
        float best = 0.0f;
        for (const cv::Mat &t : templ) {
            if (t.empty()) continue;
            const float s = reid.compare(p, t);
            if (s > best) best = s;
        }
        sum += best;
        counted++;
    }
    return (counted > 0) ? (sum / static_cast<float>(counted)) : 0.0f;
}

void IdentityMatchWorker::submit(quint64 jobId,
                                 QVector<cv::Mat> probeFeatures,
                                 QVector<MatchTemplate> templates,
                                 float threshold)
{
    QVector<QPair<quint64, QVector<cv::Mat>>> probes;
    probes.append(qMakePair(jobId, probeFeatures));
    submitBatch(probes, templates, threshold);
}

void IdentityMatchWorker::submitBatch(QVector<QPair<quint64, QVector<cv::Mat>>> probes,
                                      QVector<MatchTemplate> templates,
                                      float threshold)
{
    // 参数按值拷贝后投递到本对象所属线程串行执行（跨线程安全）
    QMetaObject::invokeMethod(this,
        [this, probes, templates, threshold]() {
            runBatch(probes, templates, threshold);
        },
        Qt::QueuedConnection);
}

void IdentityMatchWorker::runBatch(QVector<QPair<quint64, QVector<cv::Mat>>> probes,
                                   QVector<MatchTemplate> templates,
                                   float threshold)
{
    const int P = probes.size();
    const int T = templates.size();
    if (P == 0) return;

    if (T == 0) {
        // 无在车模板：直接报告无候选
        for (const auto &pr : probes)
            emit finished(pr.first, -1, 0.0f, false);
        return;
    }

    // ===== 1) 时序融合：探测与模板都取 EMA 融合特征 =====
    QVector<cv::Mat> probeFused(P), templFused(T);
    for (int i = 0; i < P; ++i)
        probeFused[i] = ReIDExtractor::temporalFuse(probes[i].second, 0.4f);
    for (int j = 0; j < T; ++j)
        templFused[j] = templates[j].fused.empty()
                ? ReIDExtractor::temporalFuse(templates[j].frames, 0.4f)
                : templates[j].fused;

    // ===== 2/3) 相似度矩阵 =====
    //  - 深度后端（OSNet 现场微调）：直接余弦相似度，区分力强且无需分量加权；
    //  - 手工后端：分量评分 + 动态加权（权重 ∝ 分量在候选间的区分度）。
    std::vector<std::vector<double>> sim(P, std::vector<double>(T, 0.0));

    if (m_reid.deepEnabled()) {
        for (int i = 0; i < P; ++i) {
            if (probeFused[i].empty()) continue;
            for (int j = 0; j < T; ++j) {
                if (templFused[j].empty()) continue;
                sim[i][j] = m_reid.compare(probeFused[i], templFused[j]);
            }
        }
    } else {
        std::vector<std::vector<ComponentScores>> comps(
                    P, std::vector<ComponentScores>(T, ComponentScores{}));
        for (int i = 0; i < P; ++i) {
            if (probeFused[i].empty()) continue;
            for (int j = 0; j < T; ++j) {
                if (templFused[j].empty()) continue;
                comps[i][j] = m_reid.compareComponents(probeFused[i], templFused[j]);
            }
        }

        const float baseC = m_reid.weightColor();
        const float baseT = m_reid.weightTexture();
        const float baseS = m_reid.weightShape();

        for (int i = 0; i < P; ++i) {
            float minC = 1.0f, maxC = 0.0f;
            float minT = 1.0f, maxT = 0.0f;
            float minS = 1.0f, maxS = 0.0f;
            for (int j = 0; j < T; ++j) {
                minC = std::min(minC, comps[i][j].color);   maxC = std::max(maxC, comps[i][j].color);
                minT = std::min(minT, comps[i][j].texture); maxT = std::max(maxT, comps[i][j].texture);
                minS = std::min(minS, comps[i][j].shape);   maxS = std::max(maxS, comps[i][j].shape);
            }

            float wC = baseC, wT = baseT, wS = baseS;
            if (m_dynamicWeighting && T > 1) {
                // 权重 ∝ 该分量在候选间的区分度（极差）：有区分度的分量被放大，
                // 全体都高分/低分的分量被抑制，从而自适应不同场景下的有效模态。
                const float spreadC = maxC - minC;
                const float spreadT = maxT - minT;
                const float spreadS = maxS - minS;
                wC = baseC * (0.25f + spreadC);
                wT = baseT * (0.25f + spreadT);
                wS = baseS * (0.25f + spreadS);
            }

            for (int j = 0; j < T; ++j)
                sim[i][j] = comps[i][j].fused(wC, wT, wS);
        }
    }

    // ===== 4) 匈牙利全局最优指派（最大化总相似度）=====
    const std::vector<int> assign = hungarianAssignMax(sim);

    // ===== 5) 逐探测输出结果（含 best-avg 精修与诊断日志）=====
    for (int i = 0; i < P; ++i) {
        int col = (i < static_cast<int>(assign.size())) ? assign[i] : -1;

        // 单探测候选排名（用于日志）
        QVector<QPair<float, int>> ranked;
        ranked.reserve(T);
        for (int j = 0; j < T; ++j)
            ranked.append(qMakePair(static_cast<float>(sim[i][j]), templates[j].passengerId));
        std::sort(ranked.begin(), ranked.end(),
                  [](const QPair<float, int> &a, const QPair<float, int> &b) {
                      return a.first > b.first;
                  });

        int bestId = -1;
        float bestSim = 0.0f;
        if (col >= 0 && col < T) {
            bestId  = templates[col].passengerId;
            bestSim = static_cast<float>(sim[i][col]);
        } else if (!ranked.isEmpty()) {
            // 未被指派（探测数 > 模板数）：取本探测最优候选，供回退/诊断
            bestSim = ranked.first().first;
            bestId  = ranked.first().second;
        }

        // best-avg 精修（仅对全局指派得到的配对）
        float finalSim = bestSim;
        if (col >= 0 && col < T) {
            const float ba = bestAvgSimilarity(m_reid, probes[i].second, templates[col].frames);
            if (ba > 0.0f) finalSim = 0.55f * bestSim + 0.45f * ba;
        }

        const bool assigned = (col >= 0 && col < T);
        const bool matched  = (assigned && bestId > 0 && finalSim >= threshold);

        QString detail;
        const int topN = std::min(5, static_cast<int>(ranked.size()));
        for (int k = 0; k < topN; ++k) {
            if (k) detail += QStringLiteral(", ");
            detail += QString("P%1=%2").arg(ranked[k].second).arg(ranked[k].first, 0, 'f', 3);
        }

        LOG_INFO(QString("[IdentityMatch] jobId=%1 批量P=%2 模板T=%3 探测帧=%4 | 全局指派=%5 "
                         "best=P%6(sim=%7→精修%8) 阈值=%9 matched=%10 | top候选: %11")
                     .arg(probes[i].first).arg(P).arg(T).arg(probes[i].second.size())
                     .arg(assigned ? QString("P%1").arg(bestId) : QStringLiteral("未指派"))
                     .arg(bestId).arg(bestSim, 0, 'f', 3).arg(finalSim, 0, 'f', 3)
                     .arg(threshold, 0, 'f', 3)
                     .arg(matched ? "true" : "false")
                     .arg(detail.isEmpty() ? QStringLiteral("(无)") : detail));

        emit finished(probes[i].first, bestId, finalSim, matched);
    }
}
