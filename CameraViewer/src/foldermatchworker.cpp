#include "foldermatchworker.h"
#include "reiddeep.h"
#include "logger.h"

#include <QDir>
#include <QFileInfo>
#include <algorithm>

FolderMatchWorker::FolderMatchWorker(QObject *parent)
    : QObject(parent)
{
}

void FolderMatchWorker::setDeepReid(DeepReID *deep, bool use)
{
    QMetaObject::invokeMethod(this, [this, deep, use]() {
        m_deep = deep;
        m_useDeep = use;
    }, Qt::QueuedConnection);
}

void FolderMatchWorker::setMaxImagesPerFolder(int n)
{
    QMetaObject::invokeMethod(this, [this, n]() {
        m_maxImages = qMax(1, n);
    }, Qt::QueuedConnection);
}

void FolderMatchWorker::setQualityKeep(int n)
{
    QMetaObject::invokeMethod(this, [this, n]() {
        m_qualityKeep = qMax(1, n);
    }, Qt::QueuedConnection);
}

void FolderMatchWorker::submit(quint64 jobId, int alightPersonId, const QString &alightDir,
                               const QVector<BoardFolderCandidate> &candidates, float threshold,
                               float minMargin)
{
    QMetaObject::invokeMethod(this, [this, jobId, alightPersonId, alightDir,
                                      candidates, threshold, minMargin]() {
        run(jobId, alightPersonId, alightDir, candidates, threshold, minMargin);
    }, Qt::QueuedConnection);
}

QStringList FolderMatchWorker::imageFiles(const QString &dir, int maxCount)
{
    QStringList out;
    if (dir.isEmpty()) return out;

    QDir d(dir);
    if (!d.exists()) return out;

    const QStringList names = d.entryList(QDir::Files | QDir::Readable, QDir::Name);
    for (const QString &n : names) {
        const QString lower = n.toLower();
        if (!(lower.endsWith(".jpg") || lower.endsWith(".jpeg")
              || lower.endsWith(".png") || lower.endsWith(".bmp")))
            continue;
        out.append(d.filePath(n));
        if (maxCount > 0 && out.size() >= maxCount) break;
    }
    return out;
}

// 单张裁剪图的清晰度（Laplacian 方差，越大越清晰）
static double imageSharpness(const cv::Mat &img)
{
    cv::Mat gray, lap;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::Laplacian(gray, lap, CV_64F);
    cv::Scalar mu, sigma;
    cv::meanStdDev(lap, mu, sigma);
    return sigma.val[0] * sigma.val[0];
}

QVector<cv::Mat> FolderMatchWorker::embedAll(DeepReID *deep, const QString &dir, int maxCount,
                                             int qualityKeep)
{
    QVector<cv::Mat> feats;
    if (!deep || !deep->ready()) return feats;

    const QStringList files = imageFiles(dir, maxCount);

    // ===== 质量筛选：模糊/过小帧会污染 OSNet 特征、放大不同乘客间的偶然相似 =====
    // 读图后按清晰度降序，只保留最清晰的 top-k 帧参与比对（实测可纠正 person_21↔person_19 误匹配）。
    struct Cand { double sharpness; cv::Mat img; };
    QVector<Cand> cands;
    cands.reserve(files.size());
    for (const QString &f : files) {
        cv::Mat img = cv::imread(f.toStdString(), cv::IMREAD_COLOR);
        if (img.empty()) continue;
        if (img.cols < 40 || img.rows < 40) continue;   // 过小图细节不足
        cands.append({imageSharpness(img), img});
    }
    if (cands.isEmpty()) return feats;

    std::sort(cands.begin(), cands.end(),
              [](const Cand &a, const Cand &b) { return a.sharpness > b.sharpness; });
    const int keep = qMin(cands.size(), qualityKeep);
    for (int i = 0; i < keep; ++i) {
        const cv::Rect full(0, 0, cands[i].img.cols, cands[i].img.rows);
        cv::Mat feat = deep->embed(cands[i].img, full);
        if (!feat.empty()) feats.append(feat);
    }
    return feats;
}

float FolderMatchWorker::bestAvgSimilarity(DeepReID *deep,
                                           const QVector<cv::Mat> &probe,
                                           const QVector<cv::Mat> &templ)
{
    if (!deep || probe.isEmpty() || templ.isEmpty()) return 0.0f;

    float sum = 0.0f;
    int counted = 0;
    for (const cv::Mat &p : probe) {
        if (p.empty()) continue;
        float best = 0.0f;
        for (const cv::Mat &t : templ) {
            if (t.empty()) continue;
            const float s = deep->compare(p, t);
            if (s > best) best = s;
        }
        sum += best;
        counted++;
    }
    return (counted > 0) ? (sum / static_cast<float>(counted)) : 0.0f;
}

void FolderMatchWorker::run(quint64 jobId, int alightPersonId, const QString &alightDir,
                            const QVector<BoardFolderCandidate> &candidates, float threshold,
                            float minMargin)
{
    if (!m_useDeep || !m_deep || !m_deep->ready()) {
        LOG_WARN("[FolderMatch] 深度 ReID 后端不可用，跳过目录比对");
        emit finished(jobId, -1, QString(), 0.0f, false);
        return;
    }

    const QVector<cv::Mat> probe = embedAll(m_deep, alightDir, m_maxImages, m_qualityKeep);
    if (probe.isEmpty()) {
        LOG_WARN(QString("[FolderMatch] 下车站目录无可用图像: %1").arg(alightDir));
        emit finished(jobId, -1, QString(), 0.0f, false);
        return;
    }

    int bestPersonId = -1;
    QString bestDir;
    float bestSim = 0.0f;
    float secondSim = 0.0f;
    QString ranked;

    for (const BoardFolderCandidate &c : candidates) {
        const QVector<cv::Mat> templ = embedAll(m_deep, c.dir, m_maxImages, m_qualityKeep);
        if (templ.isEmpty()) continue;
        // 双向 best-avg：对称验证，抑制单向「对每帧取 max」造成的偶然相似虚高
        const float fwd = bestAvgSimilarity(m_deep, probe, templ);
        const float bwd = bestAvgSimilarity(m_deep, templ, probe);
        const float sim = std::min(fwd, bwd);
        if (!ranked.isEmpty()) ranked += ", ";
        ranked += QString("person_%1=%2").arg(c.personId).arg(sim, 0, 'f', 3);
        if (sim > bestSim) {
            secondSim = bestSim;
            bestSim = sim;
            bestPersonId = c.personId;
            bestDir = c.dir;
        } else if (sim > secondSim) {
            secondSim = sim;
        }
    }

    // 最高与次高的差距：差距过小说明候选间区分度不足，匹配置信度低
    const float margin = (secondSim > 0.0f) ? (bestSim - secondSim) : 1.0f;
    // 判定：分数达标 + 区分度足够（margin 不低于最小阈值），否则保留待后续复核
    const bool matched = (bestPersonId > 0 && bestSim >= threshold && margin >= minMargin);
    LOG_INFO(QString("[FolderMatch] jobId=%1 下车=person_%2 探测图=%3 候选=%4 | "
                     "best=person_%5(sim=%6) second=%7 margin=%8 阈值=%9 minMargin=%10 matched=%11 | 排名: %12")
                 .arg(jobId).arg(alightPersonId).arg(probe.size()).arg(candidates.size())
                 .arg(bestPersonId).arg(bestSim, 0, 'f', 4)
                 .arg(secondSim, 0, 'f', 4)
                 .arg(margin, 0, 'f', 4)
                 .arg(threshold, 0, 'f', 3)
                 .arg(minMargin, 0, 'f', 3)
                 .arg(matched ? "true" : "false")
                 .arg(ranked.isEmpty() ? QStringLiteral("无") : ranked));

    emit finished(jobId, bestPersonId, bestDir, bestSim, matched);
}
