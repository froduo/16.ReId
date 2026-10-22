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

void FolderMatchWorker::submit(quint64 jobId, int alightPersonId, const QString &alightDir,
                               const QVector<BoardFolderCandidate> &candidates, float threshold)
{
    QMetaObject::invokeMethod(this, [this, jobId, alightPersonId, alightDir,
                                     candidates, threshold]() {
        run(jobId, alightPersonId, alightDir, candidates, threshold);
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

QVector<cv::Mat> FolderMatchWorker::embedAll(DeepReID *deep, const QString &dir, int maxCount)
{
    QVector<cv::Mat> feats;
    if (!deep || !deep->ready()) return feats;

    const QStringList files = imageFiles(dir, maxCount);
    for (const QString &f : files) {
        cv::Mat img = cv::imread(f.toStdString(), cv::IMREAD_COLOR);
        if (img.empty()) continue;
        const cv::Rect full(0, 0, img.cols, img.rows);
        cv::Mat feat = deep->embed(img, full);
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
                            const QVector<BoardFolderCandidate> &candidates, float threshold)
{
    if (!m_useDeep || !m_deep || !m_deep->ready()) {
        LOG_WARN("[FolderMatch] 深度 ReID 后端不可用，跳过目录比对");
        emit finished(jobId, -1, QString(), 0.0f, false);
        return;
    }

    const QVector<cv::Mat> probe = embedAll(m_deep, alightDir, m_maxImages);
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
        const QVector<cv::Mat> templ = embedAll(m_deep, c.dir, m_maxImages);
        if (templ.isEmpty()) continue;
        const float sim = bestAvgSimilarity(m_deep, probe, templ);
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

    const bool matched = (bestPersonId > 0 && bestSim >= threshold);
    // 最高与次高的差距：差距过小说明候选区分度不足，匹配置信度低（便于后续阈值标定）
    const float margin = (secondSim > 0.0f) ? (bestSim - secondSim) : 1.0f;
    LOG_INFO(QString("[FolderMatch] jobId=%1 下车=person_%2 探测图=%3 候选=%4 | "
                     "best=person_%5(sim=%6) second=%7 margin=%8 阈值=%9 matched=%10 | 排名: %11")
                 .arg(jobId).arg(alightPersonId).arg(probe.size()).arg(candidates.size())
                 .arg(bestPersonId).arg(bestSim, 0, 'f', 4)
                 .arg(secondSim, 0, 'f', 4)
                 .arg(margin, 0, 'f', 4)
                 .arg(threshold, 0, 'f', 3)
                 .arg(matched ? "true" : "false")
                 .arg(ranked.isEmpty() ? QStringLiteral("无") : ranked));

    emit finished(jobId, bestPersonId, bestDir, bestSim, matched);
}
