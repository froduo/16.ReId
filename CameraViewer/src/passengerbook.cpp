#include "passengerbook.h"
#include "reidextractor.h"
#include "visitorlog.h"
#include "logger.h"

#include <algorithm>
#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

PassengerBook::PassengerBook()
{
}

void PassengerBook::setMatchThreshold(float threshold)
{
    m_threshold = std::max(0.0f, std::min(1.0f, threshold));
}

void PassengerBook::setStorageDir(const QString &dir)
{
    m_storageDir = dir;
}

int PassengerBook::allocatePassengerId(int preferredId)
{
    if (preferredId > 0 && !m_records.contains(preferredId)) {
        if (preferredId >= m_nextPassengerId) m_nextPassengerId = preferredId + 1;
        return preferredId;
    }
    // 跳过已被占用的 ID
    while (m_records.contains(m_nextPassengerId)) m_nextPassengerId++;
    return m_nextPassengerId++;
}

int PassengerBook::registerBoarding(int preferredId, const QVector<cv::Mat> &features,
                                    const QDateTime &boardTime, const QString &imagePath)
{
    int pid = allocatePassengerId(preferredId);

    BoardingRecord rec;
    rec.passengerId = pid;
    rec.boardTime = boardTime.isValid() ? boardTime.toString(Qt::ISODate)
                                        : QDateTime::currentDateTime().toString(Qt::ISODate);
    rec.boardImagePath = imagePath;
    for (const cv::Mat &f : features) {
        if (!f.empty()) rec.features.append(f.clone());
    }
    // 时序融合模板（EMA）：后续比对以此为主特征，兼顾稳定性与时效性
    if (!rec.features.isEmpty())
        rec.fused = ReIDExtractor::temporalFuse(rec.features, 0.4f);

    m_records[pid] = rec;
    if (pid >= m_nextPassengerId) m_nextPassengerId = pid + 1;

    LOG_INFO(QString("[PassengerBook] BOARDING registered: P%1 @ %2, features=%3, image=%4, onboard=%5")
                 .arg(pid).arg(rec.boardTime).arg(rec.features.size())
                 .arg(imagePath.isEmpty() ? "-" : imagePath)
                 .arg(m_records.size()));

    save();
    return pid;
}

void PassengerBook::updateBoardingImage(int passengerId, const QString &imagePath)
{
    if (!m_records.contains(passengerId)) return;
    m_records[passengerId].boardImagePath = imagePath;
    save();
}

PassengerBook::MatchResult PassengerBook::matchAndRemove(const QVector<cv::Mat> &probeFeatures,
                                                         int fallbackId)
{
    MatchResult result;

    int bestId = -1;
    float bestSim = 0.0f;

    for (auto it = m_records.constBegin(); it != m_records.constEnd(); ++it) {
        // 用“整段序列”相似度（对探针每帧取与模板序列的最优匹配再平均）
        const float sim = sequenceSimilarity(probeFeatures, it.value().features);
        if (sim > bestSim) {
            bestSim = sim;
            bestId = it.key();
        }
    }

    result.similarity = bestSim;

    if (bestId > 0 && bestSim >= m_threshold) {
        // 图像比对命中：判定该乘客下车
        result.passengerId = bestId;
        result.matched = true;
        m_records.remove(bestId);
        LOG_INFO(QString("[PassengerBook] ALIGHTING matched: P%1 (similarity=%2 >= %3), onboard=%4")
                     .arg(bestId).arg(bestSim, 0, 'f', 3)
                     .arg(m_threshold, 0, 'f', 2).arg(m_records.size()));
    } else if (fallbackId > 0 && m_records.contains(fallbackId)) {
        // 比对未达阈值，但跟踪身份指向台账中的乘客 → 采用回退身份
        result.passengerId = fallbackId;
        result.matched = false;
        result.similarity = sequenceSimilarity(probeFeatures, m_records.value(fallbackId).features);
        m_records.remove(fallbackId);
        LOG_WARN(QString("[PassengerBook] ALIGHTING best=%1 (sim=%2 < %3), fallback to P%4, onboard=%5")
                     .arg(bestId).arg(bestSim, 0, 'f', 3)
                     .arg(m_threshold, 0, 'f', 2).arg(fallbackId).arg(m_records.size()));
    } else {
        LOG_WARN(QString("[PassengerBook] ALIGHTING no match (best=%1 sim=%2 < %3, onboard=%4)")
                     .arg(bestId).arg(bestSim, 0, 'f', 3)
                     .arg(m_threshold, 0, 'f', 2).arg(m_records.size()));
    }

    save();
    return result;
}

QVector<PassengerBook::BoardingRecord> PassengerBook::onboardRecords() const
{
    QVector<BoardingRecord> out;
    out.reserve(m_records.size());
    for (auto it = m_records.constBegin(); it != m_records.constEnd(); ++it)
        out.append(it.value());
    return out;
}

QVector<QPair<int, QVector<cv::Mat>>> PassengerBook::onboardTemplates() const
{
    QVector<QPair<int, QVector<cv::Mat>>> out;
    out.reserve(m_records.size());
    for (auto it = m_records.constBegin(); it != m_records.constEnd(); ++it) {
        QVector<cv::Mat> feats;
        for (const cv::Mat &f : it.value().features) {
            if (!f.empty()) feats.append(f.clone());
        }
        out.append(qMakePair(it.key(), feats));
    }
    return out;
}

QVector<MatchTemplate> PassengerBook::onboardTemplateBundle() const
{
    QVector<MatchTemplate> out;
    out.reserve(m_records.size());
    for (auto it = m_records.constBegin(); it != m_records.constEnd(); ++it) {
        const BoardingRecord &rec = it.value();
        MatchTemplate t;
        t.passengerId = it.key();
        t.fused = rec.fused.empty() ? cv::Mat() : rec.fused.clone();
        for (const cv::Mat &f : rec.features) {
            if (!f.empty()) t.frames.append(f.clone());
        }
        out.append(t);
    }
    return out;
}

void PassengerBook::addBoardingFeature(int passengerId, const cv::Mat &feature)
{
    if (feature.empty() || !m_records.contains(passengerId)) return;

    BoardingRecord &rec = m_records[passengerId];
    rec.features.append(feature.clone());

    // 控制序列长度（保留最近 N 帧），避免长时间驻留内存膨胀
    static constexpr int MAX_FEATURES = 30;
    while (rec.features.size() > MAX_FEATURES) rec.features.removeFirst();

    cv::Mat ff = feature.reshape(1, 1);
    ff.convertTo(ff, CV_32F);
    if (rec.fused.empty() || rec.fused.size() != ff.size()
        || rec.fused.channels() != ff.channels()) {
        rec.fused = ReIDExtractor::temporalFuse(rec.features, 0.4f);
    } else {
        rec.fused = ff * 0.4f + rec.fused * 0.6f;   // EMA 更新融合模板
    }

    LOG_INFO(QString("[PassengerBook] P%1 模板在线细化: frames=%2 (EMA 融合已更新)")
                 .arg(passengerId).arg(rec.features.size()));
}

void PassengerBook::removePassenger(int passengerId)
{
    if (m_records.remove(passengerId) > 0) {
        LOG_INFO(QString("[PassengerBook] removed P%1 (onboard=%2)")
                     .arg(passengerId).arg(m_records.size()));
        save();
    }
}

float PassengerBook::similarityOf(const QVector<cv::Mat> &a, const QVector<cv::Mat> &b) const
{
    if (!m_reid || a.isEmpty() || b.isEmpty()) return 0.0f;

    float best = 0.0f;
    for (const cv::Mat &fa : a) {
        if (fa.empty()) continue;
        for (const cv::Mat &fb : b) {
            if (fb.empty()) continue;
            const float s = m_reid->compare(fa, fb);
            if (s > best) best = s;
        }
    }
    return best;
}

float PassengerBook::sequenceSimilarity(const QVector<cv::Mat> &probe,
                                        const QVector<cv::Mat> &templ) const
{
    if (!m_reid || probe.isEmpty() || templ.isEmpty()) return 0.0f;

    float sum = 0.0f;
    int counted = 0;
    for (const cv::Mat &p : probe) {
        if (p.empty()) continue;
        float best = 0.0f;
        for (const cv::Mat &t : templ) {
            if (t.empty()) continue;
            const float s = m_reid->compare(p, t);
            if (s > best) best = s;
        }
        sum += best;
        counted++;
    }
    return (counted > 0) ? (sum / static_cast<float>(counted)) : 0.0f;
}

void PassengerBook::clear()
{
    m_records.clear();
    LOG_INFO("[PassengerBook] cleared");
    save();
}

void PassengerBook::save() const
{
    if (m_storageDir.isEmpty()) return;

    QDir dir(m_storageDir);
    if (!dir.exists() && !dir.mkpath(".")) {
        LOG_WARN("[PassengerBook] cannot create storage dir: " + m_storageDir);
        return;
    }

    QJsonObject root;
    root["next_id"] = m_nextPassengerId;

    QJsonArray arr;
    for (auto it = m_records.constBegin(); it != m_records.constEnd(); ++it) {
        const BoardingRecord &rec = it.value();
        QJsonObject o;
        o["passenger_id"] = rec.passengerId;
        o["board_time"] = rec.boardTime;
        o["board_image"] = rec.boardImagePath;

        QJsonArray feats;
        for (const cv::Mat &f : rec.features)
            feats.append(VisitorLog::matToBase64(f));
        o["features"] = feats;

        arr.append(o);
    }
    root["passengers"] = arr;

    QFile file(dir.filePath("passenger_book.json"));
    if (file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        file.write(QJsonDocument(root).toJson(QJsonDocument::Compact));
        file.close();
    } else {
        LOG_WARN("[PassengerBook] failed to write passenger_book.json");
    }
}

void PassengerBook::load()
{
    if (m_storageDir.isEmpty()) return;

    QFile file(QDir(m_storageDir).filePath("passenger_book.json"));
    if (!file.exists() || !file.open(QIODevice::ReadOnly)) {
        LOG_INFO("[PassengerBook] no previous passenger_book.json");
        return;
    }

    const QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    file.close();
    if (!doc.isObject()) return;

    const QJsonObject root = doc.object();
    m_nextPassengerId = root.value("next_id").toInt(1);

    m_records.clear();
    const QJsonArray arr = root.value("passengers").toArray();
    for (const QJsonValue &v : arr) {
        const QJsonObject o = v.toObject();
        BoardingRecord rec;
        rec.passengerId = o.value("passenger_id").toInt(-1);
        rec.boardTime = o.value("board_time").toString();
        rec.boardImagePath = o.value("board_image").toString();
        const QJsonArray feats = o.value("features").toArray();
        int dropped = 0;
        for (const QJsonValue &fv : feats) {
            cv::Mat f = VisitorLog::base64ToMat(fv.toString());
            if (f.empty()) continue;
            // 兼容性保护：仅接受与当前描述子维度一致的特征。
            // 特征维度在版本升级中可能变化（如颜色直方图分箱调整），
            // 旧维度特征参与比对会恒为 0，导致无法匹配，故此处直接丢弃。
            if (f.total() != static_cast<size_t>(ReIDExtractor::featureDim())) {
                dropped++;
                continue;
            }
            rec.features.append(f);
        }
        if (dropped > 0) {
            LOG_WARN(QString("[PassengerBook] P%1 丢弃 %2 个维度不兼容的旧特征（当前维度=%3）")
                         .arg(rec.passengerId).arg(dropped)
                         .arg(ReIDExtractor::featureDim()));
        }
        if (rec.passengerId > 0 && !rec.features.isEmpty()) {
            // 融合模板不落盘，加载后按序重新计算（EMA）
            rec.fused = ReIDExtractor::temporalFuse(rec.features, 0.4f);
            m_records[rec.passengerId] = rec;
            if (rec.passengerId >= m_nextPassengerId)
                m_nextPassengerId = rec.passengerId + 1;
        }
    }

    LOG_INFO(QString("[PassengerBook] loaded %1 onboard passengers (nextId=%2)")
                 .arg(m_records.size()).arg(m_nextPassengerId));
}
