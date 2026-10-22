#include "personfolder.h"
#include "logger.h"

#include <QDir>
#include <QFile>
#include <QByteArray>
#include <QJsonArray>
#include <QJsonDocument>

PersonFolder::PersonFolder(const QString &basePath)
    : m_basePath(basePath)
{
    if (!m_basePath.isEmpty()) {
        QDir().mkpath(m_basePath);
    }
}

void PersonFolder::reset()
{
    closeAllVideos();
    m_trackToPerson.clear();
    m_personOrder.clear();
    m_personDate.clear();
    m_personTick.clear();
    m_personSeq.clear();
    m_boarding.clear();
    m_eventTime.clear();
    m_matchedWith.clear();
    m_matchedSim.clear();
    m_matched.clear();
    m_videoSession.clear();
    m_nextPersonId = 1;
    LOG_INFO("[PersonFolder] 会话重置：person 序号归 1");
}

QString PersonFolder::dirFor(int personId, const QString &sub) const
{
    if (personId <= 0) return QString();
    QString d = QDir(m_basePath).filePath(personName(personId));
    if (!sub.isEmpty()) d = QDir(d).filePath(sub);
    return d;
}

QString PersonFolder::personName(int personId) const
{
    const QString date = m_personDate.value(personId);
    return date.isEmpty()
            ? QStringLiteral("person_%1").arg(personId)
            : QStringLiteral("person_%1_%2").arg(personId).arg(date);
}

QString PersonFolder::personDir(int personId) const
{
    return dirFor(personId, QString());
}

QString PersonFolder::rawImagesDir(int personId) const
{
    return dirFor(personId, QStringLiteral("原始图片"));
}

QString PersonFolder::cropImagesDir(int personId) const
{
    return dirFor(personId, QStringLiteral("乘客图片"));
}

QString PersonFolder::markerPath(int personId) const
{
    return QDir(personDir(personId)).filePath(QStringLiteral("match.txt"));
}

QString PersonFolder::summaryPath() const
{
    return QDir(m_basePath).filePath(QStringLiteral("match.txt"));
}

QJsonObject PersonFolder::stateToJson() const
{
    QJsonObject o;
    o["nextPersonId"] = m_nextPersonId;

    QJsonArray persons;
    for (const int pid : m_personOrder) {
        QJsonObject p;
        p["id"] = pid;
        p["date"] = m_personDate.value(pid);
        p["boarding"] = m_boarding.value(pid, false);
        p["time"] = m_eventTime.value(pid).toString(Qt::ISODateWithMs);
        p["matchedWith"] = m_matchedWith.value(pid, -1);
        p["matchedSim"] = static_cast<double>(m_matchedSim.value(pid, 0.0f));
        persons.append(p);
    }
    o["persons"] = persons;
    return o;
}

void PersonFolder::stateFromJson(const QJsonObject &o)
{
    m_nextPersonId = qMax(1, o.value(QStringLiteral("nextPersonId")).toInt(1));
    m_personDate.clear();
    m_boarding.clear();
    m_eventTime.clear();
    m_matchedWith.clear();
    m_matchedSim.clear();
    m_matched.clear();
    m_personOrder.clear();

    const QJsonArray persons = o.value(QStringLiteral("persons")).toArray();
    for (const QJsonValue &v : persons) {
        const QJsonObject p = v.toObject();
        const int pid = p.value(QStringLiteral("id")).toInt();
        if (pid <= 0) continue;
        m_personOrder.append(pid);
        m_personDate.insert(pid, p.value(QStringLiteral("date")).toString());
        m_boarding.insert(pid, p.value(QStringLiteral("boarding")).toBool());
        m_eventTime.insert(pid, QDateTime::fromString(
                               p.value(QStringLiteral("time")).toString(), Qt::ISODateWithMs));
        const int mw = p.value(QStringLiteral("matchedWith")).toInt(-1);
        if (mw > 0) {
            m_matchedWith.insert(pid, mw);
            m_matchedSim.insert(pid, static_cast<float>(p.value(QStringLiteral("matchedSim")).toDouble()));
            m_matched.insert(pid);
        }
    }
}

int PersonFolder::ensurePerson(int trackId)
{
    if (trackId <= 0) return -1;
    if (m_trackToPerson.contains(trackId)) return m_trackToPerson.value(trackId);

    const int pid = m_nextPersonId++;
    m_trackToPerson.insert(trackId, pid);
    m_personOrder.append(pid);
    m_personDate.insert(pid, QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd_HHmmss")));
    m_personTick.insert(pid, 0);
    m_personSeq.insert(pid, 0);

    if (!QDir().mkpath(rawImagesDir(pid)) || !QDir().mkpath(cropImagesDir(pid))) {
        LOG_WARN("[PersonFolder] 无法创建乘客目录: " + personDir(pid));
        return pid;
    }

    LOG_INFO(QString("[PersonFolder] 新乘客出现: person_%1 (track=%2) 目录=%3")
                 .arg(pid).arg(trackId).arg(personDir(pid)));
    return pid;
}

int PersonFolder::personIdForTrack(int trackId) const
{
    return m_trackToPerson.value(trackId, -1);
}

int PersonFolder::reusePerson(int trackId, int personId)
{
    if (trackId <= 0 || personId <= 0) return -1;

    // 该 track 已挂接某个 person：直接返回现有编号
    if (m_trackToPerson.contains(trackId))
        return m_trackToPerson.value(trackId);

    // 续接已有 person（不新增 person_N、不换号）：沿用其目录与图片序号
    m_trackToPerson.insert(trackId, personId);
    if (!m_personTick.contains(personId)) {
        m_personTick.insert(personId, 0);
        m_personSeq.insert(personId, 0);
        m_personDate.insert(personId, QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd_HHmmss")));
        m_personOrder.append(personId);
    }

    QDir().mkpath(rawImagesDir(personId));
    QDir().mkpath(cropImagesDir(personId));
    LOG_INFO(QString("[PersonFolder] 续接编号: track=%1 → person_%2（时间相邻且 ReID 相似，不换号）")
                 .arg(trackId).arg(personId));
    return personId;
}

bool PersonFolder::saveFrames(int personId, const cv::Mat &fullFrame, const cv::Mat &crop,
                              const QDateTime &ts)
{
    if (personId <= 0 || fullFrame.empty() || crop.empty()) return false;

    int &tick = m_personTick[personId];
    if ((tick++ % m_saveEveryN) != 0) return false;

    const QString rawDir = rawImagesDir(personId);
    const QString cropDir = cropImagesDir(personId);
    QDir().mkpath(rawDir);
    QDir().mkpath(cropDir);

    const QDateTime t = ts.isValid() ? ts : QDateTime::currentDateTime();
    const int seq = ++m_personSeq[personId];
    const QString name = QString("frame_%1_%2.jpg")
                             .arg(seq, 5, 10, QChar('0'))
                             .arg(t.toString("HHmmss_zzz"));

    bool ok = cv::imwrite(QDir(rawDir).filePath(name).toStdString(), fullFrame);
    ok = cv::imwrite(QDir(cropDir).filePath(name).toStdString(), crop) && ok;
    if (!ok) {
        LOG_WARN("[PersonFolder] 保存图片失败: " + personDir(personId));
    }

    // 同步写入「原始图片对应的视频」
    ensureVideoOpen(personId, fullFrame);
    const auto wit = m_writers.find(personId);
    if (wit != m_writers.end() && wit.value() && wit.value()->isOpened()) {
        wit.value()->write(fullFrame);
    }
    return ok;
}

void PersonFolder::ensureVideoOpen(int personId, const cv::Mat &frame)
{
    if (personId <= 0 || frame.empty()) return;
    if (m_writers.contains(personId) && m_writers.value(personId)
        && m_writers.value(personId)->isOpened())
        return;

    const QString dir = rawImagesDir(personId);
    QDir().mkpath(dir);

    static constexpr double VIDEO_FPS = 25.0;
    // 优先 AVI + MJPG：每帧为独立 JPEG，即使程序异常退出、索引未写全，
    // VLC/ffmpeg 也能重建索引播放；mp4 缺 moov 原子则基本无法打开。
    // 消失后再次续接同一编号时使用递增分段名，避免覆盖此前已关闭的视频段落。
    const int session = ++m_videoSession[personId];
    const QString base = session <= 1 ? QStringLiteral("视频")
                                      : QStringLiteral("视频_%1").arg(session);
    QString path = QDir(dir).filePath(base + QStringLiteral(".avi"));
    auto w = std::make_shared<cv::VideoWriter>();
    w->open(path.toStdString(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
            VIDEO_FPS, frame.size(), true);
    if (!w->isOpened()) {
        path = QDir(dir).filePath(base + QStringLiteral(".mp4"));
        w->open(path.toStdString(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                VIDEO_FPS, frame.size(), true);
    }
    if (!w->isOpened()) {
        LOG_WARN("[PersonFolder] 无法打开视频写入器: " + path);
        return;
    }
    m_writers.insert(personId, w);
    LOG_INFO(QString("[PersonFolder] person_%1 开始录制原始图片视频: %2")
                 .arg(personId).arg(path));
}

void PersonFolder::closeVideo(int personId)
{
    auto it = m_writers.find(personId);
    if (it == m_writers.end()) return;
    if (it.value()) it.value()->release();
    m_writers.remove(personId);
    LOG_INFO(QString("[PersonFolder] person_%1 原始图片视频已关闭").arg(personId));
}

void PersonFolder::closeAllVideos()
{
    for (auto it = m_writers.begin(); it != m_writers.end(); ++it) {
        if (it.value()) it.value()->release();
    }
    m_writers.clear();
}

void PersonFolder::writeMarker(int personId)
{
    if (personId <= 0) return;
    QDir().mkpath(personDir(personId));

    const QString path = markerPath(personId);
    QFile f(path);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        LOG_WARN("[PersonFolder] 无法写入标记文件: " + path);
        return;
    }

    QByteArray data;
    if (m_boarding.contains(personId)) {
        data += QString("方向: %1\n").arg(m_boarding.value(personId) ? "上车" : "下车").toUtf8();
    }
    if (m_eventTime.contains(personId)) {
        data += QString("时间: %1\n")
                    .arg(m_eventTime.value(personId).toString("yyyy-MM-dd HH:mm:ss")).toUtf8();
    }
    if (m_matchedWith.contains(personId)) {
        data += QString("匹配对象: %1\n").arg(personName(m_matchedWith.value(personId))).toUtf8();
    }
    if (m_matchedSim.contains(personId)) {
        data += QString("相似度: %1\n").arg(m_matchedSim.value(personId), 0, 'f', 4).toUtf8();
    }
    f.write(data);
    f.close();
}

void PersonFolder::writeSummary()
{
    QDir().mkpath(m_basePath);
    const QString path = summaryPath();
    QFile f(path);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        LOG_WARN("[PersonFolder] 无法写入汇总标记文件: " + path);
        return;
    }

    QByteArray data;
    data += "乘客列表汇总\n";
    data += "----------------------------------------\n";
    for (const int pid : m_personOrder) {
        const QString dirText = m_boarding.contains(pid)
                ? QString::fromUtf8(m_boarding.value(pid) ? "上车" : "下车")
                : QStringLiteral("未判定");
        QString matchText = QStringLiteral("未匹配");
        if (m_matchedWith.contains(pid)) {
            matchText = QString("已匹配 → %1")
                            .arg(personName(m_matchedWith.value(pid)));
            if (m_matchedSim.contains(pid)) {
                matchText += QString(" (sim=%1)").arg(m_matchedSim.value(pid), 0, 'f', 4);
            }
        }
        data += QString("%1: 方向=%2, %3\n")
                    .arg(personName(pid), dirText, matchText).toUtf8();
    }
    f.write(data);
    f.close();
}

void PersonFolder::setDirection(int personId, bool boarding, const QDateTime &time)
{
    if (personId <= 0) return;
    m_boarding.insert(personId, boarding);
    m_eventTime.insert(personId, time.isValid() ? time : QDateTime::currentDateTime());
    writeMarker(personId);
    writeSummary();

    LOG_INFO(QString("[PersonFolder] person_%1 方向判定: %2")
                 .arg(personId).arg(boarding ? "上车" : "下车"));
}

void PersonFolder::markMatched(int boardPersonId, int alightPersonId,
                               float similarity, const QDateTime &time)
{
    Q_UNUSED(time);
    if (boardPersonId > 0) {
        m_matchedWith.insert(boardPersonId, alightPersonId);
        m_matchedSim.insert(boardPersonId, similarity);
        m_matched.insert(boardPersonId);
    }
    if (alightPersonId > 0) {
        m_matchedWith.insert(alightPersonId, boardPersonId);
        m_matchedSim.insert(alightPersonId, similarity);
        m_matched.insert(alightPersonId);
    }

    if (boardPersonId > 0) writeMarker(boardPersonId);
    if (alightPersonId > 0 && alightPersonId != boardPersonId) writeMarker(alightPersonId);
    writeSummary();

    LOG_INFO(QString("[PersonFolder] 匹配标记已写: %1(上车) ↔ %2(下车) sim=%3")
                 .arg(personName(boardPersonId), personName(alightPersonId))
                 .arg(similarity, 0, 'f', 4));
}
