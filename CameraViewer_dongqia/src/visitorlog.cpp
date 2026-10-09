#include "visitorlog.h"
#include "logger.h"

VisitorLog::VisitorLog()
{
}

void VisitorLog::setSaveDir(const QString &dir)
{
    m_saveDir = dir;
    ensureDirExists(m_saveDir);
    m_visitorJsonPath = m_saveDir + "/visitor_records.json";
    m_eventLogPath = m_saveDir + "/visitor_events.csv";
    m_personRecordPath = m_saveDir + "/person_records.csv";
}

void VisitorLog::ensureDirExists(const QString &dir)
{
    QDir d(dir);
    if (!d.exists()) {
        d.mkpath(".");
    }
}

QString VisitorLog::matToBase64(const cv::Mat &mat)
{
    if (mat.empty()) return QString();

    // 将 Mat 序列化为字节数组
    std::vector<uchar> buf;
    cv::Mat contMat = mat.isContinuous() ? mat : mat.clone();
    buf.assign(contMat.datastart, contMat.dataend);

    QByteArray byteArray(reinterpret_cast<const char*>(buf.data()), static_cast<int>(buf.size()));

    // 在 Base64 前面附加维度信息以便恢复
    QString meta = QString("%1,%2,%3,%4,")
                       .arg(contMat.rows)
                       .arg(contMat.cols)
                       .arg(contMat.type())
                       .arg(contMat.channels());
    QByteArray metaBytes = meta.toUtf8();
    QByteArray combined = metaBytes + byteArray.toBase64();

    return QString::fromUtf8(combined);
}

cv::Mat VisitorLog::base64ToMat(const QString &base64)
{
    if (base64.isEmpty()) return cv::Mat();

    QByteArray combined = base64.toUtf8();

    // 找到元数据结束位置（第4个逗号之后）
    int metaEnd = -1;
    int commaCount = 0;
    for (int i = 0; i < combined.size(); ++i) {
        if (combined[i] == ',') {
            commaCount++;
            if (commaCount == 4) {
                metaEnd = i + 1;
                break;
            }
        }
    }

    if (metaEnd < 0) return cv::Mat();

    QByteArray metaBytes = combined.left(metaEnd);
    QByteArray base64Data = combined.mid(metaEnd);

    // 解析元数据
    QList<QByteArray> parts = metaBytes.split(',');
    if (parts.size() < 4) return cv::Mat();

    int rows = parts[0].toInt();
    int cols = parts[1].toInt();
    int type = parts[2].toInt();

    // 解码 Base64
    QByteArray rawData = QByteArray::fromBase64(base64Data);
    if (rawData.isEmpty()) return cv::Mat();

    // 恢复 Mat
    cv::Mat mat(rows, cols, type, const_cast<char*>(rawData.data()));
    return mat.clone();
}

QJsonObject VisitorLog::visitorToJson(const VisitorEntry &entry) const
{
    QJsonObject obj;
    obj["visitorId"] = entry.visitorId;
    obj["enterTime"] = entry.enterTime;
    obj["exitTime"] = entry.exitTime;
    obj["isInside"] = entry.isInside;
    obj["regionId"] = entry.regionId;
    obj["currentTrackId"] = entry.currentTrackId;
    obj["reidFeature"] = entry.reidFeatureBase64;
    obj["enterImagePath"] = entry.enterImagePath;
    obj["exitImagePath"] = entry.exitImagePath;
    obj["direction"] = entry.direction;
    obj["enterDirection"] = entry.enterDirection;
    obj["exitDirection"] = entry.exitDirection;
    obj["entrySide"] = entry.entrySide;
    obj["exitSide"] = entry.exitSide;
    obj["dwellSeconds"] = static_cast<double>(entry.dwellSeconds);
    return obj;
}

VisitorLog::VisitorEntry VisitorLog::jsonToVisitor(const QJsonObject &obj) const
{
    VisitorEntry entry;
    entry.visitorId = obj["visitorId"].toInt();
    entry.enterTime = obj["enterTime"].toString();
    entry.exitTime = obj["exitTime"].toString();
    entry.isInside = obj["isInside"].toBool();
    entry.regionId = obj["regionId"].toInt();
    entry.currentTrackId = obj["currentTrackId"].toInt(-1);
    entry.reidFeatureBase64 = obj["reidFeature"].toString();
    entry.enterImagePath = obj["enterImagePath"].toString();
    entry.exitImagePath = obj["exitImagePath"].toString();
    entry.direction = obj["direction"].toString();
    entry.enterDirection = obj["enterDirection"].toString();
    entry.exitDirection = obj["exitDirection"].toString();
    entry.entrySide = obj["entrySide"].toInt(-1);
    entry.exitSide = obj["exitSide"].toInt(-1);
    entry.dwellSeconds = static_cast<qint64>(obj["dwellSeconds"].toDouble(0));
    return entry;
}

void VisitorLog::saveVisitors(const QMap<int, VisitorEntry> &visitors)
{
    QJsonArray visitorsArray;
    for (auto it = visitors.constBegin(); it != visitors.constEnd(); ++it) {
        visitorsArray.append(visitorToJson(it.value()));
    }

    QJsonObject root;
    root["version"] = 1;
    root["savedAt"] = QDateTime::currentDateTime().toString(Qt::ISODate);
    root["count"] = visitors.size();
    root["visitors"] = visitorsArray;

    QJsonDocument doc(root);
    QFile file(m_visitorJsonPath);
    if (file.open(QIODevice::WriteOnly)) {
        file.write(doc.toJson(QJsonDocument::Indented));
        file.close();
        LOG_INFO(QString("[VisitorLog] Saved %1 visitor records to %2")
                     .arg(visitors.size()).arg(m_visitorJsonPath));
    } else {
        LOG_ERR("[VisitorLog] Failed to save visitor records to " + m_visitorJsonPath);
    }
}

QMap<int, VisitorLog::VisitorEntry> VisitorLog::loadVisitors()
{
    QMap<int, VisitorEntry> visitors;

    QFile file(m_visitorJsonPath);
    if (!file.open(QIODevice::ReadOnly)) {
        LOG_INFO("[VisitorLog] No visitor records file found: " + m_visitorJsonPath);
        return visitors;
    }

    QByteArray data = file.readAll();
    file.close();

    QJsonDocument doc = QJsonDocument::fromJson(data);
    if (!doc.isObject()) {
        LOG_WARN("[VisitorLog] Invalid visitor records file format");
        return visitors;
    }

    QJsonObject root = doc.object();
    int version = root["version"].toInt(0);
    if (version < 1) {
        LOG_WARN("[VisitorLog] Unsupported visitor records version: " + QString::number(version));
        return visitors;
    }

    QJsonArray visitorsArray = root["visitors"].toArray();
    for (const auto &val : visitorsArray) {
        QJsonObject obj = val.toObject();
        VisitorEntry entry = jsonToVisitor(obj);
        visitors[entry.visitorId] = entry;
    }

    // 将仍在区域内的访客重置为"已离开"状态（因为应用重启了）
    int resetCount = 0;
    for (auto it = visitors.begin(); it != visitors.end(); ++it) {
        if (it.value().isInside) {
            it.value().isInside = false;
            it.value().currentTrackId = -1;
            if (it.value().exitTime.isEmpty()) {
                it.value().exitTime = QDateTime::currentDateTime().toString(Qt::ISODate);
            }
            resetCount++;
        }
    }

    if (resetCount > 0) {
        LOG_INFO(QString("[VisitorLog] Reset %1 in-region visitors to exited state (app restarted)")
                     .arg(resetCount));
    }

    LOG_INFO(QString("[VisitorLog] Loaded %1 visitor records from %2 (session saved at: %3)")
                 .arg(visitors.size())
                 .arg(m_visitorJsonPath)
                 .arg(root["savedAt"].toString()));
    return visitors;
}

void VisitorLog::logEvent(const EventEntry &entry)
{
    ensureDirExists(m_saveDir);

    // 检查文件是否存在，不存在则创建并写入表头
    bool needHeader = !QFile::exists(m_eventLogPath);
    QFile file(m_eventLogPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text)) {
        LOG_ERR("[VisitorLog] Failed to open event log: " + m_eventLogPath);
        return;
    }

    QTextStream out(&file);
    if (needHeader) {
        out << "timestamp,event_type,visitor_id,track_id,region_id,similarity,details\n";
    }

    out << entry.timestamp.toString(Qt::ISODate) << ","
        << entry.eventType << ","
        << entry.visitorId << ","
        << entry.trackId << ","
        << entry.regionId << ","
        << QString::number(entry.similarity, 'f', 3) << ","
        << "\"" << QString(entry.details).replace("\"", "\"\"") << "\"\n";

    file.close();
}

void VisitorLog::logPersonRecord(const PersonRecordEntry &entry)
{
    ensureDirExists(m_saveDir);

    const bool needHeader = !QFile::exists(m_personRecordPath);
    QFile file(m_personRecordPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text)) {
        LOG_ERR("[VisitorLog] Failed to open person record log: " + m_personRecordPath);
        return;
    }

    QTextStream out(&file);
    if (needHeader) {
        out << "visitor_id,enter_time,enter_direction,enter_side,"
               "exit_time,exit_direction,exit_side,dwell_seconds,region_id\n";
    }

    auto sideLabel = [](int side) -> QString {
        if (side == 0) return QStringLiteral("A");
        if (side == 1) return QStringLiteral("B");
        return QStringLiteral("-");
    };

    out << entry.visitorId << ","
        << entry.enterTime << ","
        << entry.enterDirection << ","
        << sideLabel(entry.entrySide) << ","
        << entry.exitTime << ","
        << entry.exitDirection << ","
        << sideLabel(entry.exitSide) << ","
        << entry.dwellSeconds << ","
        << entry.regionId << "\n";

    file.close();

    LOG_INFO(QString("[VisitorLog] person record V%1 %2(%3) -> %4(%5), dwell=%6s")
                 .arg(entry.visitorId)
                 .arg(entry.enterDirection).arg(sideLabel(entry.entrySide))
                 .arg(entry.exitDirection).arg(sideLabel(entry.exitSide))
                 .arg(entry.dwellSeconds));
}

void VisitorLog::logSessionStart()
{
    EventEntry entry;
    entry.timestamp = QDateTime::currentDateTime();
    entry.eventType = "SESSION_START";
    entry.visitorId = 0;
    entry.trackId = 0;
    entry.regionId = 0;
    entry.similarity = 0.0f;
    entry.details = "Application started";
    logEvent(entry);
}

void VisitorLog::logSessionEnd()
{
    EventEntry entry;
    entry.timestamp = QDateTime::currentDateTime();
    entry.eventType = "SESSION_END";
    entry.visitorId = 0;
    entry.trackId = 0;
    entry.regionId = 0;
    entry.similarity = 0.0f;
    entry.details = "Application closed";
    logEvent(entry);
}

int VisitorLog::getNextVisitorId() const
{
    // 通过加载现有记录确定下一个可用 ID
    QFile file(m_visitorJsonPath);
    if (!file.open(QIODevice::ReadOnly)) {
        return 1;
    }

    QByteArray data = file.readAll();
    file.close();

    QJsonDocument doc = QJsonDocument::fromJson(data);
    if (!doc.isObject()) return 1;

    QJsonArray visitorsArray = doc.object()["visitors"].toArray();
    int maxId = 0;
    for (const auto &val : visitorsArray) {
        int id = val.toObject()["visitorId"].toInt();
        if (id > maxId) maxId = id;
    }

    return maxId + 1;
}
