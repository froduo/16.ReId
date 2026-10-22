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
    // 人员进出记录：(乘客ID, 上车时间, 下车时间)
    QJsonObject obj;
    obj["passengerId"] = entry.visitorId;
    obj["boardingTime"] = entry.boardTime;
    obj["alightingTime"] = entry.alightTime;
    obj["isInside"] = entry.isInside;
    obj["regionId"] = entry.regionId;
    obj["currentTrackId"] = entry.currentTrackId;
    obj["reidFeature"] = entry.reidFeatureBase64;
    obj["boardImagePath"] = entry.enterImagePath;
    obj["alightImagePath"] = entry.exitImagePath;
    obj["direction"] = entry.direction;
    // 兼容旧字段（供外部脚本按原 key 读取）
    obj["visitorId"] = entry.visitorId;
    obj["enterTime"] = entry.boardTime;
    obj["exitTime"] = entry.alightTime;
    return obj;
}

VisitorLog::VisitorEntry VisitorLog::jsonToVisitor(const QJsonObject &obj) const
{
    VisitorEntry entry;
    // 新字段优先，回退兼容旧字段（visitorId/enterTime/exitTime）
    entry.visitorId = obj.contains("passengerId") ? obj["passengerId"].toInt()
                                                  : obj["visitorId"].toInt();
    const QString board = obj.contains("boardingTime") ? obj["boardingTime"].toString()
                                                       : obj["enterTime"].toString();
    const QString alight = obj.contains("alightingTime") ? obj["alightingTime"].toString()
                                                         : obj["exitTime"].toString();
    entry.boardTime = board;
    entry.alightTime = alight;
    entry.isInside = obj["isInside"].toBool();
    entry.regionId = obj["regionId"].toInt();
    entry.currentTrackId = obj["currentTrackId"].toInt(-1);
    entry.reidFeatureBase64 = obj["reidFeature"].toString();
    entry.enterImagePath = obj.contains("boardImagePath") ? obj["boardImagePath"].toString()
                                                          : obj["enterImagePath"].toString();
    entry.exitImagePath = obj.contains("alightImagePath") ? obj["alightImagePath"].toString()
                                                          : obj["exitImagePath"].toString();
    entry.direction = obj["direction"].toString();
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

    // 将仍在车上的乘客重置为"已下车"状态（因为应用重启了）
    int resetCount = 0;
    for (auto it = visitors.begin(); it != visitors.end(); ++it) {
        if (it.value().isInside) {
            it.value().isInside = false;
            it.value().currentTrackId = -1;
            if (it.value().alightTime.isEmpty()) {
                it.value().alightTime = QDateTime::currentDateTime().toString(Qt::ISODate);
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
        const QJsonObject o = val.toObject();
        int id = o.contains("passengerId") ? o["passengerId"].toInt() : o["visitorId"].toInt();
        if (id > maxId) maxId = id;
    }

    return maxId + 1;
}
