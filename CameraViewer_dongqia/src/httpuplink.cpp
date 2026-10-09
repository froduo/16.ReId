#include "httpuplink.h"
#include "logger.h"

#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QNetworkReply>
#include <QTextStream>
#include <QUrl>

using namespace DongqiaFlow;

// ============================================================
// 协议工具函数实现
// ============================================================
namespace DongqiaFlow {

QString matToJpegBase64(const cv::Mat &image, int quality)
{
    if (image.empty()) return QString();

    cv::Mat bgr;
    if (image.channels() == 4) {
        cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
    } else if (image.channels() == 1) {
        cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
    } else {
        bgr = image;
    }

    std::vector<uchar> buf;
    std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, quality};
    if (!cv::imencode(".jpg", bgr, buf, params)) {
        return QString();
    }
    QByteArray bytes(reinterpret_cast<const char *>(buf.data()),
                     static_cast<int>(buf.size()));
    return QString::fromLatin1(bytes.toBase64());
}

QString dwellBucketOfSeconds(int seconds)
{
    // 桶边界固定（分钟，左闭右开）
    if (seconds < 60)    return DwellRange::D0_1;     // 不到 1 分钟
    if (seconds < 300)   return DwellRange::D1_5;     // 1 ~ 5 分钟
    if (seconds < 900)   return DwellRange::D5_15;    // 5 ~ 15 分钟
    if (seconds < 1800)  return DwellRange::D15_30;   // 15 ~ 30 分钟
    return DwellRange::D30Plus;                       // 30 分钟以上
}

} // namespace DongqiaFlow

// ============================================================
// HttpUplink
// ============================================================

HttpUplink::HttpUplink(QObject *parent)
    : QObject(parent)
{
    m_nam = new QNetworkAccessManager(this);
    connect(m_nam, &QNetworkAccessManager::finished,
            this, &HttpUplink::onReplyFinished);

    m_retryTimer = new QTimer(this);
    m_retryTimer->setInterval(m_cfg.retryIntervalMs);
    connect(m_retryTimer, &QTimer::timeout, this, &HttpUplink::onRetryTimer);
}

HttpUplink::~HttpUplink()
{
    try { saveSpool(); } catch (...) {}
}

void HttpUplink::setConfig(const Config &cfg)
{
    m_cfg = cfg;
    if (m_retryTimer) m_retryTimer->setInterval(qMax(500, m_cfg.retryIntervalMs));
}

void HttpUplink::start()
{
    if (m_running) return;
    ensureSpoolDir();
    loadSpool();
    pruneSpool();
    m_running = true;
    m_retryTimer->start();
    LOG_INFO(QString("[HttpUplink] started, baseUrl=%1, pending=%2")
                 .arg(m_cfg.baseUrl).arg(m_queue.size()));
    flush();
}

void HttpUplink::stop()
{
    if (!m_running) return;
    m_running = false;
    m_retryTimer->stop();
    LOG_INFO("[HttpUplink] stopped");
}

// ------------------------------------------------------------
// 入队
// ------------------------------------------------------------

void HttpUplink::enqueueInterval(const IntervalAggregate &agg)
{
    const QString key = agg.idempotentKey();
    const qint64 rev = agg.revision;

    // revision 取大：同一 key 只保留更大的那一件
    if (m_bestRevision.contains(key) && m_bestRevision.value(key) >= rev) {
        m_dropped++;
        LOG_INFO(QString("[HttpUplink] drop stale interval key=%1 rev=%2 <= %3")
                     .arg(key).arg(rev).arg(m_bestRevision.value(key)));
        emitStats();
        return;
    }
    m_bestRevision[key] = rev;

    PendingMsg msg;
    msg.kind = QStringLiteral("interval");
    msg.path = m_cfg.intervalPath;
    msg.body = buildIntervalJson(agg);
    msg.key = key;
    msg.revision = rev;
    enqueueInternal(msg, /*dedupByKey=*/true);
}

void HttpUplink::enqueuePersonEvent(const PersonFlowEvent &evt)
{
    const QString key = evt.idempotentKey();
    const qint64 rev = evt.eventTsS;

    if (m_bestRevision.contains(key)) {
        // 单人单次事件天然幂等，重复直接丢弃
        m_dropped++;
        emitStats();
        return;
    }
    m_bestRevision[key] = rev;

    PendingMsg msg;
    msg.kind = QStringLiteral("person");
    msg.path = m_cfg.personPath;
    msg.body = buildPersonJson(evt);
    msg.key = key;
    msg.revision = rev;
    enqueueInternal(msg, /*dedupByKey=*/true);
}

void HttpUplink::enqueueInternal(PendingMsg msg, bool dedupByKey)
{
    if (!msg.body.isEmpty()) {
        msg.revision = (msg.body.contains(QStringLiteral("revision")))
                           ? static_cast<qint64>(msg.body.value("revision").toDouble())
                           : msg.revision;
    }
    msg.enqueueMs = QDateTime::currentMSecsSinceEpoch();

    if (dedupByKey) {
        // 等待重试期间同一 key 被修正 → 用新版本覆盖待发件（不排队两版）
        for (int i = 0; i < m_queue.size(); ++i) {
            if (m_queue[i].key == msg.key) {
                m_queue[i] = msg;
                saveSpool();
                emitStats();
                flush();
                return;
            }
        }
    }

    m_queue.append(msg);
    saveSpool();
    emitStats();
    flush();
}

void HttpUplink::flush()
{
    if (!m_running) return;
    sendNext();
}

// ------------------------------------------------------------
// 发送
// ------------------------------------------------------------

QNetworkRequest HttpUplink::makeRequest(const QString &path) const
{
    QUrl url(m_cfg.baseUrl);
    QString base = url.toString();
    while (base.endsWith('/')) base.chop(1);
    QString p = path;
    if (!p.startsWith('/')) p.prepend('/');
    url = QUrl(base + p);

    QNetworkRequest req(url);
    req.setHeader(QNetworkRequest::ContentTypeHeader,
                  QStringLiteral("application/json"));
    req.setRawHeader("Accept", "application/json");
    req.setRawHeader("User-Agent", "DongqiaFlowUplink/1.0 (Qt)");

    if (m_cfg.useBasicAuth) {
        QByteArray creds = (m_cfg.username + ":" + m_cfg.password).toUtf8().toBase64();
        req.setRawHeader("Authorization", "Basic " + creds);
    }

    // 允许安全重定向（网关前缀适配）
    req.setAttribute(QNetworkRequest::RedirectPolicyAttribute,
                     QNetworkRequest::NoLessSafeRedirectPolicy);
    return req;
}

void HttpUplink::sendNext()
{
    if (m_busy || m_queue.isEmpty() || !m_running) return;

    const PendingMsg &msg = m_queue.first();
    QJsonDocument doc(msg.body);
    QByteArray payload = doc.toJson(QJsonDocument::Compact);

    QNetworkRequest req = makeRequest(msg.path);
    m_busy = true;
    LOG_INFO(QString("[HttpUplink] POST %1 key=%2 bytes=%3 attempt=%4")
                 .arg(req.url().toString(), msg.key)
                 .arg(payload.size()).arg(msg.attempts + 1));

    QNetworkReply *reply = m_nam->post(req, payload);
    if (m_cfg.verifyTls) {
        reply->setProperty("verifyTls", true);
    }
    reply->setProperty("uplinkKey", msg.key);
}

bool HttpUplink::handleSuccess(QNetworkReply *reply, QByteArray &bodyOut, QString &errOut) const
{
    const int status = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
    bodyOut = reply->readAll();

    if (reply->error() != QNetworkReply::NoError) {
        errOut = QStringLiteral("network error: %1").arg(reply->errorString());
        return false;
    }
    if (status < 200 || status >= 300) {
        errOut = QStringLiteral("http status %1").arg(status);
        return false;
    }
    // 协议要求：2xx 且响应体非空（哪怕只是 {}）
    if (bodyOut.trimmed().isEmpty()) {
        errOut = QStringLiteral("2xx but empty body (treated as failure per protocol)");
        return false;
    }
    return true;
}

void HttpUplink::onReplyFinished(QNetworkReply *reply)
{
    reply->deleteLater();
    m_busy = false;

    if (!m_running) return;
    if (m_queue.isEmpty()) return;

    const QString key = reply->property("uplinkKey").toString();
    QByteArray body;
    QString err;
    const bool ok = handleSuccess(reply, body, err);

    // 防止应答与当前队首错位（正常情况下一致）
    if (!m_queue.isEmpty() && key != m_queue.first().key) {
        LOG_WARN(QString("[HttpUplink] reply key mismatch: reply=%1 head=%2")
                     .arg(key, m_queue.first().key));
    }

    if (ok) {
        m_sent++;
        PendingMsg done = m_queue.takeFirst();
        LOG_INFO(QString("[HttpUplink] OK key=%1 resp=%2")
                     .arg(done.key, QString::fromUtf8(body.left(120))));
        emit messageSent(done.kind, true, QString::fromUtf8(body.left(200)));
        saveSpool();
        emitStats();
        // 一件一件发：立即尝试下一件
        sendNext();
    } else {
        m_failed++;
        if (!m_queue.isEmpty()) {
            m_queue.first().attempts++;
            LOG_WARN(QString("[HttpUplink] FAIL key=%1 attempt=%2 reason=%3")
                         .arg(m_queue.first().key)
                         .arg(m_queue.first().attempts)
                         .arg(err));
        }
        emit messageSent(m_queue.isEmpty() ? QString() : m_queue.first().kind,
                         false, err);
        emitStats();
        // 失败不立即重发，交给固定间隔重试定时器
    }
}

void HttpUplink::onRetryTimer()
{
    if (!m_running || m_busy || m_queue.isEmpty()) return;
    sendNext();
}

void HttpUplink::emitStats()
{
    emit statsChanged(m_queue.size(), m_sent, m_failed);
}

// ------------------------------------------------------------
// JSON 构建
// ------------------------------------------------------------

static QJsonArray bucketRowsToJson(const QVector<BucketRow> &rows, const QString &dimKey)
{
    QJsonArray arr;
    for (const auto &r : rows) {
        if (r.count <= 0) continue; // 只出现非零的行
        QJsonObject o;
        o.insert(dimKey, r.value);
        o.insert(QStringLiteral("count"), r.count);
        arr.append(o);
    }
    return arr;
}

static QJsonArray ageGenderRowsToJson(const QVector<AgeGenderRow> &rows)
{
    QJsonArray arr;
    for (const auto &r : rows) {
        if (r.count <= 0) continue;
        QJsonObject o;
        o.insert(QStringLiteral("age_range"), r.ageRange);
        o.insert(QStringLiteral("gender"), r.gender);
        o.insert(QStringLiteral("count"), r.count);
        arr.append(o);
    }
    return arr;
}

QJsonObject HttpUplink::buildIntervalJson(const IntervalAggregate &agg) const
{
    QJsonObject root;
    root.insert(QStringLiteral("schema_version"), agg.schemaVersion);
    root.insert(QStringLiteral("type"), agg.type);
    root.insert(QStringLiteral("master_sn"), agg.masterSn);

    QJsonArray sns;
    for (const QString &s : agg.deviceSns) sns.append(s);
    root.insert(QStringLiteral("device_sns"), sns);

    QJsonObject tz;
    tz.insert(QStringLiteral("name"), agg.tzName);
    tz.insert(QStringLiteral("utc_offset_seconds"), agg.utcOffsetSeconds);
    tz.insert(QStringLiteral("is_dst"), agg.isDst);
    root.insert(QStringLiteral("timezone"), tz);

    root.insert(QStringLiteral("batch_date"), agg.batchDate);
    root.insert(QStringLiteral("stat_basis"), agg.statBasis);
    root.insert(QStringLiteral("revision"), static_cast<double>(agg.revision));
    root.insert(QStringLiteral("interval_s"), agg.intervalS);
    root.insert(QStringLiteral("start_ts_s"), static_cast<double>(agg.startTsS));
    root.insert(QStringLiteral("end_ts_s"), static_cast<double>(agg.endTsS));

    // event_counts
    QJsonObject ec;
    ec.insert(QStringLiteral("enter"), agg.eventCounts.enter);
    ec.insert(QStringLiteral("leave"), agg.eventCounts.leave);
    ec.insert(QStringLiteral("pass_by"), agg.eventCounts.passBy);
    ec.insert(QStringLiteral("turn_back"), agg.eventCounts.turnBack);
    root.insert(QStringLiteral("event_counts"), ec);

    // raw_stats
    {
        QJsonObject bd;
        bd.insert(QStringLiteral("height_category"),
                  bucketRowsToJson(agg.rawHeight, QStringLiteral("height_category")));
        bd.insert(QStringLiteral("age_range|gender"),
                  ageGenderRowsToJson(agg.rawAgeGender));
        QJsonObject raw;
        raw.insert(QStringLiteral("count"), agg.rawCount);
        raw.insert(QStringLiteral("breakdowns"), bd);
        root.insert(QStringLiteral("raw_stats"), raw);
    }

    // final_stats
    {
        QJsonObject bd;
        bd.insert(QStringLiteral("height_category"),
                  bucketRowsToJson(agg.finalHeight, QStringLiteral("height_category")));
        bd.insert(QStringLiteral("age_range|gender"),
                  ageGenderRowsToJson(agg.finalAgeGender));
        QJsonObject fin;
        fin.insert(QStringLiteral("count"), agg.finalCount);
        fin.insert(QStringLiteral("repeat"), agg.finalRepeat);
        fin.insert(QStringLiteral("breakdowns"), bd);
        root.insert(QStringLiteral("final_stats"), fin);
    }

    // non_customer
    {
        QJsonObject bd;
        bd.insert(QStringLiteral("person_type"),
                  bucketRowsToJson(agg.nonCustomerPersonType, QStringLiteral("person_type")));
        QJsonObject nc;
        nc.insert(QStringLiteral("count"), agg.nonCustomerCount);
        nc.insert(QStringLiteral("breakdowns"), bd);
        root.insert(QStringLiteral("non_customer"), nc);
    }

    // dwell_stats
    {
        QJsonObject bd;
        bd.insert(QStringLiteral("dwell_range"),
                  bucketRowsToJson(agg.dwellRange, QStringLiteral("dwell_range")));
        QJsonObject dw;
        dw.insert(QStringLiteral("count"), agg.dwellCount);
        dw.insert(QStringLiteral("breakdowns"), bd);
        root.insert(QStringLiteral("dwell_stats"), dw);
    }

    return root;
}

QJsonObject HttpUplink::buildPersonJson(const PersonFlowEvent &evt) const
{
    QJsonObject root;
    root.insert(QStringLiteral("schema_version"), QStringLiteral("1.0"));
    root.insert(QStringLiteral("type"), QStringLiteral("person_flow_event"));
    root.insert(QStringLiteral("master_sn"), evt.masterSn);
    root.insert(QStringLiteral("batch_date"), evt.batchDate);
    root.insert(QStringLiteral("stat_basis"), evt.statBasis);
    root.insert(QStringLiteral("revision"), static_cast<double>(evt.eventTsS));

    QJsonObject tz;
    tz.insert(QStringLiteral("name"), QStringLiteral("Asia/Shanghai"));
    tz.insert(QStringLiteral("utc_offset_seconds"), 28800);
    tz.insert(QStringLiteral("is_dst"), false);
    root.insert(QStringLiteral("timezone"), tz);

    QJsonObject e;
    e.insert(QStringLiteral("visitor_id"), evt.visitorId);
    e.insert(QStringLiteral("person_type"), personKindToProtocol(evt.kind));
    e.insert(QStringLiteral("event_type"), evt.eventType);
    e.insert(QStringLiteral("direction"), evt.direction);
    e.insert(QStringLiteral("enter_ts_s"), static_cast<double>(evt.enterTsS));
    e.insert(QStringLiteral("exit_ts_s"), static_cast<double>(evt.exitTsS));
    e.insert(QStringLiteral("event_ts_s"), static_cast<double>(evt.eventTsS));
    e.insert(QStringLiteral("dwell_seconds"), evt.dwellSeconds);
    e.insert(QStringLiteral("reid_similarity"),
             evt.reidSimilarity < 0 ? QJsonValue(QJsonValue::Null)
                                    : QJsonValue(static_cast<double>(evt.reidSimilarity)));
    e.insert(QStringLiteral("region_id"), evt.regionId);
    if (!evt.snapshotBase64.isEmpty())
        e.insert(QStringLiteral("snapshot_jpeg_base64"), evt.snapshotBase64);

    root.insert(QStringLiteral("event"), e);
    return root;
}

// ------------------------------------------------------------
// 本地积压持久化（JSONL，最多保留 maxOfflineDays 天）
// ------------------------------------------------------------

QString HttpUplink::spoolFilePath() const
{
    return m_cfg.spoolDir + QStringLiteral("/uplink_pending.jsonl");
}

void HttpUplink::ensureSpoolDir() const
{
    QDir d(m_cfg.spoolDir);
    if (!d.exists()) d.mkpath(QStringLiteral("."));
}

void HttpUplink::saveSpool() const
{
    ensureSpoolDir();
    QFile f(spoolFilePath());
    if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text)) {
        return;
    }
    QTextStream out(&f);
    for (const PendingMsg &m : m_queue) {
        QJsonObject o;
        o.insert(QStringLiteral("kind"), m.kind);
        o.insert(QStringLiteral("path"), m.path);
        o.insert(QStringLiteral("key"), m.key);
        o.insert(QStringLiteral("revision"), static_cast<double>(m.revision));
        o.insert(QStringLiteral("enqueueMs"), static_cast<double>(m.enqueueMs));
        o.insert(QStringLiteral("attempts"), m.attempts);
        o.insert(QStringLiteral("body"),
                 QString::fromUtf8(QJsonDocument(m.body).toJson(QJsonDocument::Compact)));
        out << QString::fromUtf8(QJsonDocument(o).toJson(QJsonDocument::Compact)) << "\n";
    }
    f.close();
}

void HttpUplink::loadSpool()
{
    QFile f(spoolFilePath());
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) return;

    QTextStream in(&f);
    while (!in.atEnd()) {
        const QString line = in.readLine().trimmed();
        if (line.isEmpty()) continue;
        QJsonObject o = QJsonDocument::fromJson(line.toUtf8()).object();
        if (o.isEmpty()) continue;

        PendingMsg m;
        m.kind = o.value("kind").toString();
        m.path = o.value("path").toString();
        m.key = o.value("key").toString();
        m.revision = static_cast<qint64>(o.value("revision").toDouble());
        m.enqueueMs = static_cast<qint64>(o.value("enqueueMs").toDouble());
        m.attempts = o.value("attempts").toInt();
        m.body = QJsonDocument::fromJson(o.value("body").toString().toUtf8()).object();

        if (m.key.isEmpty() || m.body.isEmpty()) continue;

        // 恢复 revision 表（取大）
        if (!m_bestRevision.contains(m.key) || m_bestRevision.value(m.key) < m.revision)
            m_bestRevision[m.key] = m.revision;

        m_queue.append(m);
    }
    f.close();
    LOG_INFO(QString("[HttpUplink] loaded %1 pending messages from spool").arg(m_queue.size()));
    emitStats();
}

void HttpUplink::pruneSpool()
{
    const qint64 limitMs = QDateTime::currentMSecsSinceEpoch()
                           - static_cast<qint64>(m_cfg.maxOfflineDays) * 24 * 3600 * 1000;
    int removed = 0;
    for (int i = m_queue.size() - 1; i >= 0; --i) {
        if (m_queue[i].enqueueMs > 0 && m_queue[i].enqueueMs < limitMs) {
            m_queue.removeAt(i);
            removed++;
        }
    }
    if (removed > 0) {
        LOG_WARN(QString("[HttpUplink] pruned %1 expired pending messages (> %2 days)")
                     .arg(removed).arg(m_cfg.maxOfflineDays));
        saveSpool();
    }
}
