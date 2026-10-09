#ifndef HTTPUPLINK_H
#define HTTPUPLINK_H

/**
 * @file httpuplink.h
 * @brief 客流时段聚合上行协议客户端（设备 → footfall.dongqia.com）
 *
 * 严格实现《动恰3DV6 AI HTTP 新对接协议》中「服务端需要遵守」的部分在本端的对应面：
 *  - POST application/json 到 /api/passenger-flow/interval-aggregate
 *  - 仅当响应为 2xx 且响应体非空才算成功，否则重发（固定间隔）
 *  - 一件一件发：前一件收到确认后才发下一件
 *  - 每个时段只发最新版：等待重试期间若该时段被修正，用新版本覆盖待发件
 *  - 幂等键 (master_sn, batch_date, start_ts_s) + revision 取大
 *  - 断网积压本地留最多 2 天，恢复后先补发积压再收敛
 *
 * 另含「精准识别某人上下车」的单人事件上行端点（person-event）。
 */

#include <QObject>
#include <QString>
#include <QStringList>
#include <QVector>
#include <QMap>
#include <QJsonObject>
#include <QQueue>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QTimer>

#include "dongqia_protocol.h"

class HttpUplink : public QObject
{
    Q_OBJECT
public:
    struct Config {
        QString baseUrl{QStringLiteral("https://footfall.dongqia.com")};
        QString intervalPath{QStringLiteral("/api/passenger-flow/interval-aggregate")};
        QString personPath{QStringLiteral("/api/passenger-flow/person-event")};
        QString username{QStringLiteral("dongqia01")};
        QString password{QStringLiteral("123456")};
        QString masterSn{QStringLiteral("CAM-A01")};
        QStringList deviceSns;
        QString spoolDir{QStringLiteral("./uplink_spool")};
        int retryIntervalMs{5000};       // 失败重试固定间隔
        int maxOfflineDays{2};           // 设备侧最多留 2 天（对齐协议 4.3）
        bool useBasicAuth{true};
        bool enabled{true};
        bool verifyTls{true};
    };

    explicit HttpUplink(QObject *parent = nullptr);
    ~HttpUplink() override;

    void setConfig(const Config &cfg);
    Config config() const { return m_cfg; }

    /// 启动（加载本地积压并开始重试循环）
    void start();
    /// 停止（仅停止发送，积压保留在内存/磁盘）
    void stop();
    bool isRunning() const { return m_running; }

    /// 入队一件「时段聚合」报文（内部按 revision 取大 + 幂等覆盖）
    void enqueueInterval(const DongqiaFlow::IntervalAggregate &agg);

    /// 入队一件「单人上下车」事件报文
    void enqueuePersonEvent(const DongqiaFlow::PersonFlowEvent &evt);

    /// 立即尝试发送（异步，不阻塞）
    void flush();

    int pendingCount() const { return m_queue.size(); }
    qint64 sentCount() const { return m_sent; }
    qint64 failedCount() const { return m_failed; }
    qint64 droppedCount() const { return m_dropped; }

    /// 构建协议 JSON（公开以便单元测试 / 抓包核对）
    QJsonObject buildIntervalJson(const DongqiaFlow::IntervalAggregate &agg) const;
    QJsonObject buildPersonJson(const DongqiaFlow::PersonFlowEvent &evt) const;

signals:
    void messageSent(const QString &kind, bool ok, const QString &detail);
    void statsChanged(int pending, qint64 sent, qint64 failed);
    void logMessage(const QString &level, const QString &msg);

private slots:
    void onRetryTimer();
    void onReplyFinished(QNetworkReply *reply);

private:
    struct PendingMsg {
        QString kind;        // "interval" / "person"
        QString path;
        QJsonObject body;
        QString key;         // 幂等键
        qint64 revision{0};  // interval 用；person 用 eventTsS
        qint64 enqueueMs{0}; // 入队时间（用于离线清理）
        int attempts{0};
    };

    void enqueueInternal(PendingMsg msg, bool dedupByKey);
    void sendNext();
    QNetworkRequest makeRequest(const QString &path) const;
    bool handleSuccess(QNetworkReply *reply, QByteArray &bodyOut, QString &errOut) const;

    // 本地积压持久化（JSONL）
    QString spoolFilePath() const;
    void ensureSpoolDir() const;
    void loadSpool();
    void saveSpool() const;
    void pruneSpool();

    Config m_cfg;
    QNetworkAccessManager *m_nam{nullptr};
    QTimer *m_retryTimer{nullptr};

    QVector<PendingMsg> m_queue;      // 待发送（按入队顺序）
    QMap<QString, qint64> m_bestRevision; // 幂等键 -> 已受理的最大 revision

    bool m_running{false};
    bool m_busy{false};

    qint64 m_sent{0};
    qint64 m_failed{0};
    qint64 m_dropped{0};

    void emitStats();
};

#endif // HTTPUPLINK_H
