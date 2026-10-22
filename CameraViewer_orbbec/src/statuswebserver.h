#ifndef STATUSWEBSERVER_H
#define STATUSWEBSERVER_H

#include <QObject>
#include <QJsonObject>
#include <QMutex>

class QTcpServer;
class QTcpSocket;

/**
 * @brief 极简 HTTP 状态服务：供手机/浏览器查看上下车实时状态
 *
 * 端点：
 *   GET /            简单网页（自动刷新，移动端友好）
 *   GET /api/status  返回 JSON 状态
 *
 * 线程模型：对象须存活于带事件循环的线程（主线程即可）；
 * setStatus() 内部加锁，可在任意线程调用。
 */
class StatusWebServer : public QObject
{
    Q_OBJECT
public:
    explicit StatusWebServer(QObject *parent = nullptr);
    ~StatusWebServer();

    bool start(quint16 port);
    void stop();
    bool isRunning() const { return m_running; }
    quint16 port() const { return m_port; }

    /// 更新最新状态快照
    void setStatus(const QJsonObject &status);

private slots:
    void onNewConnection();

private:
    void handleRequest(QTcpSocket *sock, const QByteArray &request);

    QTcpServer *m_server{nullptr};
    QMutex m_mutex;
    QJsonObject m_status;
    quint16 m_port{0};
    bool m_running{false};
};

#endif // STATUSWEBSERVER_H
