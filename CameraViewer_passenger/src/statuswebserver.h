#ifndef STATUSWEBSERVER_H
#define STATUSWEBSERVER_H

#include <QObject>
#include <QJsonObject>
#include <QMutex>
#include <functional>

class QTcpServer;
class QTcpSocket;

/**
 * @brief 极简 HTTP 状态服务：供手机/浏览器查看上下车实时状态 + 页面控制
 *
 * 端点：
 *   GET /            仪表盘页面（自动刷新，含控制按钮）
 *   GET /api/status  返回 JSON 状态
 *   GET /api/frame   返回当前帧 JPEG（MJPEG 单帧，前端定时刷新）
 *   GET /api/videos  返回 passenger_videos 下的视频列表
 *   POST /api/control 前端控制指令 {action: toggleCapture/clearCounts/togglePlay/loadVideo/confirmMatch, ...}
 *
 * 线程模型：对象须存活于带事件循环的线程（主线程即可）；
 * setStatus()/setFrameProvider() 内部加锁，可在任意线程调用。
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
    /// 设置当前帧 JPEG 提供回调（/api/frame 请求时调用，返回 JPEG 字节）
    void setFrameProvider(std::function<QByteArray()> provider);
    /// 设置视频目录（/api/videos 扫描该目录下的 .mp4）
    void setVideoDir(const QString &dir);
    /// 设置 person 乘客图片目录提供回调（/api/person_images 用，返回某 personId 的乘客图片目录）
    void setPersonImageProvider(std::function<QString(int personId)> provider);

signals:
    /// 前端页面控制指令（action: toggleCapture / clearCounts / togglePlay / loadVideo / confirmMatch）
    void controlRequested(const QString &action, const QJsonObject &params);

private slots:
    void onNewConnection();

private:
    void handleRequest(QTcpSocket *sock, const QByteArray &request);

    QTcpServer *m_server{nullptr};
    QMutex m_mutex;
    QJsonObject m_status;
    std::function<QByteArray()> m_frameProvider;
    QString m_videoDir;
    std::function<QString(int)> m_personImageProvider;
    quint16 m_port{0};
    bool m_running{false};
};

#endif // STATUSWEBSERVER_H
