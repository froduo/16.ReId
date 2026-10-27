#ifndef CAMERASEARCHER_H
#define CAMERASEARCHER_H

#include <QObject>
#include <QString>
#include <QStringList>
#include <QVector>
#include <QSet>
#include <QByteArray>
#include <QHostAddress>
#include <atomic>
#include <thread>

/**
 * @brief 相机设备搜索器
 *
 * 在局域网内搜索相机设备，支持两种互补的搜索策略：
 *   1. ACP 协议 UDP 广播搜索：向控制端口（默认 8091）发送广播探测报文，
 *      等待设备响应（ACP 为设备私有协议，报文格式见 camerasearcher.cpp 注释，
 *      可按实际设备协议调整）。
 *   2. 网段 TCP 扫描：并发探测局域网内主机的视频端口（默认 554/8554），
 *      发现开放视频端口的设备。
 *
 * 搜索在后台线程执行，通过信号把结果送回 UI 线程，避免阻塞界面。
 */
class CameraSearcher : public QObject
{
    Q_OBJECT

public:
    /// 搜索到的相机设备信息
    struct Device {
        QString name;                          // 名称
        QString ip;                            // IP/域名
        QString group    = QStringLiteral("默认分组");
        QString deviceType = QStringLiteral("IPC");
        QString protocol = QStringLiteral("ACP");
        int videoPort    = 554;                // 视频端口
        int controlPort  = 8091;               // 控制端口
        QString user     = QStringLiteral("admin");
        QString pass     = QStringLiteral("123456");
        QString rtspPath = QStringLiteral("/live/camera"); // 验证成功的 RTSP 路径
        QString rtspUrl;                       // 完整 RTSP URL
        bool verified    = false;              // 是否已通过 RTSP 验证
        QString resolution;                    // 验证通过后的分辨率描述（如 1920x1080）

        bool isValid() const { return !ip.isEmpty(); }
        QString displayName() const {
            return name.isEmpty() ? ip : name;
        }
    };

    explicit CameraSearcher(QObject *parent = nullptr);
    ~CameraSearcher() override;

    /**
     * @brief 启动搜索
     * @param subnet 网段前缀（如 "192.168.200"），为空则自动探测本机所有网段
     * @param videoPorts 需要探测的视频端口
     * @param user 默认登录账号
     * @param pass 默认登录密码
     * @param rtspPaths 尝试的 RTSP 路径列表（按顺序验证）
     * @param controlPort ACP/UDP 广播搜索端口
     * @param timeoutMs RTSP 验证超时（毫秒）
     */
    void startSearch(const QString &subnet,
                     const QVector<int> &videoPorts,
                     const QString &user,
                     const QString &pass,
                     const QStringList &rtspPaths,
                     int controlPort = 8091,
                     int timeoutMs = 2500);

    /// 取消搜索
    void cancel();
    bool isSearching() const { return m_searching.load(); }

signals:
    void deviceFound(const CameraSearcher::Device &device);
    void searchProgress(int current, int total, const QString &ip);
    void searchFinished(int totalFound);

private:
    void doSearch();
    QStringList localSubnets() const;
    QStringList enumerateIps(const QString &subnet) const;
    bool probeTcpPort(const QString &ip, int port, int timeoutMs) const;
    QString verifyRtsp(const QString &ip, int port, const QString &user,
                       const QString &pass, const QStringList &paths,
                       int timeoutMs, int *width, int *height) const;
    QVector<Device> udpProbe(int controlPort, int timeoutMs);
    bool verifyDevice(Device &device, int onlyPort = 0) const;
    QVector<Device> onvifProbe(int timeoutMs);
    void parseOnvifResponse(const QByteArray &data, const QHostAddress &from,
                            QVector<Device> &found, QSet<QString> &seenIps);

    std::atomic<bool> m_cancel{false};
    std::atomic<bool> m_searching{false};
    std::thread m_thread;

    QString m_subnet;
    QVector<int> m_videoPorts;
    QString m_user;
    QString m_pass;
    QStringList m_rtspPaths;
    int m_controlPort = 8091;
    int m_timeoutMs = 2500;
};

Q_DECLARE_METATYPE(CameraSearcher::Device)

#endif // CAMERASEARCHER_H
