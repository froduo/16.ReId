#include "camerasearcher.h"
#include "logger.h"

#include <QNetworkInterface>
#include <QHostAddress>
#include <QTcpSocket>
#include <QThreadPool>
#include <QSet>
#include <QtConcurrent/QtConcurrent>

#include <opencv2/opencv.hpp>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <chrono>
#include <functional>

CameraSearcher::CameraSearcher(QObject *parent)
    : QObject(parent)
{
    qRegisterMetaType<CameraSearcher::Device>("CameraSearcher::Device");
}

CameraSearcher::~CameraSearcher()
{
    cancel();
    if (m_thread.joinable()) {
        m_thread.join();
    }
}

void CameraSearcher::startSearch(const QString &subnet,
                                 const QVector<int> &videoPorts,
                                 const QString &user,
                                 const QString &pass,
                                 const QStringList &rtspPaths,
                                 int controlPort,
                                 int timeoutMs)
{
    if (m_searching.load()) {
        return; // 已在搜索中
    }

    m_subnet = subnet;
    m_videoPorts = videoPorts;
    m_user = user;
    m_pass = pass;
    m_rtspPaths = rtspPaths;
    m_controlPort = controlPort;
    m_timeoutMs = timeoutMs;
    m_cancel.store(false);
    m_searching.store(true);

    if (m_thread.joinable()) {
        m_thread.join();
    }
    m_thread = std::thread(&CameraSearcher::doSearch, this);
}

void CameraSearcher::cancel()
{
    m_cancel.store(true);
}

void CameraSearcher::doSearch()
{
    int foundCount = 0;

    // 1. 枚举需要扫描的 IP 列表
    QStringList ips = enumerateIps(m_subnet);
    int total = ips.size();
    emit searchProgress(0, total, QString());

    // 2. ACP 协议 UDP 广播搜索（控制端口 8091）
    QVector<Device> udpDevices = udpProbe(m_controlPort, 1200);
    for (const Device &d : udpDevices) {
        if (m_cancel.load()) break;
        Device dev = d;
        verifyDevice(dev);
        if (dev.isValid()) {
            emit deviceFound(dev);
            ++foundCount;
        }
    }

    if (m_cancel.load()) {
        m_searching.store(false);
        emit searchFinished(foundCount);
        return;
    }

    // 3. 并发 TCP 扫描视频端口（554/8554）
    std::function<QString(const QString &)> probe = [this](const QString &ip) -> QString {
        if (m_cancel.load()) return QString();
        for (int port : m_videoPorts) {
            if (probeTcpPort(ip, port, 250)) {
                return ip;
            }
        }
        return QString();
    };
    QList<QString> openIps = QtConcurrent::blockingMapped(ips, probe);

    if (m_cancel.load()) {
        m_searching.store(false);
        emit searchFinished(foundCount);
        return;
    }

    emit searchProgress(total, total, QString());

    // 4. 对发现开放视频端口的 IP 做 RTSP 验证，构造设备信息
    QSet<QString> reported;
    for (const Device &d : udpDevices) {
        reported.insert(d.ip);
    }

    int candidate = 0;
    for (const QString &ip : openIps) {
        if (ip.isEmpty() || reported.contains(ip) || m_cancel.load()) {
            continue;
        }
        reported.insert(ip);
        ++candidate;
        emit searchProgress(candidate, openIps.size(), ip);

        Device dev;
        dev.ip = ip;
        dev.name = ip;
        dev.protocol = "ACP";
        dev.controlPort = m_controlPort;
        dev.user = m_user;
        dev.pass = m_pass;

        verifyDevice(dev);
        if (!dev.verified) {
            dev.videoPort = m_videoPorts.isEmpty() ? 554 : m_videoPorts.first();
        }

        emit deviceFound(dev);
        ++foundCount;
    }

    m_searching.store(false);
    emit searchFinished(foundCount);
}

QStringList CameraSearcher::localSubnets() const
{
    QStringList result;
    const QList<QHostAddress> addrs = QNetworkInterface::allAddresses();
    for (const QHostAddress &a : addrs) {
        if (a.protocol() != QAbstractSocket::IPv4Protocol) continue;
        if (a.isLoopback()) continue;
        quint32 ip = a.toIPv4Address();
        if (ip == 0) continue;
        quint8 b1 = (ip >> 24) & 0xFF;
        quint8 b2 = (ip >> 16) & 0xFF;
        quint8 b3 = (ip >> 8) & 0xFF;
        QString sub = QString("%1.%2.%3").arg(b1).arg(b2).arg(b3);
        if (!result.contains(sub)) result.append(sub);
    }
    return result;
}

QStringList CameraSearcher::enumerateIps(const QString &subnet) const
{
    QStringList result;
    QStringList subnets;
    if (subnet.trimmed().isEmpty()) {
        subnets = localSubnets();
    } else {
        subnets << subnet.trimmed();
    }

    for (const QString &sub : subnets) {
        const QStringList parts = sub.split('.');
        QString prefix;
        if (parts.size() >= 3) {
            prefix = parts[0] + "." + parts[1] + "." + parts[2];
        } else if (parts.size() == 2) {
            prefix = parts[0] + "." + parts[1];
        } else {
            continue;
        }
        for (int i = 1; i <= 254; ++i) {
            result.append(QString("%1.%2").arg(prefix).arg(i));
        }
    }
    result.removeDuplicates();
    return result;
}

bool CameraSearcher::probeTcpPort(const QString &ip, int port, int timeoutMs) const
{
    QTcpSocket sock;
    sock.connectToHost(ip, static_cast<quint16>(port));
    return sock.waitForConnected(timeoutMs);
}

QString CameraSearcher::verifyRtsp(const QString &ip, int port, const QString &user,
                                   const QString &pass, const QStringList &paths,
                                   int timeoutMs, int *width, int *height) const
{
    for (const QString &path : paths) {
        if (m_cancel.load()) break;

        QString url = QString("rtsp://%1:%2@%3:%4%5")
                          .arg(user, pass, ip)
                          .arg(port)
                          .arg(path);

        cv::VideoCapture cap;
        // 关键：OPEN_TIMEOUT 必须在 open() 之前设置，避免 RTSP 打开阻塞过久。
        cap.set(cv::CAP_PROP_OPEN_TIMEOUT_MSEC, timeoutMs);
        cap.set(cv::CAP_PROP_READ_TIMEOUT_MSEC, timeoutMs);
        if (!cap.open(url.toStdString(), cv::CAP_FFMPEG)) {
            continue;
        }
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            cap.release();
            continue;
        }
        if (width)  *width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        if (height) *height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        cap.release();
        return path;
    }
    return QString();
}

bool CameraSearcher::verifyDevice(Device &device) const
{
    if (device.ip.isEmpty()) {
        return false;
    }

    // 凭据候选：优先使用界面填写的账号密码，再回退常见默认凭据，
    // 便于自动识别出厂默认设置的相机（如 admin/123456）。
    struct Cred {
        QString user;
        QString pass;
    };
    QVector<Cred> creds;
    auto addCred = [&creds](const QString &u, const QString &p) {
        for (const Cred &c : creds) {
            if (c.user == u && c.pass == p) return;
        }
        creds.append(Cred{u, p});
    };

    addCred(m_user, m_pass);
    addCred(m_user, QStringLiteral("123456"));
    addCred(QStringLiteral("admin"), QStringLiteral("123456"));
    addCred(QStringLiteral("admin"), QStringLiteral("admin123"));
    addCred(QStringLiteral("admin"), QStringLiteral("admin"));
    addCred(QStringLiteral("admin"), QString());

    for (const Cred &c : creds) {
        if (m_cancel.load()) return false;
        for (int port : m_videoPorts) {
            int w = 0, h = 0;
            const QString path = verifyRtsp(device.ip, port, c.user, c.pass,
                                            m_rtspPaths, m_timeoutMs, &w, &h);
            if (!path.isEmpty()) {
                device.videoPort = port;
                device.rtspPath = path;
                device.verified = true;
                device.user = c.user;
                device.pass = c.pass;
                device.rtspUrl = QString("rtsp://%1:%2@%3:%4%5")
                                     .arg(c.user, c.pass, device.ip)
                                     .arg(port).arg(path);
                device.resolution = QString("%1x%2").arg(w).arg(h);
                return true;
            }
        }
    }
    return false;
}

QVector<CameraSearcher::Device> CameraSearcher::udpProbe(int controlPort, int timeoutMs)
{
    QVector<Device> found;

    int sock = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        return found;
    }

    int broadcast = 1;
    ::setsockopt(sock, SOL_SOCKET, SO_BROADCAST, &broadcast, sizeof(broadcast));

    // 允许地址复用，避免本机控制端口被占用时无法绑定
    int reuse = 1;
    ::setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = timeoutMs * 1000;
    ::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    struct sockaddr_in local;
    std::memset(&local, 0, sizeof(local));
    local.sin_family = AF_INET;
    local.sin_addr.s_addr = INADDR_ANY;
    local.sin_port = 0; // 随机源端口，设备会回复到该端口
    ::bind(sock, reinterpret_cast<struct sockaddr *>(&local), sizeof(local));

    // ACP 搜索探测报文（私有协议，可按实际设备协议文档调整报文格式）。
    // 这里发送一个带魔数前缀的 ASCII 探测包；收到响应的源 IP 即视为设备地址。
    static const char kProbe[] = "ACP_DISCOVER";
    const int probeLen = static_cast<int>(std::strlen(kProbe));

    // 向全局广播地址及每个本机网段的广播地址发送
    QSet<QString> broadcastAddrs;
    broadcastAddrs.insert(QStringLiteral("255.255.255.255"));
    for (const QString &sub : localSubnets()) {
        broadcastAddrs.insert(sub + ".255");
    }

    struct sockaddr_in dest;
    std::memset(&dest, 0, sizeof(dest));
    dest.sin_family = AF_INET;
    dest.sin_port = htons(static_cast<quint16>(controlPort));

    for (const QString &addr : broadcastAddrs) {
        dest.sin_addr.s_addr = inet_addr(addr.toLatin1().constData());
        ::sendto(sock, kProbe, probeLen, 0,
                 reinterpret_cast<struct sockaddr *>(&dest), sizeof(dest));
    }

    // 等待设备响应
    QSet<QString> seenIps;
    char buf[4096];
    auto deadline = std::chrono::steady_clock::now()
                    + std::chrono::milliseconds(timeoutMs);
    while (std::chrono::steady_clock::now() < deadline) {
        if (m_cancel.load()) break;
        struct sockaddr_in from;
        socklen_t fromLen = sizeof(from);
        ssize_t n = ::recvfrom(sock, buf, sizeof(buf) - 1, 0,
                               reinterpret_cast<struct sockaddr *>(&from), &fromLen);
        if (n <= 0) {
            break; // 超时或出错
        }
        QString ip = QString::fromLatin1(inet_ntoa(from.sin_addr));
        if (ip.isEmpty() || seenIps.contains(ip)) {
            continue;
        }
        seenIps.insert(ip);

        Device d;
        d.ip = ip;
        d.name = ip;
        d.protocol = "ACP";
        d.controlPort = controlPort;
        // 设备从控制端口回复，其源端口即为实际控制端口（若与默认不同）
        int fromPort = ntohs(from.sin_port);
        if (fromPort > 0) {
            d.controlPort = fromPort;
        }
        found.append(d);
    }

    ::close(sock);
    return found;
}



