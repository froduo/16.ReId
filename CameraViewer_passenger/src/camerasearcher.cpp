#include "camerasearcher.h"
#include "logger.h"

#include <QNetworkInterface>
#include <QHostAddress>
#include <QTcpSocket>
#include <QThreadPool>
#include <QSet>
#include <QHash>
#include <QtConcurrent/QtConcurrent>
#include <QRegularExpression>
#include <QUdpSocket>
#include <QUrl>
#include <QElapsedTimer>
#include <QUuid>
#include <QMutex>

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

    // 0. 统一设置 FFMPEG RTSP 超时（stimeout，微秒），整个搜索期间生效，
    //    避免并发 RTSP 验证时的环境变量竞态。
    const QByteArray prevOpt = qgetenv("OPENCV_FFMPEG_CAPTURE_OPTIONS");
    qputenv("OPENCV_FFMPEG_CAPTURE_OPTIONS",
            QByteArray("rtsp_transport;tcp|stimeout;") + QByteArray::number(m_timeoutMs * 1000));

    // 1. 枚举需要扫描的 IP 列表
    QStringList ips = enumerateIps(m_subnet);
    int total = ips.size();
    emit searchProgress(0, total, QString());

    // 2. ONVIF WS-Discovery 多播搜索（标准协议，支持海康/大华等主流 IPC，全网通）
    emit searchProgress(0, total, QStringLiteral("正在 ONVIF 搜索…"));
    QSet<QString> reported;
    QVector<Device> onvifDevices = onvifProbe(3000);
    for (Device &d : onvifDevices) {
        if (m_cancel.load()) break;
        if (!m_user.isEmpty()) d.user = m_user;
        if (!m_pass.isEmpty()) d.pass = m_pass;
        verifyDevice(d, 554);
        reported.insert(d.ip);
        emit deviceFound(d);
        ++foundCount;
    }

    if (m_cancel.load()) {
        m_searching.store(false);
        emit searchFinished(foundCount);
        return;
    }

    // 3. ACP 协议 UDP 广播搜索（控制端口 8091）
    QVector<Device> udpDevices = udpProbe(m_controlPort, 1200);
    for (const Device &d : udpDevices) {
        if (m_cancel.load()) break;
        if (reported.contains(d.ip)) {
            continue;
        }
        Device dev = d;
        verifyDevice(dev);
        if (dev.isValid()) {
            reported.insert(dev.ip);
            emit deviceFound(dev);
            ++foundCount;
        }
    }

    if (m_cancel.load()) {
        m_searching.store(false);
        emit searchFinished(foundCount);
        return;
    }

    // 3. 并发 TCP 扫描视频端口：按端口顺序（554 优先），并记录每个 IP 开放的端口，
    //    后续只对「开放的端口」做 RTSP 验证，避免对无效端口重复超时
    //    提高线程池并发 + 缩短探测超时，加速网段扫描（254 地址 × 2 端口）
    const int prevMaxThread = QThreadPool::globalInstance()->maxThreadCount();
    QThreadPool::globalInstance()->setMaxThreadCount(64);
    QList<QString> openIps;
    QHash<QString, int> openPorts;  // ip -> 开放的视频端口
    for (int port : m_videoPorts) {
        std::function<QString(const QString &)> probe = [this, port](const QString &ip) -> QString {
            if (m_cancel.load()) return QString();
            if (probeTcpPort(ip, port, 120)) {
                return ip;
            }
            return QString();
        };
        const QList<QString> found = QtConcurrent::blockingMapped(ips, probe);
        for (const QString &ip : found) {
            if (!ip.isEmpty() && !openIps.contains(ip)) {
                openIps.append(ip);
                openPorts.insert(ip, port);
            }
        }
    }
    QThreadPool::globalInstance()->setMaxThreadCount(prevMaxThread);

    if (m_cancel.load()) {
        m_searching.store(false);
        emit searchFinished(foundCount);
        return;
    }

    emit searchProgress(total, total, QString());

    // 4. 对发现开放视频端口的 IP 做 RTSP 验证，构造设备信息
    //    （reported 已包含 ONVIF / ACP UDP 发现的设备，避免重复验证）

    int candidate = 0;
    // 收集待验证的 IP（跳过已报告的 ONVIF/UDP 设备）
    QStringList pendingIps;
    for (const QString &ip : openIps) {
        if (ip.isEmpty() || reported.contains(ip)) {
            continue;
        }
        reported.insert(ip);
        pendingIps.append(ip);
    }

    // 并发 RTSP 验证（多个 IP 并行，大幅缩短网段扫描时间）
    QMutex resultMutex;
    QVector<Device> verifiedDevices;
    std::function<void(const QString &)> verifyOne = [&](const QString &ip) {
        if (m_cancel.load()) return;
        Device dev;
        dev.ip = ip;
        dev.name = ip;
        dev.protocol = "ACP";
        dev.controlPort = m_controlPort;
        dev.user = m_user;
        dev.pass = m_pass;
        const int port = openPorts.value(ip, m_videoPorts.isEmpty() ? 554 : m_videoPorts.first());
        dev.videoPort = port;
        verifyDevice(dev, port);
        if (!dev.verified) {
            dev.videoPort = port;
        }
        QMutexLocker locker(&resultMutex);
        verifiedDevices.append(dev);
    };
    QtConcurrent::blockingMap(pendingIps, verifyOne);

    // 按 TCP 扫描顺序回传结果
    for (const Device &dev : verifiedDevices) {
        if (m_cancel.load()) break;
        ++candidate;
        emit searchProgress(candidate, openIps.size(), dev.ip);
        emit deviceFound(dev);
        ++foundCount;
    }

    // 恢复 FFMPEG 环境变量
    if (prevOpt.isEmpty()) {
        qunsetenv("OPENCV_FFMPEG_CAPTURE_OPTIONS");
    } else {
        qputenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", prevOpt);
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
    const QString spec = subnet.trimmed();
    if (spec.isEmpty()) {
        subnets = localSubnets();
    } else {
        // 支持逗号/分号/空白分隔的多个网段（如 "192.168.200,192.168.10"）
        const QStringList parts = spec.split(QRegularExpression(QStringLiteral("[,\\s;]+")),
                                            Qt::SkipEmptyParts);
        for (const QString &p : parts) {
            const QString s = p.trimmed();
            if (!s.isEmpty() && !subnets.contains(s)) {
                subnets << s;
            }
        }
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
    // 注：FFMPEG RTSP 打开/读超时（stimeout）在 doSearch 里统一通过
    // OPENCV_FFMPEG_CAPTURE_OPTIONS 设置（进程级一次），这里不再临时切换，
    // 以支持多线程并发验证时避免环境变量竞态。
    QString result;
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
        result = path;
        break;
    }

    return result;
}

bool CameraSearcher::verifyDevice(Device &device, int onlyPort) const
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
    addCred(QStringLiteral("admin"), QStringLiteral("123456"));

    for (const Cred &c : creds) {
        if (m_cancel.load()) return false;
        const QVector<int> ports = (onlyPort > 0) ? QVector<int>{onlyPort} : m_videoPorts;
        for (int port : ports) {
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
    tv.tv_sec = timeoutMs / 1000;
    tv.tv_usec = (timeoutMs % 1000) * 1000;
    if (::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) != 0) {
        ::close(sock);
        return found;
    }

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

    // 向全局广播地址、每个本机网段的广播地址、以及用户指定网段的广播地址发送
    QSet<QString> broadcastAddrs;
    broadcastAddrs.insert(QStringLiteral("255.255.255.255"));
    for (const QString &sub : localSubnets()) {
        broadcastAddrs.insert(sub + ".255");
    }
    if (!m_subnet.trimmed().isEmpty()) {
        const QStringList parts = m_subnet.split(QRegularExpression(QStringLiteral("[,\\s;]+")),
                                                Qt::SkipEmptyParts);
        for (const QString &p : parts) {
            const QStringList seg = p.trimmed().split('.');
            if (seg.size() >= 3) {
                broadcastAddrs.insert(seg[0] + "." + seg[1] + "." + seg[2] + ".255");
            }
        }
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


QVector<CameraSearcher::Device> CameraSearcher::onvifProbe(int timeoutMs)
{
    QVector<Device> found;
    const QHostAddress multicast(QStringLiteral("239.255.255.250"));
    const quint16 wsPort = 3702;

    QUdpSocket sock;
    if (!sock.bind(QHostAddress::AnyIPv4, 0,
                   QUdpSocket::ShareAddress | QUdpSocket::ReuseAddressHint)) {
        return found;
    }
    // 提高多播 TTL，尽量跨网段可达（默认 TTL=1 只在本网段）
    sock.setSocketOption(QAbstractSocket::MulticastTtlOption, 3);
    if (!sock.joinMulticastGroup(multicast)) {
        return found;
    }

    // WS-Discovery Probe（SOAP over UDP）
    const QString msgId = QUuid::createUuid().toString(QUuid::WithoutBraces);
    const QByteArray probe = QByteArray(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        "<e:Envelope xmlns:e=\"http://www.w3.org/2003/05/soap-envelope\""
        " xmlns:w=\"http://schemas.xmlsoap.org/ws/2004/08/addressing\""
        " xmlns:d=\"http://schemas.xmlsoap.org/ws/2005/04/discovery\""
        " xmlns:dn=\"http://www.onvif.org/ver10/network/wsdl\">"
        "<e:Header>"
        "<w:MessageID>uuid:")
        + msgId.toUtf8()
        + QByteArray(
        "</w:MessageID>"
        "<w:To e:mustUnderstand=\"true\">urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>"
        "<w:Action e:mustUnderstand=\"true\">http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>"
        "</e:Header>"
        "<e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></e:Body>"
        "</e:Envelope>");

    sock.writeDatagram(probe, multicast, wsPort);

    QSet<QString> seenIps;
    QElapsedTimer timer;
    timer.start();
    while (timer.elapsed() < timeoutMs) {
        if (m_cancel.load()) break;
        if (!sock.waitForReadyRead(200)) {
            continue;
        }
        while (sock.hasPendingDatagrams()) {
            QByteArray data;
            data.resize(static_cast<int>(sock.pendingDatagramSize()));
            QHostAddress from;
            quint16 fromPort = 0;
            sock.readDatagram(data.data(), data.size(), &from, &fromPort);
            parseOnvifResponse(data, from, found, seenIps);
        }
    }

    sock.leaveMulticastGroup(multicast);
    sock.close();
    return found;
}


void CameraSearcher::parseOnvifResponse(const QByteArray &data, const QHostAddress &from,
                                        QVector<Device> &found, QSet<QString> &seenIps)
{
    const QString xml = QString::fromUtf8(data);
    if (!xml.contains(QStringLiteral("ProbeMatch"))) {
        return;
    }

    // 优先从 XAddrs/Scopes 提取设备 IP，取不到则用响应源 IP
    QString ip = from.toString();
    const QRegularExpression ipRe(
        QStringLiteral("(?:https?://)?(\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3})"));
    const auto m = ipRe.match(xml);
    if (m.hasMatch()) {
        ip = m.captured(1);
    }

    // 从 Scopes 提取设备名称（onvif://www.onvif.org/name/xxx）
    QString name = ip;
    const QRegularExpression nameRe(
        QStringLiteral("onvif://www\\.onvif\\.org/name/([^\\s<]+)"));
    const auto nm = nameRe.match(xml);
    if (nm.hasMatch()) {
        name = QUrl::fromPercentEncoding(nm.captured(1).toUtf8());
    }

    if (ip.isEmpty() || seenIps.contains(ip)) {
        return;
    }
    seenIps.insert(ip);

    Device d;
    d.ip = ip;
    d.name = name;
    d.group = QStringLiteral("默认分组");
    d.deviceType = QStringLiteral("IPC");
    d.protocol = QStringLiteral("ONVIF");
    d.videoPort = 554;
    d.controlPort = 3702;
    found.append(d);
}



