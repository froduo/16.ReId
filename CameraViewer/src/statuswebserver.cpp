#include "statuswebserver.h"
#include "logger.h"

#include <QTcpServer>
#include <QTcpSocket>
#include <QHostAddress>
#include <QJsonDocument>
#include <QJsonArray>

StatusWebServer::StatusWebServer(QObject *parent)
    : QObject(parent)
{
}

StatusWebServer::~StatusWebServer()
{
    stop();
}

bool StatusWebServer::start(quint16 port)
{
    if (m_running) return true;

    m_server = new QTcpServer(this);
    connect(m_server, &QTcpServer::newConnection, this, &StatusWebServer::onNewConnection);

    if (!m_server->listen(QHostAddress::Any, port)) {
        LOG_WARN(QString("[WebStatus] 监听端口 %1 失败: %2")
                     .arg(port).arg(m_server->errorString()));
        delete m_server;
        m_server = nullptr;
        return false;
    }

    m_port = m_server->serverPort();
    m_running = true;
    LOG_INFO(QString("[WebStatus] Web 服务已启动: http://0.0.0.0:%1").arg(m_port));
    return true;
}

void StatusWebServer::stop()
{
    if (m_server) {
        m_server->close();
        delete m_server;
        m_server = nullptr;
    }
    m_running = false;
}

void StatusWebServer::setStatus(const QJsonObject &status)
{
    QMutexLocker locker(&m_mutex);
    m_status = status;
}

void StatusWebServer::onNewConnection()
{
    while (m_server && m_server->hasPendingConnections()) {
        QTcpSocket *sock = m_server->nextPendingConnection();
        connect(sock, &QTcpSocket::readyRead, this, [this, sock]() {
            const QByteArray req = sock->readAll();
            handleRequest(sock, req);
        });
        connect(sock, &QTcpSocket::disconnected, sock, &QObject::deleteLater);
    }
}

void StatusWebServer::handleRequest(QTcpSocket *sock, const QByteArray &request)
{
    // 解析请求首行：GET /path HTTP/1.1
    QString path = QStringLiteral("/");
    const QString req = QString::fromUtf8(request);
    const int p1 = req.indexOf(' ');
    if (p1 >= 0) {
        const int p2 = req.indexOf(' ', p1 + 1);
        if (p2 > p1) path = req.mid(p1 + 1, p2 - p1 - 1);
    }

    QByteArray body;
    QByteArray contentType;
    if (path.startsWith(QLatin1String("/api/status"))) {
        QJsonObject st;
        {
            QMutexLocker locker(&m_mutex);
            st = m_status;
        }
        body = QJsonDocument(st).toJson(QJsonDocument::Compact);
        contentType = QByteArrayLiteral("application/json; charset=utf-8");
    } else {
        body = R"(<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>上下车状态</title>
<style>
body{font-family:sans-serif;background:#10151f;color:#e6edf3;margin:0;padding:16px}
h2{color:#00e5ff}.card{background:#16202e;border:1px solid #2b3b52;border-radius:10px;padding:14px;margin-bottom:12px}
.big{font-size:28px;font-weight:bold}.up{color:#69f0ae}.down{color:#ff8a80}.on{color:#ffab40}
.m{margin:0 4px}.row{display:flex;gap:10px;flex-wrap:wrap}
.row .card{flex:1;min-width:90px;text-align:center}
ul{list-style:none;padding:0}li{padding:6px 0;border-bottom:1px dashed #2b3b52}
</style>
</head>
<body>
<h2>上下车实时状态</h2>
<div class="row">
<div class="card"><div class="up big" id="boarding">--</div><div>上车</div></div>
<div class="card"><div class="down big" id="alighting">--</div><div>下车</div></div>
<div class="card"><div class="on big" id="onboard">--</div><div>车厢内</div></div>
</div>
<div class="card" id="info">加载中…</div>
<div class="card"><h3 style="color:#00e5ff">匹配记录</h3><ul id="matches"><li>暂无</li></ul></div>
<script>
async function refresh(){
  try{
    const r=await fetch('/api/status');
    const s=await r.json();
    document.getElementById('boarding').textContent=s.boarding??'--';
    document.getElementById('alighting').textContent=s.alighting??'--';
    document.getElementById('onboard').textContent=s.onboard??'--';
    document.getElementById('info').textContent='时间: '+(s.time||'')+'  站点: '+(s.station||'')+'  区域人数: '+(s.regionPersons??'--');
    const ul=document.getElementById('matches');
    const arr=s.matches||[];
    ul.innerHTML = arr.length ? arr.map(m=>'<li><span class="up">'+m.board+'</span> ↔ <span class="down">'+m.alight+'</span> <span class="m">相似度 '+Number(m.similarity).toFixed(4)+'</span></li>').join('') : '<li>暂无</li>';
  }catch(e){document.getElementById('info').textContent='连接失败: '+e;}
}
refresh();setInterval(refresh,3000);
</script>
</body>
</html>)";
        contentType = QByteArrayLiteral("text/html; charset=utf-8");
    }

    const QByteArray resp = QByteArrayLiteral("HTTP/1.1 200 OK\r\n")
            + QByteArrayLiteral("Content-Type: ") + contentType + QByteArrayLiteral("\r\n")
            + QByteArrayLiteral("Content-Length: ") + QByteArray::number(body.size()) + QByteArrayLiteral("\r\n")
            + QByteArrayLiteral("Connection: close\r\n")
            + QByteArrayLiteral("Access-Control-Allow-Origin: *\r\n")
            + QByteArrayLiteral("\r\n")
            + body;
    sock->write(resp);
    sock->flush();
    sock->disconnectFromHost();
}
