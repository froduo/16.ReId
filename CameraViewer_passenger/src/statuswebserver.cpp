#include "statuswebserver.h"
#include "logger.h"

#include <QTcpServer>
#include <QTcpSocket>
#include <QHostAddress>
#include <QJsonDocument>
#include <QJsonArray>
#include <QDir>
#include <QUrlQuery>
#include <QImage>
#include <QBuffer>

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

void StatusWebServer::setFrame(const QByteArray &jpeg)
{
    QMutexLocker locker(&m_mutex);
    m_frameJpeg = jpeg;
}

void StatusWebServer::setVideoDir(const QString &dir)
{
    QMutexLocker locker(&m_mutex);
    m_videoDir = dir;
}

void StatusWebServer::setPersonImageProvider(std::function<QString(int personId)> provider)
{
    QMutexLocker locker(&m_mutex);
    m_personImageProvider = std::move(provider);
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
    // 解析请求首行：METHOD /path HTTP/1.1
    QString method = QStringLiteral("GET");
    QString path = QStringLiteral("/");
    const QString req = QString::fromUtf8(request);
    const int p1 = req.indexOf(' ');
    if (p1 >= 0) {
        method = req.left(p1);
        const int p2 = req.indexOf(' ', p1 + 1);
        if (p2 > p1) path = req.mid(p1 + 1, p2 - p1 - 1);
    }

    QByteArray body;
    QByteArray contentType;
    if (method == QLatin1String("POST") && path.startsWith(QLatin1String("/api/control"))) {
        // 前端控制指令：解析 JSON body {action, ...}
        const int bodyPos = request.indexOf("\r\n\r\n");
        QByteArray payload;
        if (bodyPos >= 0) payload = request.mid(bodyPos + 4);
        const QJsonObject params = QJsonDocument::fromJson(payload).object();
        const QString action = params.value(QStringLiteral("action")).toString();
        emit controlRequested(action, params);
        QJsonObject resp;
        resp["ok"] = true;
        resp["action"] = action;
        body = QJsonDocument(resp).toJson(QJsonDocument::Compact);
        contentType = QByteArrayLiteral("application/json; charset=utf-8");
    } else if (path.startsWith(QLatin1String("/api/person_images"))) {
        const QUrl url(QStringLiteral("http://localhost") + path);
        const QUrlQuery query(url);
        const int personId = query.queryItemValue(QStringLiteral("personId")).toInt();
        const int maxN = qBound(1, query.queryItemValue(QStringLiteral("max")).toInt(), 60);
        std::function<QString(int)> provider;
        {
            QMutexLocker locker(&m_mutex);
            provider = m_personImageProvider;
        }
        QJsonArray images;
        if (provider) {
            const QString dir = provider(personId);
            QDir d(dir);
            QStringList files = d.entryList({QStringLiteral("*.jpg"), QStringLiteral("*.jpeg"),
                                             QStringLiteral("*.png"), QStringLiteral("*.bmp")},
                                            QDir::Files, QDir::Name);
            if (files.size() > maxN) {
                const double step = files.size() / static_cast<double>(maxN);
                QStringList sampled;
                for (int i = 0; i < maxN; ++i) sampled << files[static_cast<int>(i * step)];
                files = sampled;
            }
            for (const QString &f : files) {
                QImage img(d.filePath(f));
                if (img.isNull()) continue;
                img = img.scaled(100, 130, Qt::KeepAspectRatio, Qt::SmoothTransformation);
                QByteArray b64;
                QBuffer buf(&b64);
                buf.open(QIODevice::WriteOnly);
                img.save(&buf, "JPEG", 60);
                QJsonObject o;
                o["name"] = f;
                o["b64"] = QString::fromLatin1(b64.toBase64());
                images.append(o);
            }
        }
        QJsonObject resp;
        resp["images"] = images;
        body = QJsonDocument(resp).toJson(QJsonDocument::Compact);
        contentType = QByteArrayLiteral("application/json; charset=utf-8");
    } else if (path.startsWith(QLatin1String("/api/frame"))) {
        QMutexLocker locker(&m_mutex);
        body = m_frameJpeg;
        contentType = QByteArrayLiteral("image/jpeg");
    } else if (path.startsWith(QLatin1String("/api/videos"))) {
        QJsonArray arr;
        {
            QMutexLocker locker(&m_mutex);
            QDir dir(m_videoDir);
            if (dir.exists()) {
                const QStringList files = dir.entryList({QStringLiteral("*.mp4")},
                                                        QDir::Files, QDir::Name);
                for (const QString &f : files) {
                    QJsonObject v;
                    v["name"] = f;
                    v["path"] = dir.filePath(f);
                    arr.append(v);
                }
            }
        }
        body = QJsonDocument(arr).toJson(QJsonDocument::Compact);
        contentType = QByteArrayLiteral("application/json; charset=utf-8");
    } else if (path.startsWith(QLatin1String("/api/status"))) {
        QJsonObject st;
        {
            QMutexLocker locker(&m_mutex);
            st = m_status;
        }
        body = QJsonDocument(st).toJson(QJsonDocument::Compact);
        contentType = QByteArrayLiteral("application/json; charset=utf-8");
    } else {
        body = R"HTML(<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HX-CCD23 乘客检测系统</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC',sans-serif;background:#0d1117;color:#e6edf3;min-height:100vh}
header{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap;padding:14px 20px;background:#161b22;border-bottom:1px solid #30363d}
header .title{font-size:18px;font-weight:bold;color:#00e5ff}
header .meta{font-size:12px;color:#8b949e;display:flex;gap:14px;flex-wrap:wrap}
.status-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px}
.status-dot.run{background:#69f0ae;box-shadow:0 0 6px #69f0ae}
.status-dot.stop{background:#ff8a80;box-shadow:0 0 6px #ff8a80}
main{padding:16px;max-width:1200px;margin:0 auto}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:16px}
.card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:18px;text-align:center}
.card .num{font-size:32px;font-weight:bold}
.card .lbl{font-size:12px;color:#8b949e;margin-top:4px}
.up{color:#69f0ae}.down{color:#ff8a80}.on{color:#ffab40}.region{color:#40c4ff}
.section{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px;margin-bottom:16px}
.section h3{font-size:14px;color:#00e5ff;margin-bottom:10px;border-bottom:1px solid #30363d;padding-bottom:8px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:8px;text-align:left;border-bottom:1px solid #21262d}
th{color:#8b949e;font-weight:normal}
.match-list{list-style:none}
.match-list li{padding:8px 0;border-bottom:1px dashed #21262d;font-size:13px}
.foot{text-align:center;color:#8b949e;font-size:12px;padding:16px}
.btn{background:#21262d;border:1px solid #30363d;color:#e6edf3;padding:8px 16px;border-radius:6px;cursor:pointer;font-size:13px}
.btn:hover{background:#30363d}
.btn.danger{color:#ff8a80;border-color:#5c2b2b}
.btn.primary{background:#0d419d;border-color:#1f6feb;color:#fff}
.imgs-row{display:flex;gap:4px;overflow-x:auto;padding:6px 0}
.imgs-row img{width:60px;height:78px;object-fit:cover;border:1px solid #30363d;border-radius:4px;flex-shrink:0}
</style>
</head>
<body>
<header>
<div class="title">HX-CCD23 乘客检测系统</div>
<div class="meta">
<span><span class="status-dot" id="dot"></span><span id="status">--</span></span>
<span>站点: <span id="station">--</span></span>
<span>模式: <span id="mode">--</span></span>
<span>FPS: <span id="fps">--</span></span>
<span id="version">--</span>
</div>
</header>
<main>
<div class="section" style="display:flex;gap:10px;flex-wrap:wrap;align-items:center">
<span style="font-size:13px;color:#8b949e">控制：</span>
<button class="btn primary" onclick="ctrl('toggleCapture')">启动/停止采集</button>
<button class="btn danger" onclick="ctrl('clearCounts')">清除计数</button>
</div>
<div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px">
<div class="section" style="flex:2;min-width:280px;margin-bottom:0">
<h3>实时画面</h3>
<img id="frame" src="/api/frame" style="width:100%;border-radius:8px;background:#000" alt="实时画面">
</div>
<div class="section" style="flex:1;min-width:220px;margin-bottom:0">
<h3>视频列表</h3>
<select id="videoSel" style="width:100%;padding:8px;background:#0d1117;color:#e6edf3;border:1px solid #30363d;border-radius:6px;margin-bottom:8px"></select>
<button class="btn primary" onclick="loadVideo()" style="width:100%">加载所选视频</button>
<button class="btn" onclick="ctrl('togglePlay')" style="width:100%;margin-top:6px">播放/暂停</button>
</div>
</div>
<div class="grid">
<div class="card"><div class="num up" id="boarding">--</div><div class="lbl">上车</div></div>
<div class="card"><div class="num down" id="alighting">--</div><div class="lbl">下车</div></div>
<div class="card"><div class="num on" id="onboard">--</div><div class="lbl">车厢内</div></div>
<div class="card"><div class="num region" id="regionPersons">--</div><div class="lbl">区域人数</div></div>
</div>
<div class="section">
<h3>匹配人工确认（上/下车配对）</h3>
<table><thead><tr><th>上车</th><th>下车</th><th>相似度</th><th>时间</th><th>确认</th><th>图片</th></tr></thead>
<tbody id="matchPairs"><tr><td colspan="6">暂无</td></tr></tbody></table>
<div id="pairImages" style="margin-top:10px;display:none">
<div style="color:#69f0ae;font-size:12px;margin-bottom:4px">上车图片：</div>
<div class="imgs-row" id="boardImgs"></div>
<div style="color:#ff8a80;font-size:12px;margin:8px 0 4px">下车图片：</div>
<div class="imgs-row" id="alightImgs"></div>
</div>
</div>
<div class="section">
<h3>人员进出记录</h3>
<table><thead><tr><th>person</th><th>状态</th><th>时间</th><th>相似度</th></tr></thead>
<tbody id="events"><tr><td colspan="4">暂无</td></tr></tbody></table>
</div>
</main>
<div class="foot">更新于 <span id="time">--</span> · 自动刷新 3s</div>
<script>
async function refresh(){
try{
const r=await fetch('/api/status');
const s=await r.json();
document.getElementById('status').textContent=s.status||'--';
document.getElementById('dot').className='status-dot '+(s.status==='运行中'?'run':'stop');
document.getElementById('station').textContent=s.station||'--';
document.getElementById('mode').textContent=(s.mode||'--')+(s.speed?' ('+s.speed+')':'');
document.getElementById('fps').textContent=s.fps||'--';
document.getElementById('version').textContent=s.version||'';
document.getElementById('boarding').textContent=s.boarding??'--';
document.getElementById('alighting').textContent=s.alighting??'--';
document.getElementById('onboard').textContent=s.onboard??'--';
document.getElementById('regionPersons').textContent=s.regionPersons??'--';
document.getElementById('time').textContent=s.time||'';
const mpTb=document.getElementById('matchPairs');
const mp=s.matchPairs||[];
mpTb.innerHTML=mp.length?mp.map(p=>{
const c=['未确认','正确','错误'][p.confirm]||'未确认';
return '<tr><td class="up">person_'+p.boardId+'</td><td class="down">person_'+p.alightId+'</td><td>'+Number(p.similarity).toFixed(3)+'</td><td>'+p.time+'</td><td><button class="btn" onclick="confirmMatch('+p.boardId+',1)">✓正确</button> <button class="btn danger" onclick="confirmMatch('+p.boardId+',2)">✗错误</button> <span style="color:#8b949e">'+c+'</span></td><td><button class="btn" onclick="showPairImages('+p.boardId+','+p.alightId+')">查看</button></td></tr>';
}).join(''):'<tr><td colspan="6">暂无</td></tr>';
const tb=document.getElementById('events');
const ev=s.events||[];
tb.innerHTML=ev.length?ev.map(e=>'<tr><td>person_'+e.person+'</td><td>'+e.status+'</td><td>'+e.time+'</td><td>'+e.sim+'</td></tr>').join(''):'<tr><td colspan="4">暂无</td></tr>';
}catch(e){document.getElementById('time').textContent='连接失败: '+e;}
}
async function ctrl(action,params){
try{await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(Object.assign({action},params||{}))});setTimeout(refresh,300);}catch(e){alert('控制失败: '+e);}
}
function confirmMatch(boardId,confirm){ctrl('confirmMatch',{boardId,confirm});}
function showPairImages(boardId,alightId){
document.getElementById('pairImages').style.display='block';
loadPersonImages(boardId,'boardImgs');
loadPersonImages(alightId,'alightImgs');
}
async function loadPersonImages(personId,elId){
try{
const r=await fetch('/api/person_images?personId='+personId+'&max=8');
const d=await r.json();
const el=document.getElementById(elId);
const imgs=d.images||[];
el.innerHTML=imgs.length?imgs.map(i=>'<img src="data:image/jpeg;base64,'+i.b64+'" title="'+i.name+'">').join(''):'<span style="color:#8b949e">无图片</span>';
}catch(e){}
}
async function loadVideos(){
try{
const r=await fetch('/api/videos');
const arr=await r.json();
const sel=document.getElementById('videoSel');
sel.innerHTML=arr.map(v=>'<option value="'+v.path+'">'+v.name+'</option>').join('')||'<option value="">无视频</option>';
}catch(e){}
}
function loadVideo(){
const path=document.getElementById('videoSel').value;
if(path)ctrl('loadVideo',{path});
}
function refreshFrame(){
const img=document.getElementById('frame');
img.onload=function(){setTimeout(refreshFrame,200);};
img.onerror=function(){setTimeout(refreshFrame,500);};
img.src='/api/frame?t='+Date.now();
}
refresh();setInterval(refresh,3000);
loadVideos();
refreshFrame();
</script>
</body>
</html>)HTML";
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
