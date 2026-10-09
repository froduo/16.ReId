#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QMessageBox>
#include <QFile>
#include <QTimer>
#include <QThread>
#include <QCoreApplication>
#include <QStorageInfo>
#include <QDir>
#include <QDateTime>
#include <QPainter>
#include <opencv2/opencv.hpp>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    QString stylePath = QCoreApplication::applicationDirPath() + "/style.qss";
    QFile styleFile(stylePath);
    if (!styleFile.exists()) {
        stylePath = QCoreApplication::applicationDirPath() + "/../style.qss";
        styleFile.setFileName(stylePath);
    }
    if (styleFile.open(QFile::ReadOnly)) {
        QString styleSheet = QLatin1String(styleFile.readAll());
        this->setStyleSheet(styleSheet);
        styleFile.close();
        LOG_INFO("Stylesheet loaded: " + stylePath);
    } else {
        LOG_WARN("Stylesheet not found: " + stylePath);
    }

    LOG_INFO("CameraViewer starting...");
    m_config.load();

    {
        QDir saveDir(m_config.savePath);
        if (!saveDir.isAbsolute()) {
            QString appDir = QCoreApplication::applicationDirPath();
            m_config.savePath = QDir(appDir).absoluteFilePath(m_config.savePath);
        }
        saveDir.mkpath(".");
        LOG_INFO("Save path: " + m_config.savePath);
    }

    LOG_INFO("Config loaded.");
    this->setWindowTitle(m_config.windowTitle);
    LOG_INFO("CameraViewer starting... Version: " + QString(APP_VERSION_STR));

    // 替换 UI 中的 QGraphicsView 为自定义 ImageView
    QGraphicsView* oldView = ui->imageView;
    if (oldView) {
        m_imageView = new ImageView(this);
        m_imageView->setObjectName("imageViewMain");
        m_imageView->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        QVBoxLayout* mainLayout = ui->mainVerticalLayout;
        if (mainLayout) {
            // 找到 imageView 在 imageLayout 中的位置
            QHBoxLayout* imageLayout = ui->imageLayout;
            if (imageLayout) {
                int idx = imageLayout->indexOf(oldView);
                if (idx >= 0) {
                    imageLayout->removeWidget(oldView);
                    imageLayout->insertWidget(idx, m_imageView);
                }
            }
        }
        delete oldView;
    }

    ui->statusbar->hide();
    ui->timeLabel->setStyleSheet("color: #4dd0e1; font-family: 'Consolas'; font-size: 12px; font-weight: bold; padding: 0 8px;");
    ui->runTimeLabel->setStyleSheet("color: #ffd54f; font-family: 'Consolas'; font-size: 12px; font-weight: bold; padding: 0 8px;");
    ui->diskLabel->setStyleSheet("color: #81c784; font-family: 'Consolas'; font-size: 12px; font-weight: bold; padding: 0 8px;");
    ui->captureStatusLabel->setStyleSheet("color: #b0bec5; font-family: 'Consolas'; font-size: 12px; font-weight: bold; padding: 0 8px;");

    // 连接信号槽 - 使用 toggle 按钮替代 btnOpen/btnClose
    connect(ui->btnToggle, &QPushButton::clicked, this, &MainWindow::onToggleCapture);
    connect(ui->btnSave, &QPushButton::clicked, this, &MainWindow::onSaveCurrentFrame);
    connect(ui->chkAutoSave, &QCheckBox::toggled, this, &MainWindow::onAutoSaveToggled);

    // 区域绘制相关信号槽
    connect(ui->btnDrawRegion, &QPushButton::clicked, this, &MainWindow::onDrawRegion);
    connect(ui->btnClearRegion, &QPushButton::clicked, this, &MainWindow::onClearRegions);
    connect(ui->btnOfflineDetect, &QPushButton::clicked, this, &MainWindow::onOfflineDetect);
    connect(ui->btnSettings, &QPushButton::clicked, this, &MainWindow::onSetParams);
    // 删除历史上下车记录
    connect(ui->btnClearHistory, &QPushButton::clicked, this, &MainWindow::onClearHistory);

    // ImageView 多边形绘制完成信号
    if (m_imageView) {
        connect(m_imageView, &ImageView::polygonCompleted, this, &MainWindow::onPolygonCompleted);
    }

    // 初始化访客记录表格
    // 列：ID | 进入时间 | 进入方向 | 离开时间 | 离开方向 | 停留 | 状态
    if (ui->tableVisitors) {
        ui->tableVisitors->setColumnCount(7);
        ui->tableVisitors->setHorizontalHeaderLabels(
            {"ID", "进入时间", "进入方向", "离开时间", "离开方向", "停留", "状态"});
        ui->tableVisitors->horizontalHeader()->setStretchLastSection(true);
        ui->tableVisitors->setEditTriggers(QAbstractItemView::NoEditTriggers);
        ui->tableVisitors->setSelectionBehavior(QAbstractItemView::SelectRows);
        ui->tableVisitors->setSelectionMode(QAbstractItemView::SingleSelection);
        // 设置列宽（紧凑布局）
        ui->tableVisitors->setColumnWidth(0, 34);   // ID
        ui->tableVisitors->setColumnWidth(1, 56);   // 进入时间
        ui->tableVisitors->setColumnWidth(2, 50);   // 进入方向
        ui->tableVisitors->setColumnWidth(3, 56);   // 离开时间
        ui->tableVisitors->setColumnWidth(4, 50);   // 离开方向
        ui->tableVisitors->setColumnWidth(5, 42);   // 停留
        // 第7列(状态)自动拉伸
        ui->tableVisitors->verticalHeader()->setDefaultSectionSize(22); // 行高紧凑
    }

    // 初始化进入方向下拉框
    if (ui->cmbEntryDirection) {
        m_currentEntryDirection = (ui->cmbEntryDirection->currentIndex() == 1)
                                      ? EntryDirection::BottomToTop
                                      : EntryDirection::TopToBottom;
    }

    ui->btnSave->setEnabled(false);
    m_isCapturing = false;

    // 创建区域管理器
    m_regionManager = new RegionManager(this);

    // 加载已保存的区域
    {
        QString regionPath = m_config.regionSavePath;
        if (!QDir(regionPath).isAbsolute()) {
            regionPath = QCoreApplication::applicationDirPath() + "/" + regionPath;
        }
        m_regionManager->loadRegions(regionPath);

        // 将已保存的区域显示在 ImageView 中
        if (m_imageView) {
            QVector<QVector<QPointF>> polys;
            for (const auto &r : m_regionManager->regions()) {
                polys.append(r.polygon);
            }
            m_imageView->setRegionPolygons(polys);
        }
    }

    m_sysTimer = new QTimer(this);
    connect(m_sysTimer, &QTimer::timeout, this, &MainWindow::onUpdateSystemTime);
    m_sysTimer->start(1000);
    onUpdateSystemTime();

    m_diskTimer = new QTimer(this);
    connect(m_diskTimer, &QTimer::timeout, this, &MainWindow::updateDiskSpaceStatus);
    m_diskTimer->start(30000);
    updateDiskSpaceStatus();

    initSaveWorker();
    initVisitorLog();

    // 动恰客流上行（时段聚合 + 单人上下车事件）
    initUplink();

    // 全局身份注册表（跨轨迹稳定 ID）
    m_registry = new VisitorRegistry();
    m_registry->setBaseThreshold(m_config.reidMatchThreshold);
    m_registry->setGalleryMaxAgeMs(qMax<qint64>(60000,
                                    static_cast<qint64>(m_config.trackerGalleryMaxAgeMs) * 6));

    // 创建通道监控器（上车/下车检测）
    m_zoneMonitor = new ZoneMonitor(this);
    m_zoneMonitor->setVisitorRegistry(m_registry);
    m_zoneMonitor->setReidMatchThreshold(m_config.reidMatchThreshold);
    m_zoneMonitor->setConfirmFrames(3);
    connect(m_zoneMonitor, &ZoneMonitor::boardingEvent, this, [this](const ZoneEvent &evt) {
        m_boardingCount++;
        // 稳定身份：优先使用穿越事件携带的 personId（与显示 ID 一致）
        const int pid = evt.visitorId > 0 ? evt.visitorId
                                          : m_trackToVisitor.value(evt.trackId, -1);
        // 喂给时段聚合器（上车 = B→A 的完整穿越）
        if (m_aggregator) {
            m_aggregator->onBoarding(evt.visitorId, evt.enterTime, evt.exitTime,
                                     0.0f, evt.enterImage, evt.exitImage, evt.regionId);
        }
        // 落地「同一 ID：进入时间/方向 + 离开时间/方向 + 停留」记录
        applyCrossingRecord(pid, evt, QStringLiteral("boarding"));
        LOG_INFO(QString("[MainWindow] ★ BOARDING EVENT: V%1, count=%2, dwell=%3s")
                     .arg(pid).arg(m_boardingCount)
                     .arg(evt.enterTime.isValid() && evt.exitTime.isValid()
                              ? evt.enterTime.secsTo(evt.exitTime) : 0));
    });
    connect(m_zoneMonitor, &ZoneMonitor::alightingEvent, this, [this](const ZoneEvent &evt) {
        m_alightingCount++;
        const int pid = evt.visitorId > 0 ? evt.visitorId
                                          : m_trackToVisitor.value(evt.trackId, -1);
        // 喂给时段聚合器（下车 = A→B 的完整穿越）
        if (m_aggregator) {
            m_aggregator->onAlighting(evt.visitorId, evt.enterTime, evt.exitTime,
                                      0.0f, evt.enterImage, evt.exitImage, evt.regionId);
        }
        applyCrossingRecord(pid, evt, QStringLiteral("alighting"));
        LOG_INFO(QString("[MainWindow] ★ ALIGHTING EVENT: V%1, count=%2, dwell=%3s")
                     .arg(pid).arg(m_alightingCount)
                     .arg(evt.enterTime.isValid() && evt.exitTime.isValid()
                              ? evt.enterTime.secsTo(evt.exitTime) : 0));
    });

    // 同侧进出（跨过门线后又退回）→ 折返
    connect(m_zoneMonitor, &ZoneMonitor::zoneCrossingEvent, this, [this](const ZoneEvent &evt) {
        if (evt.entrySide != evt.exitSide) return;   // 折返 = 同侧进出

        if (m_aggregator) m_aggregator->onTurnBack(evt.exitTime);

        const int pid = evt.visitorId > 0 ? evt.visitorId
                                          : m_trackToVisitor.value(evt.trackId, -1);
        applyCrossingRecord(pid, evt, QStringLiteral("turnback"));
    });

    m_elapsedTimer.start();
    m_lastUiUpdate.start();
    m_fpsTimer.start();

    updateCaptureStatus();
}

void MainWindow::initUplink()
{
    // ===== 时段聚合器：把上车/下车/进出事件聚合成协议报文 =====
    m_aggregator = new FlowAggregator(this);
    m_aggregator->setMasterSn(m_config.uplinkMasterSn);

    QStringList sns;
    if (!m_config.uplinkDeviceSns.trimmed().isEmpty()) {
        sns = m_config.uplinkDeviceSns.split(',', Qt::SkipEmptyParts);
        for (QString &s : sns) s = s.trimmed();
    } else {
        sns = QStringList{m_config.uplinkMasterSn};
    }
    m_aggregator->setDeviceSns(sns);
    m_aggregator->setStatBasis(m_config.uplinkStatBasis);
    m_aggregator->setDayCutHour(m_config.uplinkDayCutHour);
    m_aggregator->setTimezone(m_config.uplinkTzName, m_config.uplinkUtcOffsetSeconds, false);

    // ===== HTTP 上行客户端 =====
    m_uplink = new HttpUplink(this);
    HttpUplink::Config uc;
    uc.baseUrl = m_config.uplinkBaseUrl;
    uc.intervalPath = m_config.uplinkIntervalPath;
    uc.personPath = m_config.uplinkPersonPath;
    uc.username = m_config.uplinkUser;
    uc.password = m_config.uplinkPass;
    uc.masterSn = m_config.uplinkMasterSn;
    uc.deviceSns = sns;
    uc.spoolDir = m_config.uplinkSpoolDir;
    uc.retryIntervalMs = m_config.uplinkRetryIntervalMs;
    uc.enabled = m_config.uplinkEnabled;
    m_uplink->setConfig(uc);

    // 聚合结果 → 上行队列
    connect(m_aggregator, &FlowAggregator::intervalReady,
            this, [this](const DongqiaFlow::IntervalAggregate &agg) {
                if (m_uplink) m_uplink->enqueueInterval(agg);
            });
    connect(m_aggregator, &FlowAggregator::personEventReady,
            this, [this](const DongqiaFlow::PersonFlowEvent &evt) {
                if (m_uplink) m_uplink->enqueuePersonEvent(evt);
            });

    // 精准识别某人上下车 → 打印该人的完整时间线
    connect(m_aggregator, &FlowAggregator::personTracked,
            this, [this](int visitorId, const QString &type, const QDateTime &t) {
                LOG_INFO(QString("[精准识别] event=%1 @ %2 | %3")
                             .arg(type, t.toString("HH:mm:ss.zzz"),
                                  m_aggregator->summarizePerson(visitorId)));
            });

    connect(m_uplink, &HttpUplink::messageSent,
            this, [](const QString &kind, bool ok, const QString &detail) {
                LOG_INFO(QString("[HttpUplink] %1 %2: %3")
                             .arg(kind, ok ? QStringLiteral("OK") : QStringLiteral("FAIL"), detail));
            });

    if (m_config.uplinkEnabled) {
        m_uplink->start();
        LOG_INFO(QString("[MainWindow] 动恰客流上行已启动: %1%2 (sn=%3, user=%4)")
                     .arg(m_config.uplinkBaseUrl, m_config.uplinkIntervalPath,
                          m_config.uplinkMasterSn, m_config.uplinkUser));
    } else {
        LOG_WARN("[MainWindow] 动恰客流上行已禁用（配置 Uplink/Enabled=false）");
    }
}

void MainWindow::onClearHistory()
{
    QMessageBox box(this);
    box.setWindowTitle(QStringLiteral("删除历史上下车记录"));
    box.setIcon(QMessageBox::Warning);
    box.setText(QStringLiteral(
        "将删除本机保存的历史上下车/进出记录：\n"
        "  • 事件日志 visitor_events.csv\n"
        "  • 访客记录 visitor_records.json\n"
        "  • 界面访客表格、上车/下车计数、识别时间线\n\n"
        "可选择是否同时删除抓拍图片（ReID 进出抓拍、访客帧）。\n"
        "注意：待上行的离线积压（uplink_spool）不会被删除。"));
    QPushButton *allBtn = box.addButton(QStringLiteral("删除记录+抓拍图片"),
                                        QMessageBox::AcceptRole);
    QPushButton *recBtn = box.addButton(QStringLiteral("仅删除记录"),
                                        QMessageBox::ActionRole);
    box.addButton(QStringLiteral("取消"), QMessageBox::RejectRole);
    box.exec();

    if (box.clickedButton() == allBtn) {
        QMessageBox::information(this, QStringLiteral("删除完成"),
                                 clearHistoryRecords(true));
    } else if (box.clickedButton() == recBtn) {
        QMessageBox::information(this, QStringLiteral("删除完成"),
                                 clearHistoryRecords(false));
    }
}

QString MainWindow::clearHistoryRecords(bool deleteImages)
{
    LOG_INFO(QString("[MainWindow] 删除历史上下车记录开始 (deleteImages=%1)")
                 .arg(deleteImages ? 1 : 0));

    // 1) 内存状态：访客、映射、淡出标签、上下车计数、识别时间线
    m_visitors.clear();
    m_trackToVisitor.clear();
    m_fadingLabels.clear();
    m_boardingCount = 0;
    m_alightingCount = 0;
    m_nextVisitorId = 1;
    if (m_zoneMonitor) m_zoneMonitor->reset();
    if (m_aggregator) m_aggregator->reset();
    if (m_registry) m_registry->reset();

    // 2) 界面访客表格
    if (ui->tableVisitors) ui->tableVisitors->setRowCount(0);

    // 3) 磁盘记录文件
    QString logDir = m_config.visitorLogPath;
    if (!QDir(logDir).isAbsolute())
        logDir = QCoreApplication::applicationDirPath() + "/" + logDir;
    int filesRemoved = 0;
    const QStringList recordFiles{logDir + "/visitor_events.csv",
                                  logDir + "/visitor_records.json",
                                  logDir + "/person_records.csv"};
    for (const QString &f : recordFiles) {
        if (QFile::exists(f) && QFile::remove(f)) filesRemoved++;
    }

    // 4) 抓拍图片（可选）
    int dirsRemoved = 0;
    if (deleteImages) {
        // 4.1 ReID 进出抓拍：save_images/reid/person_*
        QString reidDir = m_config.reidSavePath;
        if (!QDir(reidDir).isAbsolute())
            reidDir = QCoreApplication::applicationDirPath() + "/" + reidDir;
        if (QDir(reidDir).exists()) {
            QDir rd(reidDir);
            const QStringList personDirs =
                rd.entryList(QStringList() << "person_*",
                             QDir::Dirs | QDir::NoDotAndDotDot);
            for (const QString &d : personDirs) {
                if (QDir(reidDir + "/" + d).removeRecursively()) dirsRemoved++;
            }
        }
        // 4.2 访客帧：save_images/<date>/visitor_V*
        QString saveDir = m_config.savePath;
        if (!QDir(saveDir).isAbsolute())
            saveDir = QCoreApplication::applicationDirPath() + "/" + saveDir;
        if (QDir(saveDir).exists()) {
            QDir sd(saveDir);
            const QStringList dateDirs =
                sd.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
            for (const QString &dateDir : dateDirs) {
                QDir dd(saveDir + "/" + dateDir);
                const QStringList vDirs =
                    dd.entryList(QStringList() << "visitor_V*",
                                 QDir::Dirs | QDir::NoDotAndDotDot);
                for (const QString &v : vDirs) {
                    if (QDir(saveDir + "/" + dateDir + "/" + v).removeRecursively())
                        dirsRemoved++;
                }
            }
        }
    }

    // 5) 刷新界面
    updateVisitorTable();
    if (ui->label_personCount)
        ui->label_personCount->setText(QStringLiteral("区域人数: 0 | 上车: 0 | 下车: 0"));

    LOG_INFO(QString("[MainWindow] 历史删除完成: 记录文件=%1, 抓拍目录=%2")
                 .arg(filesRemoved).arg(dirsRemoved));

    return QStringLiteral("已删除记录文件 %1 个，抓拍目录 %2 个。")
        .arg(filesRemoved).arg(dirsRemoved);
}

MainWindow::~MainWindow()
{
    if (m_isCapturing) {
        onCloseCamera();
    }

    // 停止客流上行（未发出的报文保留在本地 spool，下次启动续传）
    if (m_uplink) {
        m_uplink->stop();
    }

    // 停止访客记录自动保存定时器
    if (m_visitorSaveTimer) {
        m_visitorSaveTimer->stop();
    }

    // 保存访客记录
    saveVisitorRecords();
    if (m_visitorLog) {
        m_visitorLog->logSessionEnd();
    }

    if (m_saveThread) { m_saveThread->quit(); m_saveThread->wait(); }
    QCoreApplication::processEvents();

    delete m_visitorLog;
    m_visitorLog = nullptr;
    delete ui;
}

void MainWindow::initSaveWorker()
{
    m_saveWorker = new SaveImageWorker();
    m_saveThread = new QThread(this);
    m_saveWorker->moveToThread(m_saveThread);
    connect(m_saveThread, &QThread::finished, m_saveWorker, &QObject::deleteLater);
    m_saveThread->start();
    m_saveWorker->setSaveParams(m_config.savePath, m_config.saveFormat, m_config.jpegQuality);
}

void MainWindow::onToggleCapture()
{
    if (m_isCapturing) {
        // 当前正在采集 → 停止采集
        onCloseCamera();
        m_isCapturing = false;
        ui->btnToggle->setText("启动采集");
        ui->btnSave->setEnabled(false);
    } else {
        // 当前未采集 → 启动采集
        LOG_INFO("============================================================");
        LOG_INFO("[MainWindow] [BTN] 启动采集按钮点击 - onToggleCapture START");
        LOG_INFO("============================================================");

        if (!m_camera.openCamera(m_config.rtspUrl)) {
            QMessageBox::critical(this, "错误", "无法打开 RTSP 流:\n" + m_config.rtspUrl);
            return;
        }
        LOG_INFO("Camera opened: " + m_config.rtspUrl);

        // 创建采集线程
        m_grabThread = new GrabThread(&m_camera, "HX-CCD23", this);

        // 创建检测管线
        m_detector = new PersonDetector();
        m_tracker = new PersonTracker(m_config.trackerIouThreshold, m_config.trackerMaxMissed);
        m_tracker->setKalmanEnabled(m_config.trackerKalmanEnabled);
        m_tracker->setReidMatchingEnabled(m_config.trackerReidEnabled);
        m_tracker->setReidMatchThreshold(m_config.trackerReidThreshold);
        m_tracker->setGalleryMaxAge(m_config.trackerGalleryMaxAgeMs);
        // 共享身份注册表：跟踪器负责「一人一 ID」的去重与身份解析
        m_tracker->setVisitorRegistry(m_registry);
        m_tracker->setTrackMergeSimilarity(m_config.trackerMergeSimilarity);
        m_tracker->setTrackMergeMaxYoungAge(m_config.trackerMergeMaxYoungAge);
        m_reidExtractor = new ReIDExtractor();

        // 加载 YOLO 模型（优先 .engine，回退 .onnx）
        QString modelPath = m_config.modelPath;
        if (!QFileInfo(modelPath).isAbsolute()) {
            modelPath = QCoreApplication::applicationDirPath() + "/" + modelPath;
        }

        bool modelLoaded = m_detector->loadModel(modelPath, m_config.detectorConfThreshold, m_config.detectorNmsThreshold);

        // 如果 .engine 加载失败，尝试回退到 .onnx
        if (!modelLoaded && modelPath.endsWith(".engine", Qt::CaseInsensitive)) {
            QString onnxPath = modelPath.left(modelPath.lastIndexOf('.')) + ".onnx";
            if (QFileInfo::exists(onnxPath)) {
                LOG_WARN("[MainWindow] Engine load failed, falling back to ONNX: " + onnxPath);
                modelLoaded = m_detector->loadModel(onnxPath, m_config.detectorConfThreshold, m_config.detectorNmsThreshold);
                if (modelLoaded) modelPath = onnxPath;
            }
        }

        if (modelLoaded) {
            QString backend = m_detector->isTrtBackend() ? "TensorRT" : "OpenCV DNN";
            LOG_INFO(QString("[MainWindow] YOLO model loaded [%1]: %2").arg(backend, modelPath));
        } else {
            LOG_WARN("[MainWindow] Failed to load any YOLO model (detection disabled)");
        }

        // 设置检测管线到采集线程
        m_grabThread->setDetector(m_detector);
        m_grabThread->setTracker(m_tracker);
        m_grabThread->setReidExtractor(m_reidExtractor);

        // 连接信号
        connect(m_grabThread, &GrabThread::deviceLost, this, &MainWindow::onGrabThreadDeviceLost);
        connect(m_grabThread, &GrabThread::frameProcessed, this, &MainWindow::onFrameProcessed);
        connect(m_grabThread, &GrabThread::frameGrabbed, this, &MainWindow::onFrameGrabbed);
        connect(m_grabThread, &GrabThread::rawFrameGrabbed, this, [this](const cv::Mat &frame) {
            QMutexLocker locker(&m_rawFrameMutex);
            m_currentRawFrame = frame.clone();
        });

        m_grabThread->start();
        LOG_INFO("[MainWindow] GrabThread started with detection pipeline");

        // 配置身份注册表（外观提取器）与通道监控器
        if (m_registry) {
            m_registry->setExtractor(m_reidExtractor);
            m_registry->setBaseThreshold(m_config.reidMatchThreshold);
        }
        if (m_zoneMonitor) {
            m_zoneMonitor->setReidExtractor(m_reidExtractor);
            m_zoneMonitor->setReidMatchThreshold(m_config.reidMatchThreshold);
            m_zoneMonitor->setChannelZones(m_regionManager->regions());
            LOG_INFO("[MainWindow] ZoneMonitor configured with ReID extractor");
        }

        // 重置 FPS 计算
        m_fpsTimer.restart();
        m_fpsFrameCount = 0;
        m_currentFps = 0.0;

        m_isCapturing = true;
        ui->btnToggle->setText("停止采集");
        ui->btnSave->setEnabled(true);
        updateCameraStatus(true);
        m_elapsedTimer.start();
    }

    updateCaptureStatus();
}

void MainWindow::updateCaptureStatus()
{
    if (!m_isCapturing) {
        ui->captureStatusLabel->setText("状态: 已停止");
        ui->captureStatusLabel->setStyleSheet("color: #b0bec5; font-family: 'Consolas'; font-size: 12px; font-weight: bold; padding: 0 8px;");
    } else if (m_regionManager && !m_regionManager->regions().isEmpty()) {
        ui->captureStatusLabel->setText("状态: 已启动 - 区域监控中");
        ui->captureStatusLabel->setStyleSheet("color: #00e676; font-family: 'Consolas'; font-size: 12px; font-weight: bold; padding: 0 8px;");
    } else {
        ui->captureStatusLabel->setText("状态: 已启动");
        ui->captureStatusLabel->setStyleSheet("color: #00e676; font-family: 'Consolas'; font-size: 12px; font-weight: bold; padding: 0 8px;");
    }
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    LOG_INFO("[MainWindow] closeEvent triggered");

    // 保存区域
    {
        QString regionPath = m_config.regionSavePath;
        if (!QDir(regionPath).isAbsolute()) {
            regionPath = QCoreApplication::applicationDirPath() + "/" + regionPath;
        }
        m_regionManager->saveRegions(regionPath);
    }

    // 保存访客记录
    saveVisitorRecords();
    if (m_visitorLog) {
        m_visitorLog->logSessionEnd();
    }

    if (m_isCapturing) {
        onCloseCamera();
    }
    QMainWindow::closeEvent(event);
}

void MainWindow::onCloseCamera()
{
    LOG_INFO("============================================================");
    LOG_INFO("[MainWindow] [BTN] 停止采集按钮点击 - onCloseCamera START");
    LOG_INFO("============================================================");

    if (m_grabThread) {
        m_grabThread->stop();
        m_grabThread->wait();
        delete m_grabThread;
        m_grabThread = nullptr;
    }

    // 清理检测管线
    delete m_detector;
    m_detector = nullptr;
    delete m_tracker;
    m_tracker = nullptr;
    delete m_reidExtractor;
    m_reidExtractor = nullptr;

    m_camera.closeCamera();
    QCoreApplication::processEvents();

    LOG_INFO("[MainWindow] onCloseCamera END");

    updateCameraStatus(false);
    m_elapsedTimer.invalidate();
    ui->runTimeLabel->setText("运行: --:--:--");
    ui->runTimeLabel->setStyleSheet("color: #ffd54f; font-family: 'Consolas'; font-size: 12px; font-weight: bold;");

    // 清理淡出标签
    m_fadingLabels.clear();
}

void MainWindow::onFrameGrabbed(QImage img)
{
    // 更新 FPS
    m_fpsFrameCount++;
    qint64 fpsElapsed = m_fpsTimer.elapsed();
    if (fpsElapsed >= 1000) {
        m_currentFps = m_fpsFrameCount * 1000.0 / fpsElapsed;
        m_fpsFrameCount = 0;
        m_fpsTimer.restart();
    }

    // 节流 UI 更新
    const bool shouldRender = m_lastUiUpdate.elapsed() >= UI_UPDATE_INTERVAL_MS;
    if (shouldRender) {
        if (m_imageView) m_imageView->setImage(img);
        m_lastUiUpdate.restart();
    }

    // 更新 FPS 显示
    ui->label_fps->setText(QString("FPS: %1").arg(m_currentFps, 0, 'f', 1));

    // 保存当前帧副本（用于手动保存）
    {
        QMutexLocker locker(&m_frameMutex);
        m_currentFrame = img;
    }

    // 自动保存
    if (m_autoSaveEnabled && m_saveWorker && !m_diskFull) {
        QMetaObject::invokeMethod(m_saveWorker, "onSaveImage",
                                  Qt::QueuedConnection,
                                  Q_ARG(QImage, img));
    }
}

void MainWindow::updateFadingLabels(const QVector<TrackedPerson> &persons)
{
    QDateTime now = QDateTime::currentDateTime();

    // 收集当前活跃的 trackId 集合
    QSet<int> activeTrackIds;
    for (const auto &p : persons) {
        activeTrackIds.insert(p.trackId);
    }

    // 更新淡出标签：活跃的更新位置和时间，不活跃的更新 alpha
    QMap<int, FadingLabel>::iterator it = m_fadingLabels.begin();
    while (it != m_fadingLabels.end()) {
        int tid = it.key();
        FadingLabel &label = it.value();

        if (activeTrackIds.contains(tid)) {
            // track 仍然活跃，更新位置和时间，重置 alpha
            for (const auto &p : persons) {
                if (p.trackId == tid) {
                    label.lastBbox = p.bbox;
                    label.lastSeen = now;
                    label.alpha = 1.0f;
                    // 更新 visitorId
                    if (m_trackToVisitor.contains(tid)) {
                        label.visitorId = m_trackToVisitor[tid];
                    }
                    break;
                }
            }
            ++it;
        } else {
            // track 不再活跃，计算淡出 alpha
            float elapsed = label.lastSeen.msecsTo(now);
            if (elapsed >= FADING_DURATION_MS) {
                // 淡出完成，移除
                it = m_fadingLabels.erase(it);
            } else {
                label.alpha = 1.0f - (elapsed / FADING_DURATION_MS);
                ++it;
            }
        }
    }

    // 为新出现的活跃 track 创建淡出标签
    for (const auto &p : persons) {
        if (!m_fadingLabels.contains(p.trackId)) {
            FadingLabel fl;
            fl.trackId = p.trackId;
            fl.visitorId = (p.visitorId > 0)
                               ? p.visitorId
                               : m_trackToVisitor.value(p.trackId, -1);
            fl.lastBbox = p.bbox;
            fl.lastSeen = now;
            fl.alpha = 1.0f;
            m_fadingLabels[p.trackId] = fl;
        }
    }
}

void MainWindow::drawActiveIdOnCvMat(cv::Mat &frame, const TrackedPerson &person)
{
    if (frame.empty()) return;

    // 防御：跟踪框可能越界/退化（例如卡尔曼预测出框产生负坐标或高宽极小），
    // 与画面无交集时直接跳过，避免空 ROI 触发 OpenCV addWeighted 尺寸异常。
    const cv::Rect frameRect(0, 0, frame.cols, frame.rows);
    cv::Rect safeBBox = person.bbox & frameRect;
    if (safeBBox.width <= 0 || safeBBox.height <= 0) return;

    // ID 标签格式：优先显示稳定身份 visitorId（同一人始终同一个 V 号），
    // 否则退回 trackId
    int displayId = (person.visitorId > 0)
                        ? person.visitorId
                        : m_trackToVisitor.value(person.trackId, person.trackId);
    std::string label = "V" + std::to_string(displayId);

    // 位置：边界框左上方
    int x = safeBBox.x;
    int y = safeBBox.y - 10;
    if (y < 20) y = safeBBox.y + 20;

    double fontScale = 0.8;
    int thickness = 2;
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);

    // 绘制背景矩形（半透明黑色）
    cv::Rect bgRect(x - 2, y - textSize.height - 4, textSize.width + 8, textSize.height + 8);
    // 确保不超出图像范围
    bgRect &= frameRect;
    if (bgRect.width > 0 && bgRect.height > 0) {
        cv::Mat roi = frame(bgRect);
        if (!roi.empty()) {
            cv::Mat overlay = roi.clone();
            cv::rectangle(overlay, cv::Point(0, 0), cv::Point(roi.cols, roi.rows),
                          cv::Scalar(0, 0, 0), -1);
            cv::addWeighted(overlay, 0.6, roi, 0.4, 0, roi);
        }
    }

    // 根据是否在区域内选择颜色
    cv::Scalar textColor = person.isInsideRegion ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 255, 255);

    // 黑色描边（偏移 1px）
    cv::putText(frame, label, cv::Point(x + 1, y + 1),
                cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 0, 0), thickness + 1);
    // 彩色文本
    cv::putText(frame, label, cv::Point(x, y),
                cv::FONT_HERSHEY_SIMPLEX, fontScale, textColor, thickness);
}

void MainWindow::drawFadingIdOnCvMat(cv::Mat &frame, const FadingLabel &label)
{
    if (frame.empty() || label.alpha <= 0.01f) return;

    // ID 标签格式
    int displayId = (label.visitorId > 0) ? label.visitorId : label.trackId;
    std::string text = "V" + std::to_string(displayId);

    int x = label.lastBbox.x;
    int y = label.lastBbox.y - 10;
    if (y < 20) y = label.lastBbox.y + 20;

    double fontScale = 0.8;
    int thickness = 2;
    float alpha = label.alpha;

    // 使用 alpha 混合绘制文本
    cv::Mat textLayer = cv::Mat::zeros(frame.size(), frame.type());
    cv::putText(textLayer, text, cv::Point(x + 1, y + 1),
                cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 0, 0), thickness + 1);
    cv::putText(textLayer, text, cv::Point(x, y),
                cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 255, 255), thickness);

    // 直接使用 alpha 混合
    cv::addWeighted(textLayer, alpha, frame, 1.0, 0, frame);
}

// ===== 方向感知的穿越检测 =====
bool MainWindow::detectDirectionalEntry(const TrackedPerson &person, const DetectionRegion &region, int imgH)
{
    if (imgH <= 0) return false;

    // 计算区域的边界框（归一化坐标）
    float minY = 1.0f, maxY = 0.0f;
    for (const QPointF &pt : region.polygon) {
        if (pt.y() < minY) minY = pt.y();
        if (pt.y() > maxY) maxY = pt.y();
    }

    // 当前中心的归一化 Y 坐标
    float currNormY = person.center.y / imgH;
    // 前一帧中心的归一化 Y 坐标
    float prevNormY = person.prevCenter.y / imgH;

    if (region.entryDirection == EntryDirection::TopToBottom) {
        // 从上往下：前一帧在区域上方边界之外，当前帧在区域内或下方
        // 入口条件：前一帧的 Y < 区域顶部，当前帧的 Y >= 区域顶部
        return (prevNormY < minY) && (currNormY >= minY);
    } else {
        // 从下往上：前一帧在区域下方边界之外，当前帧在区域内或上方
        // 入口条件：前一帧的 Y > 区域底部，当前帧的 Y <= 区域底部
        return (prevNormY > maxY) && (currNormY <= maxY);
    }
}

// ===== ReID 匹配或创建新访客 =====
int MainWindow::matchOrCreateVisitor(const TrackedPerson &person, const cv::Mat &rawFrame,
                                     int regionId, int imgW, int imgH)
{
    // 已有映射：同一 track 直接复用，避免重复解析
    if (m_trackToVisitor.contains(person.trackId)) {
        return m_trackToVisitor[person.trackId];
    }

    // 提取 ReID 特征
    cv::Mat reidFeature = person.reidFeature;
    if (reidFeature.empty() && !rawFrame.empty() && m_reidExtractor) {
        reidFeature = m_reidExtractor->extract(rawFrame, person.bbox);
    }

    // ===== 稳定身份解析（同一人永远只有一个 ID）=====
    // (1) 优先采用跟踪器已解析出的全局身份（跨轨迹唯一，与上下车 ID 完全一致）
    // (2) 否则退回注册表解析（自适应阈值 + 时空门限）
    // (3) 最后才退回本地编号（未设置注册表时）
    cv::Point2f normPos(0.5f, 0.5f);
    if (imgW > 0 && imgH > 0) {
        normPos = cv::Point2f(person.center.x / imgW, person.center.y / imgH);
    }

    int visitorId = person.visitorId;
    if (visitorId > 0) {
        // 用当前帧特征继续强化该身份的画廊
        if (m_registry && !reidFeature.empty()) {
            m_registry->reinforce(visitorId, reidFeature,
                                  QDateTime::currentMSecsSinceEpoch(), normPos);
        }
    } else if (m_registry) {
        visitorId = m_registry->resolve(reidFeature,
                                        QDateTime::currentMSecsSinceEpoch(), normPos);
    }
    if (visitorId <= 0) {
        visitorId = m_nextVisitorId++;   // 回退：无注册表/无特征时本地编号
    }

    const bool reentry = m_visitors.contains(visitorId);
    float similarity = 0.0f;
    if (reentry && m_registry) {
        similarity = m_registry->similarity(reidFeature, m_registry->feature(visitorId));
        if (similarity < 0.0f) similarity = 0.0f;
    }

    VisitorRecord &vr = m_visitors[visitorId];
    vr.visitorId = visitorId;
    vr.currentTrackId = person.trackId;
    vr.regionId = regionId;
    if (!vr.isInside) {
        // 新的一次进出周期：重置穿越结果，等待 ZoneMonitor 判定方向后回填
        vr.enterTime = QDateTime::currentDateTime();
        vr.exitTime = QDateTime();
        vr.enterDirection.clear();
        vr.exitDirection.clear();
        vr.direction.clear();
        vr.entrySide = -1;
        vr.exitSide = -1;
        vr.dwellSeconds = 0;
        vr.loggedCrossingMs = 0;
    }
    vr.isInside = true;
    if (!reidFeature.empty()) vr.reidFeature = reidFeature.clone();
    if (!rawFrame.empty()) vr.enterImage = rawFrame.clone();

    m_trackToVisitor[person.trackId] = visitorId;
    if (m_registry) m_registry->markInside(visitorId, true);

    if (reentry) {
        LOG_INFO(QString("[MainWindow] ReID RE-ENTRY: V%1 (track %2, sim=%3)")
                     .arg(visitorId).arg(person.trackId).arg(similarity, 0, 'f', 3));
        if (m_visitorLog) {
            VisitorLog::EventEntry evt;
            evt.timestamp = QDateTime::currentDateTime();
            evt.eventType = "REENTRY";
            evt.visitorId = visitorId;
            evt.trackId = person.trackId;
            evt.regionId = regionId;
            evt.similarity = similarity;
            evt.details = QStringLiteral("ReID re-entry match");
            m_visitorLog->logEvent(evt);
        }
    } else {
        LOG_INFO(QString("[MainWindow] New visitor V%1 created (track %2)")
                     .arg(visitorId).arg(person.trackId));
        if (m_visitorLog) {
            VisitorLog::EventEntry evt;
            evt.timestamp = QDateTime::currentDateTime();
            evt.eventType = "ENTER";
            evt.visitorId = visitorId;
            evt.trackId = person.trackId;
            evt.regionId = regionId;
            evt.similarity = 0.0f;
            evt.details = QStringLiteral("New visitor created");
            m_visitorLog->logEvent(evt);
        }
    }

    return visitorId;
}

void MainWindow::updateVisitorExit(int visitorId, const cv::Mat &exitImage)
{
    if (!m_visitors.contains(visitorId)) return;

    VisitorRecord &vr = m_visitors[visitorId];
    vr.isInside = false;
    vr.exitTime = QDateTime::currentDateTime();
    if (!exitImage.empty()) {
        vr.exitImage = exitImage.clone();
    }
    // 停留时长（若已由穿越事件写入，则以穿越事件的精确边界为准）
    if (vr.enterTime.isValid())
        vr.dwellSeconds = vr.enterTime.secsTo(vr.exitTime);
    const int exitedTrackId = vr.currentTrackId;
    vr.currentTrackId = -1;

    // 更新 ReID 特征（离开时保留最新特征用于后续匹配）
    // 特征已在进入时更新

    LOG_INFO(QString("[MainWindow] Visitor V%1 EXITED at %2 (dwell=%3s)")
                 .arg(visitorId).arg(vr.exitTime.toString("HH:mm:ss.zzz"))
                 .arg(vr.dwellSeconds));

    // 记录离开事件
    if (m_visitorLog) {
        VisitorLog::EventEntry evt;
        evt.timestamp = QDateTime::currentDateTime();
        evt.eventType = "EXIT";
        evt.visitorId = visitorId;
        evt.trackId = exitedTrackId;
        evt.regionId = vr.regionId;
        evt.similarity = 0.0f;
        evt.details = QString("Visitor exited region");
        m_visitorLog->logEvent(evt);
    }
}

void MainWindow::updateVisitorTable()
{
    if (!ui->tableVisitors) return;

    // 阻止信号避免闪烁
    ui->tableVisitors->blockSignals(true);

    int row = 0;
    for (auto it = m_visitors.constBegin(); it != m_visitors.constEnd(); ++it) {
        const VisitorRecord &vr = it.value();

        if (row >= ui->tableVisitors->rowCount()) {
            ui->tableVisitors->insertRow(row);
        }

        // 单元格获取/创建
        auto cell = [&](int col) -> QTableWidgetItem * {
            QTableWidgetItem *item = ui->tableVisitors->item(row, col);
            if (!item) {
                item = new QTableWidgetItem();
                ui->tableVisitors->setItem(row, col, item);
            }
            item->setTextAlignment(Qt::AlignCenter);
            return item;
        };

        // 方向 → 文本与颜色
        auto dirStyle = [](const QString &dir, QString *text, QColor *color) {
            if (dir == "boarding") {
                *text = QStringLiteral("↑上车");
                *color = QColor(0, 200, 255);
            } else if (dir == "alighting") {
                *text = QStringLiteral("↓下车");
                *color = QColor(255, 171, 64);
            } else if (dir == "turnback") {
                *text = QStringLiteral("折返");
                *color = QColor(186, 104, 200);
            } else {
                *text = QStringLiteral("--");
                *color = QColor(120, 120, 120);
            }
        };

        // 0 ID
        cell(0)->setText(QString("V%1").arg(vr.visitorId));

        // 1 进入时间
        cell(1)->setText(vr.enterTime.isValid() ? vr.enterTime.toString("HH:mm:ss") : "--");

        // 2 进入方向
        {
            QString text;
            QColor color;
            dirStyle(vr.enterDirection, &text, &color);
            QTableWidgetItem *item = cell(2);
            item->setText(text);
            item->setForeground(QBrush(color));
        }

        // 3 离开时间
        cell(3)->setText(vr.exitTime.isValid() ? vr.exitTime.toString("HH:mm:ss") : "--");

        // 4 离开方向
        {
            QString text;
            QColor color;
            dirStyle(vr.exitDirection, &text, &color);
            QTableWidgetItem *item = cell(4);
            item->setText(text);
            item->setForeground(QBrush(color));
        }

        // 5 停留时长（秒 → mm:ss；未离开则实时累加）
        {
            QTableWidgetItem *item = cell(5);
            if (vr.enterTime.isValid()) {
                const QDateTime endTime = vr.exitTime.isValid() ? vr.exitTime
                                                                : QDateTime::currentDateTime();
                const qint64 secs = vr.enterTime.secsTo(endTime);
                item->setText(QString("%1:%2")
                                  .arg(secs / 60, 2, 10, QChar('0'))
                                  .arg(secs % 60, 2, 10, QChar('0')));
            } else {
                item->setText("--");
            }
        }

        // 6 状态
        {
            QTableWidgetItem *item = cell(6);
            if (vr.isInside) {
                item->setText(QStringLiteral("● 在区域内"));
                item->setForeground(QBrush(QColor(0, 230, 118)));
            } else {
                item->setText(QStringLiteral("○ 已离开"));
                item->setForeground(QBrush(QColor(176, 190, 197)));
            }
        }

        row++;
    }

    // 移除多余的行
    while (ui->tableVisitors->rowCount() > row) {
        ui->tableVisitors->removeRow(ui->tableVisitors->rowCount() - 1);
    }

    ui->tableVisitors->blockSignals(false);
}

QString MainWindow::directionText(const QString &direction)
{
    if (direction == QLatin1String("boarding"))  return QStringLiteral("上车");
    if (direction == QLatin1String("alighting")) return QStringLiteral("下车");
    if (direction == QLatin1String("turnback"))  return QStringLiteral("折返");
    return QStringLiteral("--");
}

void MainWindow::applyCrossingRecord(int visitorId, const ZoneEvent &evt,
                                     const QString &direction)
{
    if (visitorId <= 0) {
        LOG_WARN(QString("[MainWindow] crossing record skipped: invalid personId (%1)")
                     .arg(visitorId));
        return;
    }

    const QDateTime enterTime = evt.enterTime.isValid() ? evt.enterTime
                                                       : QDateTime::currentDateTime();
    const QDateTime exitTime = evt.exitTime.isValid() ? evt.exitTime
                                                     : QDateTime::currentDateTime();
    const qint64 dwell = enterTime.secsTo(exitTime);

    // ===== 1) 更新「同一 ID」的人员记录 =====
    VisitorRecord &vr = m_visitors[visitorId];
    vr.visitorId = visitorId;
    vr.enterTime = enterTime;
    vr.exitTime = exitTime;
    vr.entrySide = evt.entrySide;
    vr.exitSide = evt.exitSide;
    vr.enterDirection = direction;
    vr.exitDirection = direction;
    vr.direction = direction;
    vr.dwellSeconds = dwell;
    vr.isInside = false;
    vr.regionId = evt.regionId;
    if (!evt.enterImage.empty()) vr.enterImage = evt.enterImage.clone();
    if (!evt.exitImage.empty()) vr.exitImage = evt.exitImage.clone();

    // 关联映射：确保随后同一轨迹的查询能拿到该身份
    if (evt.trackId > 0) {
        const int mapped = m_trackToVisitor.value(evt.trackId, -1);
        if (mapped <= 0) m_trackToVisitor[evt.trackId] = visitorId;
    }
    if (m_registry) m_registry->markInside(visitorId, false);

    LOG_INFO(QString("[MainWindow] 进出记录 V%1: %2 | 进入 %3(%4) | 离开 %5(%6) | 停留 %7s")
                 .arg(visitorId)
                 .arg(directionText(direction))
                 .arg(enterTime.toString("HH:mm:ss.zzz"))
                 .arg(evt.entrySide == 0 ? "A端" : (evt.entrySide == 1 ? "B端" : "-"))
                 .arg(exitTime.toString("HH:mm:ss.zzz"))
                 .arg(evt.exitSide == 0 ? "A端" : (evt.exitSide == 1 ? "B端" : "-"))
                 .arg(dwell));

    // ===== 2) 写入 person_records.csv（同一穿越只写一次，防重复）=====
    const qint64 crossingKey = exitTime.toMSecsSinceEpoch();
    if (m_visitorLog && crossingKey != vr.loggedCrossingMs) {
        vr.loggedCrossingMs = crossingKey;

        VisitorLog::PersonRecordEntry rec;
        rec.visitorId = visitorId;
        rec.enterTime = enterTime.toString(Qt::ISODate);
        rec.enterDirection = direction;
        rec.entrySide = evt.entrySide;
        rec.exitTime = exitTime.toString(Qt::ISODate);
        rec.exitDirection = direction;
        rec.exitSide = evt.exitSide;
        rec.dwellSeconds = dwell;
        rec.regionId = evt.regionId;
        m_visitorLog->logPersonRecord(rec);
    }

    // ===== 3) 事件日志 =====
    if (m_visitorLog) {
        VisitorLog::EventEntry logEntry;
        logEntry.timestamp = exitTime;
        logEntry.eventType = (direction == QLatin1String("boarding"))
                                 ? QStringLiteral("BOARDING")
                                 : (direction == QLatin1String("alighting"))
                                       ? QStringLiteral("ALIGHTING")
                                       : QStringLiteral("TURNBACK");
        logEntry.visitorId = visitorId;
        logEntry.trackId = evt.trackId;
        logEntry.regionId = evt.regionId;
        logEntry.similarity = 0.0f;
        logEntry.details = QString("%1(%2→%3), enter=%4, exit=%5, dwell=%6s")
                               .arg(directionText(direction))
                               .arg(evt.entrySide == 0 ? "A" : (evt.entrySide == 1 ? "B" : "-"))
                               .arg(evt.exitSide == 0 ? "A" : (evt.exitSide == 1 ? "B" : "-"))
                               .arg(enterTime.toString("HH:mm:ss.zzz"))
                               .arg(exitTime.toString("HH:mm:ss.zzz"))
                               .arg(dwell);
        m_visitorLog->logEvent(logEntry);
    }

    // ===== 4) 刷新记录表 =====
    updateVisitorTable();
}

void MainWindow::drawDirectionArrowOnFrame(cv::Mat &frame, const DetectionRegion &region, int imgW, int imgH)
{
    if (frame.empty() || region.polygon.size() < 3) return;

    // 计算区域中心点（像素坐标）
    float cx = 0, cy = 0;
    for (const QPointF &pt : region.polygon) {
        cx += pt.x() * imgW;
        cy += pt.y() * imgH;
    }
    cx /= region.polygon.size();
    cy /= region.polygon.size();

    cv::Point center(static_cast<int>(cx), static_cast<int>(cy));

    // 绘制方向箭头
    cv::Scalar arrowColor(0, 200, 255); // 青色
    int arrowLen = 30;

    if (region.entryDirection == EntryDirection::TopToBottom) {
        // 向下箭头 ↓
        cv::Point tip(center.x, center.y + arrowLen);
        cv::Point left(center.x - 10, center.y + arrowLen - 15);
        cv::Point right(center.x + 10, center.y + arrowLen - 15);
        cv::line(frame, cv::Point(center.x, center.y - arrowLen), tip, arrowColor, 3);
        cv::line(frame, left, tip, arrowColor, 3);
        cv::line(frame, right, tip, arrowColor, 3);
        cv::putText(frame, "Top->Bottom", cv::Point(center.x - 40, center.y - arrowLen - 8),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, arrowColor, 1);
    } else {
        // 向上箭头 ↑
        cv::Point tip(center.x, center.y - arrowLen);
        cv::Point left(center.x - 10, center.y - arrowLen + 15);
        cv::Point right(center.x + 10, center.y - arrowLen + 15);
        cv::line(frame, cv::Point(center.x, center.y + arrowLen), tip, arrowColor, 3);
        cv::line(frame, left, tip, arrowColor, 3);
        cv::line(frame, right, tip, arrowColor, 3);
        cv::putText(frame, "Bottom->Top", cv::Point(center.x - 40, center.y + arrowLen + 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, arrowColor, 1);
    }
}

void MainWindow::onFrameProcessed(QImage img, QVector<TrackedPerson> persons)
{
    // 更新 FPS
    m_fpsFrameCount++;
    qint64 fpsElapsed = m_fpsTimer.elapsed();
    if (fpsElapsed >= 1000) {
        m_currentFps = m_fpsFrameCount * 1000.0 / fpsElapsed;
        m_fpsFrameCount = 0;
        m_fpsTimer.restart();
    }

    // 获取当前原始帧
    cv::Mat rawFrame;
    {
        QMutexLocker locker(&m_rawFrameMutex);
        rawFrame = m_currentRawFrame.clone();
    }

    int imgW = img.width();
    int imgH = img.height();

    // ===== 通道监控：上车/下车方向检测 =====
    if (m_zoneMonitor && m_regionManager) {
        m_zoneMonitor->processFrame(persons, rawFrame, m_regionManager, imgW, imgH);
    }

    // ===== 状态机：方向感知的穿越检测 + ReID 匹配 =====
    // 确认帧数阈值：需要连续 N 帧在区域内/外才触发进入/离开事件
    static constexpr int CROSS_CONFIRM_FRAMES = 3;

    int personInRegionCount = 0;
    for (int i = 0; i < persons.size(); ++i) {
        TrackedPerson &p = persons[i];

        // 计算归一化中心点
        float normCx = 0.0f, normCy = 0.0f;
        if (imgW > 0 && imgH > 0) {
            normCx = p.center.x / imgW;
            normCy = p.center.y / imgH;
        }
        QPointF normCenter(normCx, normCy);

        bool nowInRegion = m_regionManager->isPointInAnyRegion(normCenter);
        int currentRegionId = m_regionManager->pointInWhichRegion(normCenter);

        // 获取跟踪器中持久化的状态
        TrackedPerson *trackerTrack = m_tracker ? m_tracker->getTrack(p.trackId) : nullptr;

        // 从跟踪器同步状态机状态到当前 persons 条目
        if (trackerTrack) {
            p.crossState = trackerTrack->crossState;
            p.crossConfirmFrames = trackerTrack->crossConfirmFrames;
            p.crossDirectionSign = trackerTrack->crossDirectionSign;
            p.hasEntered = trackerTrack->hasEntered;
            p.hasExited = trackerTrack->hasExited;
            p.enterTime = trackerTrack->enterTime;
            p.exitTime = trackerTrack->exitTime;
            p.visitorId = trackerTrack->visitorId;
            if (!trackerTrack->enterImage.empty()) p.enterImage = trackerTrack->enterImage.clone();
            if (!trackerTrack->exitImage.empty()) p.exitImage = trackerTrack->exitImage.clone();
        }

        // ===== 状态机驱动的进入/离开检测 =====
        switch (p.crossState) {
        case RegionCrossState::Outside: {
            if (nowInRegion) {
                // 从区域外进入区域内，开始进入确认
                p.crossState = RegionCrossState::Entering;
                p.crossConfirmFrames = 1;

                // 检测方向（用于日志和验证）
                bool directionOk = false;
                for (const auto &region : m_regionManager->regions()) {
                    if (m_regionManager->isPointInRegion(normCenter, region.id)) {
                        if (detectDirectionalEntry(p, region, imgH)) {
                            directionOk = true;
                            currentRegionId = region.id;
                            break;
                        }
                    }
                }
                // 如果方向检测不通过，也允许进入（回退模式）
                if (!directionOk) {
                    p.crossDirectionSign = 0; // 未确定方向
                }
            }
            break;
        }
        case RegionCrossState::Entering: {
            if (nowInRegion) {
                p.crossConfirmFrames++;
                if (p.crossConfirmFrames >= CROSS_CONFIRM_FRAMES) {
                    // 连续 N 帧确认在区域内 → 触发进入事件
                    p.crossState = RegionCrossState::Inside;
                    p.hasEntered = true;
                    p.hasExited = false; // 重置退出标记（支持重入）
                    p.enterTime = QDateTime::currentDateTime();

                    // ReID 匹配或创建新访客
                    int visitorId = matchOrCreateVisitor(p, rawFrame, currentRegionId, imgW, imgH);
                    p.visitorId = visitorId;

                    // 保存进入图像
                    if (!rawFrame.empty()) {
                        p.enterImage = rawFrame.clone();
                        saveReidImage(rawFrame, visitorId, "ENTER");
                    }

                    LOG_INFO(QString("[MainWindow] Visitor V%1 CONFIRMED ENTERED at %2 (track=%3, confirm=%4 frames)")
                                 .arg(visitorId)
                                 .arg(p.enterTime.toString("HH:mm:ss.zzz"))
                                 .arg(p.trackId)
                                 .arg(p.crossConfirmFrames));

                    // 更新区域人数
                    personInRegionCount++;
                }
                // 还在确认中，也计入区域人数
                personInRegionCount++;
            } else {
                // 确认期间离开区域 → 取消进入，回到 Outside
                p.crossState = RegionCrossState::Outside;
                p.crossConfirmFrames = 0;
            }
            break;
        }
        case RegionCrossState::Inside: {
            if (nowInRegion) {
                // 正常在区域内停留 —— 持续保存该 visitor 的帧
                personInRegionCount++;
                if (!rawFrame.empty() && p.visitorId > 0) {
                    saveVisitorFrame(rawFrame, p.bbox, p.visitorId);
                }
            } else {
                // 离开区域，开始离开确认
                p.crossState = RegionCrossState::Exiting;
                p.crossConfirmFrames = 1;
            }
            break;
        }
        case RegionCrossState::Exiting: {
            if (!nowInRegion) {
                p.crossConfirmFrames++;
                if (p.crossConfirmFrames >= CROSS_CONFIRM_FRAMES) {
                    // 连续 N 帧确认在区域外 → 触发离开事件
                    p.crossState = RegionCrossState::Outside;
                    p.hasExited = true;
                    p.exitTime = QDateTime::currentDateTime();

                    // 更新访客离开记录（优先使用稳定身份）
                    int visitorId = (p.visitorId > 0)
                                        ? p.visitorId
                                        : m_trackToVisitor.value(p.trackId, -1);
                    if (visitorId > 0) {
                        updateVisitorExit(visitorId, rawFrame);
                        p.visitorId = visitorId;
                    }

                    // 保存离开图像
                    if (!rawFrame.empty()) {
                        p.exitImage = rawFrame.clone();
                        saveReidImage(rawFrame, visitorId > 0 ? visitorId : p.trackId, "EXIT");
                    }

                    LOG_INFO(QString("[MainWindow] Visitor V%1 CONFIRMED EXITED at %2 (track=%3, confirm=%4 frames)")
                                 .arg(visitorId > 0 ? visitorId : p.trackId)
                                 .arg(p.exitTime.toString("HH:mm:ss.zzz"))
                                 .arg(p.trackId)
                                 .arg(p.crossConfirmFrames));

                    // 重置状态，允许重新进入
                    p.hasEntered = false;
                    p.crossConfirmFrames = 0;
                }
                // 还在确认中
            } else {
                // 确认期间重新进入区域 → 取消离开，回到 Inside
                p.crossState = RegionCrossState::Inside;
                p.crossConfirmFrames = 0;
                personInRegionCount++;
            }
            break;
        }
        }

        // 同步状态回跟踪器
        if (trackerTrack) {
            trackerTrack->isInsideRegion = nowInRegion;
            trackerTrack->crossState = p.crossState;
            trackerTrack->crossConfirmFrames = p.crossConfirmFrames;
            trackerTrack->crossDirectionSign = p.crossDirectionSign;
            trackerTrack->hasEntered = p.hasEntered;
            trackerTrack->hasExited = p.hasExited;
            trackerTrack->enterTime = p.enterTime;
            trackerTrack->exitTime = p.exitTime;
            trackerTrack->visitorId = p.visitorId;
            if (!p.enterImage.empty()) trackerTrack->enterImage = p.enterImage.clone();
            if (!p.exitImage.empty()) trackerTrack->exitImage = p.exitImage.clone();
        }
    }

    // 更新淡出标签
    updateFadingLabels(persons);

    // ===== 在 cv::Mat 上绘制叠加信息 =====
    QImage rgbImg = img.convertToFormat(QImage::Format_RGB888);
    cv::Mat cvFrame(rgbImg.height(), rgbImg.width(), CV_8UC3,
                    const_cast<uchar*>(rgbImg.bits()), rgbImg.bytesPerLine());
    cv::Mat displayFrame = cvFrame.clone();
    cv::cvtColor(displayFrame, displayFrame, cv::COLOR_RGB2BGR);

    // 1. 绘制区域多边形（半透明蓝色填充 + 绿色边框 + 方向箭头）
    auto regions = m_regionManager->regions();
    static int s_drawLogCounter = 0;
    if (s_drawLogCounter % 100 == 0 && !regions.isEmpty()) {
        LOG_INFO(QString("[MainWindow] Drawing %1 regions on frame %2x%3")
                     .arg(regions.size()).arg(imgW).arg(imgH));
        for (const auto &r : regions) {
            LOG_INFO(QString("[MainWindow]   %1: polygon.size=%2, first_norm=(%3,%4)")
                         .arg(r.name).arg(r.polygon.size())
                         .arg(r.polygon.isEmpty() ? -1.0 : r.polygon[0].x(), 0, 'f', 4)
                         .arg(r.polygon.isEmpty() ? -1.0 : r.polygon[0].y(), 0, 'f', 4));
        }
    }
    s_drawLogCounter++;

    for (const auto &region : regions) {
        if (region.polygon.size() < 3) {
            if (s_drawLogCounter <= 2) {
                LOG_WARN(QString("[MainWindow] Skipping region '%1': polygon.size=%2 < 3")
                             .arg(region.name).arg(region.polygon.size()));
            }
            continue;
        }

        std::vector<cv::Point> polyPoints;
        for (const QPointF &pt : region.polygon) {
            int px = static_cast<int>(pt.x() * imgW);
            int py = static_cast<int>(pt.y() * imgH);
            polyPoints.push_back(cv::Point(px, py));
        }

        // 半透明蓝色填充
        cv::Mat overlay = displayFrame.clone();
        cv::fillPoly(overlay, std::vector<std::vector<cv::Point>>{polyPoints},
                     cv::Scalar(255, 100, 0)); // BGR: blue
        cv::addWeighted(overlay, 0.2, displayFrame, 0.8, 0, displayFrame);

        // 绿色边框
        cv::polylines(displayFrame, std::vector<std::vector<cv::Point>>{polyPoints},
                      true, cv::Scalar(0, 255, 0), 2);

        // 绘制区域名称 + 方向标识
        if (!region.name.isEmpty()) {
            cv::Point namePos(polyPoints[0].x + 5, polyPoints[0].y + 15);
            QString dirStr = (region.entryDirection == EntryDirection::TopToBottom) ? " [↓]" : " [↑]";
            cv::putText(displayFrame, (region.name + dirStr).toStdString(), namePos,
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }

        // 绘制方向箭头
        drawDirectionArrowOnFrame(displayFrame, region, imgW, imgH);
    }

    // 2. 绘制活跃人员边界框
    for (const auto &p : persons) {
        bool inside = p.isInsideRegion || p.hasEntered;
        cv::Scalar boxColor = inside ? cv::Scalar(0, 255, 0) : cv::Scalar(180, 180, 180);
        cv::rectangle(displayFrame, p.bbox, boxColor, 2);

        // 3. 绘制活跃 ID 标签
        drawActiveIdOnCvMat(displayFrame, p);
    }

    // 4. 绘制淡出中的 ID 标签（仅绘制不在活跃列表中的）
    QSet<int> activeIds;
    for (const auto &p : persons) {
        activeIds.insert(p.trackId);
    }
    for (auto it = m_fadingLabels.constBegin(); it != m_fadingLabels.constEnd(); ++it) {
        if (!activeIds.contains(it.key()) && it.value().alpha > 0.01f) {
            drawFadingIdOnCvMat(displayFrame, it.value());
        }
    }

    // 转换回 QImage 用于显示
    cv::Mat rgbDisplay;
    cv::cvtColor(displayFrame, rgbDisplay, cv::COLOR_BGR2RGB);
    QImage displayImg(rgbDisplay.data, rgbDisplay.cols, rgbDisplay.rows,
                      static_cast<int>(rgbDisplay.step), QImage::Format_RGB888);
    displayImg = displayImg.copy();

    // 节流 UI 更新
    const bool shouldRender = m_lastUiUpdate.elapsed() >= UI_UPDATE_INTERVAL_MS;
    if (shouldRender) {
        if (m_imageView) m_imageView->setImage(displayImg);
        m_lastUiUpdate.restart();
    }

    // 更新 FPS 和人数显示
    ui->label_fps->setText(QString("FPS: %1").arg(m_currentFps, 0, 'f', 1));
    int boardingCnt = m_zoneMonitor ? m_zoneMonitor->boardingCount() : m_boardingCount;
    int alightingCnt = m_zoneMonitor ? m_zoneMonitor->alightingCount() : m_alightingCount;
    ui->label_personCount->setText(QString("区域人数: %1 | 上车: %2 | 下车: %3")
                                       .arg(personInRegionCount)
                                       .arg(boardingCnt)
                                       .arg(alightingCnt));
    m_currentPersonCount = personInRegionCount;

    // 更新访客记录表格
    updateVisitorTable();

    // 保存当前帧副本（用于手动保存）
    {
        QMutexLocker locker(&m_frameMutex);
        m_currentFrame = displayImg;
    }

    // 自动保存
    if (m_autoSaveEnabled && m_saveWorker && !m_diskFull) {
        QMetaObject::invokeMethod(m_saveWorker, "onSaveImage",
                                  Qt::QueuedConnection,
                                  Q_ARG(QImage, displayImg));
    }
}

void MainWindow::saveReidImage(const cv::Mat &frame, int trackId, const QString &type)
{
    if (frame.empty()) return;

    QString basePath = m_config.reidSavePath;
    if (!QDir(basePath).isAbsolute()) {
        basePath = QCoreApplication::applicationDirPath() + "/" + basePath;
    }

    QString personDir = QString("%1/person_%2").arg(basePath).arg(trackId);
    QDir().mkpath(personDir);

    QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss_zzz");
    QString filePath = QString("%1/%2_%3.jpg").arg(personDir).arg(type.toLower()).arg(timestamp);

    // 在图像上绘制信息
    cv::Mat saveFrame = frame.clone();
    std::string label = type.toStdString() + " " + QDateTime::currentDateTime().toString("HH:mm:ss").toStdString();
    cv::putText(saveFrame, "ID: " + std::to_string(trackId),
                cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    cv::putText(saveFrame, label,
                cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                type == "ENTER" ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);

    bool ok = cv::imwrite(filePath.toStdString(), saveFrame);
    if (ok) {
        LOG_INFO(QString("[MainWindow] ReID image saved: %1").arg(filePath));
    } else {
        LOG_WARN(QString("[MainWindow] Failed to save ReID image: %1").arg(filePath));
    }
}

void MainWindow::saveVisitorFrame(const cv::Mat &rawFrame, const cv::Rect &personBbox, int visitorId)
{
    if (rawFrame.empty() || visitorId <= 0) return;

    // 目录：save_images/visitor_V{id}/
    QString basePath = m_config.savePath;
    if (!QDir(basePath).isAbsolute()) {
        basePath = QCoreApplication::applicationDirPath() + "/" + basePath;
    }

    // 按日期分组
    QString dateStr = QDateTime::currentDateTime().toString("yyyyMMdd");
    QString visitorDir = QString("%1/%2/visitor_V%3")
                             .arg(basePath).arg(dateStr).arg(visitorId);
    QDir().mkpath(visitorDir);

    // 裁剪人物区域（带 padding）
    int padX = static_cast<int>(personBbox.width * 0.3);
    int padY = static_cast<int>(personBbox.height * 0.3);
    int x1 = std::max(0, personBbox.x - padX);
    int y1 = std::max(0, personBbox.y - padY);
    int x2 = std::min(rawFrame.cols, personBbox.x + personBbox.width + padX);
    int y2 = std::min(rawFrame.rows, personBbox.y + personBbox.height + padY);

    cv::Mat crop;
    if (x2 > x1 && y2 > y1) {
        crop = rawFrame(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
    } else {
        crop = rawFrame.clone();
    }

    // 在裁剪图上绘制 visitor ID 和时间戳
    QString ts = QDateTime::currentDateTime().toString("HHmmss_zzz");
    std::string label = "V" + std::to_string(visitorId) + " " + ts.toStdString();
    cv::putText(crop, label, cv::Point(10, 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

    // 文件名：V{id}_{timestamp}.jpg
    QString fileName = QString("%1/V%2_%3.jpg")
                           .arg(visitorDir).arg(visitorId).arg(ts);

    // 异步保存（通过 SaveImageWorker 或直接写入）
    cv::imwrite(fileName.toStdString(), crop);
}

void MainWindow::showDrawGuide(const QString &message)
{
    ui->captureStatusLabel->setText(message);
    ui->captureStatusLabel->setStyleSheet("color: #ffd54f; font-family: 'Consolas'; font-size: 12px; font-weight: bold; padding: 0 8px;");
}

void MainWindow::clearDrawGuide()
{
    updateCaptureStatus();
}

void MainWindow::onDrawRegion()
{
    if (!m_imageView) return;

    // 检查是否有图像
    QSize imgSize = m_imageView->displayedImageSize();
    bool hasImage = imgSize.isValid() && imgSize.width() > 0 && imgSize.height() > 0;
    if (!hasImage && !m_isDrawingRegion) {
        QMessageBox msgBox(this);
        msgBox.setWindowTitle("绘制检测区域 - 操作引导");
        msgBox.setText("⚠️ 当前没有画面，绘制的区域坐标可能不准确。\n\n"
                       "📋 推荐操作顺序：\n"
                       "  ① 先点击「启动采集」获取实时画面\n"
                       "  ② 然后点击「绘制检测区域」\n"
                       "  ③ 在画面上 左键点击 添加顶点（至少3个）\n"
                       "  ④ 右键点击 完成绘制\n\n"
                       "是否仍然继续？");
        msgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
        msgBox.button(QMessageBox::Yes)->setText("仍然绘制");
        msgBox.button(QMessageBox::No)->setText("先启动采集");
        msgBox.setDefaultButton(QMessageBox::No);
        if (msgBox.exec() != QMessageBox::Yes) {
            return;
        }
    }

    m_isDrawingRegion = !m_isDrawingRegion;

    if (m_isDrawingRegion) {
        m_imageView->setDrawMode(ImageView::DrawMode::DrawRectangle);
        ui->btnDrawRegion->setText("✓ 完成绘制");
        ui->btnDrawRegion->setStyleSheet(
            "QPushButton { background-color: #2a1a40; border: 1px solid #ff80a0; border-radius: 5px; "
            "color: #ff90b0; padding: 8px 22px; font-weight: bold; font-size: 13px; }"
            "QPushButton:hover { border-color: #ffa0c0; color: #ffb0d0; }");

        // 读取当前方向设置
        if (ui->cmbEntryDirection) {
            m_currentEntryDirection = (ui->cmbEntryDirection->currentIndex() == 1)
                                          ? EntryDirection::BottomToTop
                                          : EntryDirection::TopToBottom;
        }

        // 显示初始绘制引导
        QString dirStr = (m_currentEntryDirection == EntryDirection::TopToBottom) ? "↓从上往下" : "↑从下往上";
        showDrawGuide(QString("🖊️ 矩形绘制: 按住左键拖拽画矩形 | 方向: %1 | 松开完成").arg(dirStr));

        LOG_INFO(QString("[MainWindow] Enter region draw mode (direction: %1)")
                     .arg(m_currentEntryDirection == EntryDirection::TopToBottom ? "TopToBottom" : "BottomToTop"));
    } else {
        m_imageView->setDrawMode(ImageView::DrawMode::View);
        m_imageView->finishCurrentPolygon();
        ui->btnDrawRegion->setText("绘制检测区域");
        ui->btnDrawRegion->setStyleSheet(""); // 恢复默认样式
        clearDrawGuide();

        LOG_INFO("[MainWindow] Exit region draw mode");
    }
}

void MainWindow::onClearRegions()
{
    if (m_regionManager) {
        m_regionManager->clearRegions();
    }
    if (m_imageView) {
        m_imageView->clearRegionPolygons();
        disconnect(m_imageView, &ImageView::drawingPointAdded, this, nullptr);
    }
    m_isDrawingRegion = false;
    m_imageView->setDrawMode(ImageView::DrawMode::View);
    ui->btnDrawRegion->setText("绘制检测区域");
    ui->btnDrawRegion->setStyleSheet(""); // 恢复默认样式

    // 清除访客记录
    m_visitors.clear();
    m_trackToVisitor.clear();
    m_nextVisitorId = 1;

    // 重置通道监控器
    if (m_zoneMonitor) {
        m_zoneMonitor->reset();
    }
    m_boardingCount = 0;
    m_alightingCount = 0;

    if (ui->tableVisitors) {
        ui->tableVisitors->setRowCount(0);
    }

    clearDrawGuide();
    LOG_INFO("[MainWindow] All regions cleared");
}

void MainWindow::onPolygonCompleted(QVector<QPointF> points)
{
    if (points.size() < 3) {
        showDrawGuide("⚠️ 至少需要3个顶点！当前: " + QString::number(points.size()) + " 个，请继续添加");
        return;
    }

    // 获取图像尺寸以归一化坐标
    // 优先使用 ImageView 显示的图像尺寸，其次使用当前帧
    int imgW = 0, imgH = 0;

    if (m_imageView) {
        QSize displaySize = m_imageView->displayedImageSize();
        if (displaySize.isValid() && displaySize.width() > 0 && displaySize.height() > 0) {
            imgW = displaySize.width();
            imgH = displaySize.height();
        }
    }

    if (imgW <= 0 || imgH <= 0) {
        QMutexLocker locker(&m_frameMutex);
        if (!m_currentFrame.isNull()) {
            imgW = m_currentFrame.width();
            imgH = m_currentFrame.height();
        }
    }

    if (imgW <= 0) imgW = 1;
    if (imgH <= 0) imgH = 1;

    // 转换为归一化坐标
    QVector<QPointF> normalizedPoints;
    for (const QPointF &pt : points) {
        normalizedPoints.append(QPointF(pt.x() / imgW, pt.y() / imgH));
    }

    // 检查归一化坐标是否合理（应在0~2范围内，允许略微超出）
    bool coordsValid = true;
    for (const QPointF &pt : normalizedPoints) {
        if (pt.x() < -0.5 || pt.x() > 2.0 || pt.y() < -0.5 || pt.y() > 2.0) {
            coordsValid = false;
            break;
        }
    }

    if (!coordsValid) {
        LOG_WARN(QString("[MainWindow] Region coordinates may be invalid (imgW=%1, imgH=%2). "
                         "Consider starting capture first.")
                     .arg(imgW).arg(imgH));
    }

    // 创建区域
    DetectionRegion region;
    region.polygon = normalizedPoints;
    region.name = QString("区域%1").arg(m_regionManager->regions().size() + 1);
    region.entryDirection = m_currentEntryDirection;

    // 分配颜色
    static const QColor colors[] = {
        QColor(0, 255, 0), QColor(0, 200, 255), QColor(255, 100, 0),
        QColor(255, 0, 255), QColor(255, 255, 0), QColor(0, 255, 200)
    };
    int colorIdx = m_regionManager->regions().size() % 6;
    region.color = colors[colorIdx];

    m_regionManager->addRegion(region);

    // 更新 ImageView 显示
    QVector<QVector<QPointF>> allPolys;
    for (const auto &r : m_regionManager->regions()) {
        allPolys.append(r.polygon);
    }
    m_imageView->setRegionPolygons(allPolys);

    // 自动保存区域
    QString regionPath = m_config.regionSavePath;
    if (!QDir(regionPath).isAbsolute()) {
        regionPath = QCoreApplication::applicationDirPath() + "/" + regionPath;
    }
    m_regionManager->saveRegions(regionPath);

    // 更新采集状态和引导提示
    updateCaptureStatus();

    QString dirStr = (region.entryDirection == EntryDirection::TopToBottom) ? "从上往下↓" : "从下往上↑";
    int regionCount = m_regionManager->regions().size();
    showDrawGuide(QString("✅ 区域%1 已创建！%2个顶点，方向: %3 | 点击「完成绘制」结束")
                      .arg(regionCount).arg(points.size()).arg(dirStr));

    // 详细日志：原始坐标和归一化坐标
    LOG_INFO(QString("[MainWindow] ===== Region Created: %1 =====").arg(region.name));
    LOG_INFO(QString("[MainWindow]   imgSize=%1x%2, direction=%3").arg(imgW).arg(imgH).arg(dirStr));
    for (int i = 0; i < points.size(); ++i) {
        LOG_INFO(QString("[MainWindow]   raw[%1]=(%2,%3) -> norm=(%4,%5)")
                     .arg(i).arg(points[i].x(), 0, 'f', 1).arg(points[i].y(), 0, 'f', 1)
                     .arg(normalizedPoints[i].x(), 0, 'f', 4).arg(normalizedPoints[i].y(), 0, 'f', 4));
    }
    LOG_INFO(QString("[MainWindow]   Region count: %1, polygons in ImageView: %2")
                 .arg(regionCount).arg(m_imageView ? m_imageView->regionPolygons().size() : 0));
    LOG_INFO(QString("[MainWindow] ===== Region Created END ====="));
}

void MainWindow::onSaveCurrentFrame()
{
    QMutexLocker locker(&m_frameMutex);
    if (m_currentFrame.isNull()) {
        LOG_WARN("[MainWindow] No frame available to save");
        return;
    }

    if (m_saveWorker && !m_diskFull) {
        QMetaObject::invokeMethod(m_saveWorker, "onSaveImage",
                                  Qt::QueuedConnection,
                                  Q_ARG(QImage, m_currentFrame));
        LOG_INFO("[MainWindow] Manual save triggered");
    }
}

void MainWindow::onAutoSaveToggled(bool checked)
{
    m_autoSaveEnabled = checked;
    LOG_INFO(QString("[MainWindow] Auto save %1").arg(checked ? "enabled" : "disabled"));
}

void MainWindow::onGrabThreadDeviceLost()
{
    LOG_ERR("Camera device lost");
    updateCameraStatus(false);
    if (!m_reconnectTimer) {
        m_reconnectTimer = new QTimer(this);
        connect(m_reconnectTimer, &QTimer::timeout, this, &MainWindow::onTryReconnect);
    }
    m_reconnectAttempts = 0;
    m_reconnectTimer->start(RECONNECT_INTERVAL_MS);
}

void MainWindow::onTryReconnect()
{
    m_reconnectAttempts++;
    LOG_INFO(QString("[MainWindow] Reconnect attempt %1/%2")
                 .arg(m_reconnectAttempts).arg(MAX_RECONNECT_ATTEMPTS));

    if (m_camera.openCamera(m_config.rtspUrl)) {
        m_reconnectTimer->stop();

        // 重新创建采集线程（含检测管线）
        m_grabThread = new GrabThread(&m_camera, "HX-CCD23", this);

        if (m_detector) m_grabThread->setDetector(m_detector);
        if (m_tracker) m_grabThread->setTracker(m_tracker);
        if (m_reidExtractor) m_grabThread->setReidExtractor(m_reidExtractor);

        connect(m_grabThread, &GrabThread::deviceLost, this, &MainWindow::onGrabThreadDeviceLost);
        connect(m_grabThread, &GrabThread::frameProcessed, this, &MainWindow::onFrameProcessed);
        connect(m_grabThread, &GrabThread::frameGrabbed, this, &MainWindow::onFrameGrabbed);
        connect(m_grabThread, &GrabThread::rawFrameGrabbed, this, [this](const cv::Mat &frame) {
            QMutexLocker locker(&m_rawFrameMutex);
            m_currentRawFrame = frame.clone();
        });

        // 重置 FPS 计算
        m_fpsTimer.restart();
        m_fpsFrameCount = 0;
        m_currentFps = 0.0;

        m_grabThread->start();
        updateCameraStatus(true);
        LOG_INFO("[MainWindow] Reconnected successfully with detection pipeline");
    } else if (m_reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
        m_reconnectTimer->stop();
        LOG_ERR("Max reconnect attempts reached");
    }
}

void MainWindow::updateCameraStatus(bool connected) {
    if (connected) {
        ui->label_camStatus->setText("● 相机: ONLINE");
        ui->label_camStatus->setStyleSheet("color: #00e676; font-family: 'Consolas'; font-size: 12px; font-weight: bold;");
    } else {
        ui->label_camStatus->setText("● 相机: OFFLINE");
        ui->label_camStatus->setStyleSheet("color: #ef5350; font-family: 'Consolas'; font-size: 12px; font-weight: bold;");
    }
}

void MainWindow::onUpdateSystemTime() {
    QString timeStr = QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss");
    ui->timeLabel->setText(timeStr);
    if (m_elapsedTimer.isValid()) {
        qint64 elapsed = m_elapsedTimer.elapsed();
        int hours = elapsed / 3600000;
        int minutes = (elapsed % 3600000) / 60000;
        int seconds = (elapsed % 60000) / 1000;
        ui->runTimeLabel->setText(QString("运行: %1:%2:%3")
            .arg(hours, 2, 10, QChar('0'))
            .arg(minutes, 2, 10, QChar('0'))
            .arg(seconds, 2, 10, QChar('0')));
    }

    // 实时更新访客表格中的停留时长（每秒更新一次）
    if (m_isCapturing && !m_visitors.isEmpty()) {
        updateVisitorTable();
    }
}

void MainWindow::updateDiskSpaceStatus() {
    QStorageInfo storage(m_config.savePath.isEmpty()
                         ? QCoreApplication::applicationDirPath()
                         : m_config.savePath);
    if (storage.isValid()) {
        qint64 freeGB = storage.bytesAvailable() / (1024LL * 1024 * 1024);
        ui->diskLabel->setText(QString("磁盘: %1 GB 可用").arg(freeGB));
        m_diskFull = (freeGB < 1);
        if (m_saveWorker) m_saveWorker->setDiskFull(m_diskFull);
    }
}

void MainWindow::onOfflineDetect()
{
    // 创建离线检测使用的临时检测器（如果主检测器未加载）
    PersonDetector *offlineDetector = nullptr;
    bool ownsDetector = false;

    if (m_detector && m_detector->isLoaded()) {
        offlineDetector = m_detector;
        LOG_INFO("[OfflineDetect] Using existing loaded detector");
    } else {
        // 尝试加载模型
        offlineDetector = new PersonDetector();
        QString modelPath = m_config.modelPath;
        if (!QFileInfo(modelPath).isAbsolute()) {
            modelPath = QCoreApplication::applicationDirPath() + "/" + modelPath;
        }

        LOG_INFO("[OfflineDetect] Trying to load model: " + modelPath);

        bool loaded = offlineDetector->loadModel(modelPath, m_config.detectorConfThreshold, m_config.detectorNmsThreshold);

        // 如果 .engine 加载失败，尝试回退到 .onnx
        if (!loaded && modelPath.endsWith(".engine", Qt::CaseInsensitive)) {
            QString onnxPath = modelPath.left(modelPath.lastIndexOf('.')) + ".onnx";
            LOG_INFO("[OfflineDetect] Engine failed, trying ONNX fallback: " + onnxPath);
            if (QFileInfo::exists(onnxPath)) {
                loaded = offlineDetector->loadModel(onnxPath, m_config.detectorConfThreshold, m_config.detectorNmsThreshold);
            } else {
                LOG_WARN("[OfflineDetect] ONNX fallback not found: " + onnxPath);
            }
        }

        // 如果 .onnx 加载失败，尝试回退到 .engine
        if (!loaded && modelPath.endsWith(".onnx", Qt::CaseInsensitive)) {
            QString enginePath = modelPath.left(modelPath.lastIndexOf('.')) + ".engine";
            LOG_INFO("[OfflineDetect] ONNX failed, trying engine fallback: " + enginePath);
            if (QFileInfo::exists(enginePath)) {
                loaded = offlineDetector->loadModel(enginePath, m_config.detectorConfThreshold, m_config.detectorNmsThreshold);
            } else {
                LOG_WARN("[OfflineDetect] Engine fallback not found: " + enginePath);
            }
        }

        if (!loaded) {
            QString errorMsg = QString(
                "无法加载检测模型！\n\n"
                "配置的模型路径: %1\n"
                "解析后路径: %2\n\n"
                "请通过「系统配置」→「推理设置」检查模型路径，"
                "或确认模型文件存在于 model/ 目录下。")
                .arg(m_config.modelPath).arg(modelPath);
            QMessageBox::warning(this, "离线检测 - 模型加载失败", errorMsg);
            delete offlineDetector;
            return;
        }
        ownsDetector = true;

        QString backend = offlineDetector->isTrtBackend() ? "TensorRT" : "OpenCV DNN";
        LOG_INFO(QString("[OfflineDetect] Model loaded [%1]: %2").arg(backend, modelPath));
    }

    OfflineDetectForm form(offlineDetector, m_regionManager, this);
    form.exec();

    if (ownsDetector) {
        delete offlineDetector;
    }

    LOG_INFO("[MainWindow] Offline detect form closed");
}

void MainWindow::onSetParams()
{
    SettingForm dlg(m_config, this);
    if (dlg.exec() == QDialog::Accepted) {
        m_config = dlg.getConfig();
        m_config.save();
        applySettings();
        LOG_INFO("[MainWindow] Settings saved and applied");
    }
}

void MainWindow::applySettings()
{
    // 应用窗口标题
    this->setWindowTitle(m_config.windowTitle);

    // 应用保存参数
    if (m_saveWorker) {
        m_saveWorker->setSaveParams(m_config.savePath, m_config.saveFormat, m_config.jpegQuality);
    }

    // 应用跟踪器参数（运行时生效）
    if (m_tracker) {
        m_tracker->setIouThreshold(m_config.trackerIouThreshold);
        m_tracker->setMaxMissed(m_config.trackerMaxMissed);
        m_tracker->setKalmanEnabled(m_config.trackerKalmanEnabled);
        m_tracker->setReidMatchingEnabled(m_config.trackerReidEnabled);
        m_tracker->setReidMatchThreshold(m_config.trackerReidThreshold);
        m_tracker->setGalleryMaxAge(m_config.trackerGalleryMaxAgeMs);
        m_tracker->setVisitorRegistry(m_registry);
        m_tracker->setTrackMergeSimilarity(m_config.trackerMergeSimilarity);
        m_tracker->setTrackMergeMaxYoungAge(m_config.trackerMergeMaxYoungAge);
    }

    // 应用 ReID 匹配阈值（运行时生效）
    if (m_zoneMonitor) {
        m_zoneMonitor->setReidMatchThreshold(m_config.reidMatchThreshold);
    }

    LOG_INFO(QString("[MainWindow] Applied settings: model=%1, conf=%2, nms=%3, iou=%4, reid=%5, kalman=%6, trackerReid=%7")
                 .arg(m_config.modelPath)
                 .arg(m_config.detectorConfThreshold, 0, 'f', 2)
                 .arg(m_config.detectorNmsThreshold, 0, 'f', 2)
                 .arg(m_config.trackerIouThreshold, 0, 'f', 2)
                 .arg(m_config.reidMatchThreshold, 0, 'f', 2)
                 .arg(m_config.trackerKalmanEnabled ? "ON" : "OFF")
                 .arg(m_config.trackerReidEnabled ? "ON" : "OFF"));
}

// ===== 访客记录持久化 =====
void MainWindow::initVisitorLog()
{
    m_visitorLog = new VisitorLog();

    QString logDir = m_config.visitorLogPath;
    if (!QDir(logDir).isAbsolute()) {
        logDir = QCoreApplication::applicationDirPath() + "/" + logDir;
    }
    m_visitorLog->setSaveDir(logDir);

    // 加载历史访客记录
    loadVisitorRecords();

    // 记录会话开始
    m_visitorLog->logSessionStart();

    // 设置自动保存定时器
    m_visitorSaveTimer = new QTimer(this);
    connect(m_visitorSaveTimer, &QTimer::timeout, this, [this]() {
        if (m_isCapturing && !m_visitors.isEmpty()) {
            saveVisitorRecords();
        }
    });
    m_visitorSaveTimer->start(VISITOR_SAVE_INTERVAL_MS);

    LOG_INFO("[MainWindow] Visitor log initialized: " + logDir);
}

void MainWindow::saveVisitorRecords()
{
    if (!m_visitorLog) return;

    QMap<int, VisitorLog::VisitorEntry> entries;
    for (auto it = m_visitors.constBegin(); it != m_visitors.constEnd(); ++it) {
        const VisitorRecord &vr = it.value();
        VisitorLog::VisitorEntry entry;
        entry.visitorId = vr.visitorId;
        entry.enterTime = vr.enterTime.isValid() ? vr.enterTime.toString(Qt::ISODate) : QString();
        entry.exitTime = vr.exitTime.isValid() ? vr.exitTime.toString(Qt::ISODate) : QString();
        entry.isInside = vr.isInside;
        entry.regionId = vr.regionId;
        entry.currentTrackId = vr.currentTrackId;
        entry.reidFeatureBase64 = VisitorLog::matToBase64(vr.reidFeature);
        // 进入/离开图像路径由 saveReidImage 管理，这里记录目录
        entry.enterImagePath = QString("person_%1").arg(vr.visitorId);
        entry.exitImagePath = QString("person_%1").arg(vr.visitorId);
        entry.direction = vr.direction;
        entry.enterDirection = vr.enterDirection;
        entry.exitDirection = vr.exitDirection;
        entry.entrySide = vr.entrySide;
        entry.exitSide = vr.exitSide;
        entry.dwellSeconds = vr.dwellSeconds;
        entries[vr.visitorId] = entry;
    }

    m_visitorLog->saveVisitors(entries);
    LOG_INFO(QString("[MainWindow] Visitor records saved: %1 entries").arg(entries.size()));
}

void MainWindow::loadVisitorRecords()
{
    if (!m_visitorLog) return;

    QMap<int, VisitorLog::VisitorEntry> entries = m_visitorLog->loadVisitors();
    if (entries.isEmpty()) {
        LOG_INFO("[MainWindow] No previous visitor records to load");
        return;
    }

    int restoredCount = 0;
    for (auto it = entries.constBegin(); it != entries.constEnd(); ++it) {
        const VisitorLog::VisitorEntry &entry = it.value();

        VisitorRecord vr;
        vr.visitorId = entry.visitorId;
        vr.isInside = false; // 应用重启后所有访客都标记为已离开
        vr.currentTrackId = -1;
        vr.regionId = entry.regionId;

        // 恢复时间
        if (!entry.enterTime.isEmpty()) {
            vr.enterTime = QDateTime::fromString(entry.enterTime, Qt::ISODate);
        }
        if (!entry.exitTime.isEmpty()) {
            vr.exitTime = QDateTime::fromString(entry.exitTime, Qt::ISODate);
        }

        // 恢复 ReID 特征
        if (!entry.reidFeatureBase64.isEmpty()) {
            vr.reidFeature = VisitorLog::base64ToMat(entry.reidFeatureBase64);
        }

        // 恢复进出方向与停留时长
        vr.direction = entry.direction;
        vr.enterDirection = entry.enterDirection;
        vr.exitDirection = entry.exitDirection;
        vr.entrySide = entry.entrySide;
        vr.exitSide = entry.exitSide;
        vr.dwellSeconds = entry.dwellSeconds;

        m_visitors[entry.visitorId] = vr;

        // 更新 m_nextVisitorId
        if (entry.visitorId >= m_nextVisitorId) {
            m_nextVisitorId = entry.visitorId + 1;
        }
        if (m_registry) m_registry->ensureNextIdAtLeast(entry.visitorId + 1);

        restoredCount++;
    }

    // 更新表格显示
    updateVisitorTable();

    LOG_INFO(QString("[MainWindow] Loaded %1 visitor records from previous session (nextVisitorId=%2)")
                 .arg(restoredCount).arg(m_nextVisitorId));
}
