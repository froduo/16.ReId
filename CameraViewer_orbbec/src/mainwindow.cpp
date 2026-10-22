#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "version.h"
#include <QMessageBox>
#include <QTimer>
#include <QThread>
#include <QCoreApplication>
#include <QStorageInfo>
#include <QDir>
#include <QFile>
#include <QDateTime>
#include <QPainter>
#include <QAction>
#include <QDesktopServices>
#include <QUrl>
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QPair>
#include <QSplitter>
#include <opencv2/opencv.hpp>
#include <QHeaderView>
#include "overlaytext.h"
#include "statuswebserver.h"

#include <cmath>

// 旧链路（PassengerBook 特征匹配 + passenger/P{id}_* 序列 + visitor_V{id}/* 逐帧 + 上下车 mp4）
// 暂时停用（保留代码便于回退）。需要恢复旧行为时改为 true。
static constexpr bool kLegacyPipelineEnabled = false;

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
    LOG_INFO("CameraViewer starting... Version: " + appVersionText());

    // 替换 UI 中的 QGraphicsView 为自定义 ImageView
    QGraphicsView* oldView = ui->imageView;
    if (oldView) {
        m_imageView = new ImageView(this);
        m_imageView->setObjectName("imageViewMain");
        // 图像区域横向拉伸权重 14，右侧人员记录面板 11：
        // 图像约占横向 56%（原为 80%），宽度缩小约 30%，腾出的空间给右侧记录表。
        QSizePolicy imagePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        imagePolicy.setHorizontalStretch(14);
        imagePolicy.setVerticalStretch(0);
        m_imageView->setSizePolicy(imagePolicy);
        QVBoxLayout* mainLayout = ui->mainVerticalLayout;
        if (mainLayout) {
            // 找到 imageView 在 imageLayout 中的位置
            QHBoxLayout* imageLayout = ui->imageLayout;
            if (imageLayout) {
                int idx = imageLayout->indexOf(oldView);
                if (idx >= 0) {
                    imageLayout->removeWidget(oldView);
                    imageLayout->insertWidget(idx, m_imageView);
                    // 显式设置拉伸比：图像 14 : 右侧面板 11
                    imageLayout->setStretch(idx, 14);
                    if (idx + 1 < imageLayout->count())
                        imageLayout->setStretch(idx + 1, 11);
                }
            }
        }
        delete oldView;
    }

    // ===== 深度伪彩视图（深度流）：与彩色视图左右分屏 =====
    if (m_imageView) {
        if (QHBoxLayout* imageLayout = ui->imageLayout) {
            m_depthView = new ImageView(this);
            m_depthView->setObjectName("depthViewMain");
            m_depthView->setDrawMode(ImageView::DrawMode::View);
            m_depthView->setToolTip("深度伪彩视图（近处偏暖色；黑=无有效深度）");
            QSizePolicy depthPolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
            depthPolicy.setHorizontalStretch(14);
            m_depthView->setSizePolicy(depthPolicy);

            m_mediaSplitter = new QSplitter(Qt::Horizontal, this);
            m_mediaSplitter->setObjectName("mediaSplitter");
            m_mediaSplitter->setChildrenCollapsible(false);

            const int idx = imageLayout->indexOf(m_imageView);
            if (idx >= 0) {
                imageLayout->removeWidget(m_imageView);
                m_mediaSplitter->addWidget(m_imageView);
                m_mediaSplitter->addWidget(m_depthView);
                m_mediaSplitter->setStretchFactor(0, 3);
                m_mediaSplitter->setStretchFactor(1, 2);
                imageLayout->insertWidget(idx, m_mediaSplitter);
                imageLayout->setStretch(idx, 14);
                if (idx + 1 < imageLayout->count())
                    imageLayout->setStretch(idx + 1, 11);
            } else {
                imageLayout->addWidget(m_mediaSplitter);
            }
            m_depthView->setVisible(true);
        }
    }

    ui->statusbar->hide();
    ui->timeLabel->setStyleSheet("color: #4dd0e1; font-family: 'Consolas'; font-size: 12px; font-weight: bold; padding: 0 8px;");
    ui->runTimeLabel->setStyleSheet("color: #ffd54f; font-family: 'Consolas'; font-size: 12px; font-weight: bold; padding: 0 8px;");
    ui->diskLabel->setStyleSheet("color: #81c784; font-family: 'Consolas'; font-size: 12px; font-weight: bold; padding: 0 8px;");
    ui->captureStatusLabel->setStyleSheet("color: #b0bec5; font-family: 'Consolas'; font-size: 12px; font-weight: bold; padding: 0 8px;");

    // ===== 版本号显示（单一来源 version.h）=====
    if (ui->versionLabel) {
        ui->versionLabel->setText(QString("版本: %1").arg(appVersionText()));
    }
    this->setWindowTitle(QString("%1  %2").arg(m_config.windowTitle, appVersionText()));
    // 关于 / 版本信息、检查更新 菜单
    if (ui->actionversion) {
        connect(ui->actionversion, &QAction::triggered, this, &MainWindow::onAbout);
    }
    if (ui->actionCheckUpdate) {
        connect(ui->actionCheckUpdate, &QAction::triggered, this, &MainWindow::onCheckUpdate);
    }
    m_netMgr = new QNetworkAccessManager(this);
    LOG_INFO(QString("[MainWindow] App version: %1 (build %2 %3, code=%4)")
                 .arg(APP_VERSION_DISPLAY).arg(APP_BUILD_DATE).arg(APP_BUILD_TIME).arg(APP_VERSION_CODE));

    // 连接信号槽 - 使用 toggle 按钮替代 btnOpen/btnClose
    connect(ui->btnToggle, &QPushButton::clicked, this, &MainWindow::onToggleCapture);
    connect(ui->btnSelectVideo, &QPushButton::clicked, this, &MainWindow::onSelectVideo);
    connect(ui->btnSave, &QPushButton::clicked, this, &MainWindow::onSaveCurrentFrame);
    connect(ui->chkAutoSave, &QCheckBox::toggled, this, &MainWindow::onAutoSaveToggled);

    // 区域绘制相关信号槽
    connect(ui->btnDrawRegion, &QPushButton::clicked, this, &MainWindow::onDrawRegion);
    connect(ui->btnClearRegion, &QPushButton::clicked, this, &MainWindow::onClearCounts);
    connect(ui->btnOfflineDetect, &QPushButton::clicked, this, &MainWindow::onOfflineDetect);
    connect(ui->btnSettings, &QPushButton::clicked, this, &MainWindow::onSetParams);

    // ImageView 多边形绘制完成信号
    if (m_imageView) {
        connect(m_imageView, &ImageView::polygonCompleted, this, &MainWindow::onPolygonCompleted);
    }

    // 初始化访客记录表格
    if (ui->tableVisitors) {
        // 人员进出记录格式：乘客ID / 状态(上车|下车) / 乘客匹配ID / 相似度
        ui->tableVisitors->setColumnCount(4);
        ui->tableVisitors->setHorizontalHeaderLabels({"乘客ID", "状态", "乘客匹配ID", "相似度"});
        ui->tableVisitors->setEditTriggers(QAbstractItemView::NoEditTriggers);
        ui->tableVisitors->setSelectionBehavior(QAbstractItemView::SelectRows);
        ui->tableVisitors->setSelectionMode(QAbstractItemView::SingleSelection);
        // 四列均随内容自适应（ID/匹配ID 为 person_N_yyyyMMdd_HHmmss 长文本，相似度为百分制短文本）
        ui->tableVisitors->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
        ui->tableVisitors->setColumnWidth(0, 230);   // 乘客ID 初始宽度
        ui->tableVisitors->setColumnWidth(1, 60);    // 状态
        ui->tableVisitors->setColumnWidth(2, 230);   // 乘客匹配ID 初始宽度
        ui->tableVisitors->setColumnWidth(3, 70);    // 相似度
        ui->tableVisitors->verticalHeader()->setDefaultSectionSize(22); // 行高紧凑
    }

    // 初始化进入方向下拉框：默认加载上次保存的下车方向
    if (ui->cmbEntryDirection) {
        const bool bottomToTop = (m_config.entryDirection == QLatin1String("BottomToTop"));
        ui->cmbEntryDirection->setCurrentIndex(bottomToTop ? 1 : 0);
        m_currentEntryDirection = bottomToTop ? EntryDirection::BottomToTop
                                              : EntryDirection::TopToBottom;
        connect(ui->cmbEntryDirection, QOverload<int>::of(&QComboBox::currentIndexChanged),
                this, [this](int idx) {
                    m_currentEntryDirection = (idx == 1) ? EntryDirection::BottomToTop
                                                         : EntryDirection::TopToBottom;
                    m_config.entryDirection = (idx == 1) ? QStringLiteral("BottomToTop")
                                                         : QStringLiteral("TopToBottom");
                    m_config.save();
                    // 同步更新所有已绘制区域的方向并落盘，避免界面设置与实际判定不一致
                    if (m_regionManager) {
                        m_regionManager->setAllEntryDirection(m_currentEntryDirection);
                        QString regionPath = m_config.regionSavePath;
                        if (!QDir(regionPath).isAbsolute()) {
                            regionPath = QCoreApplication::applicationDirPath() + "/" + regionPath;
                        }
                        m_regionManager->saveRegions(regionPath);
                        refreshRegionOverlays();
                    }
                    LOG_INFO(QString("[MainWindow] 下车方向已保存: %1（已同步 %2 个区域）")
                                 .arg(m_config.entryDirection)
                                 .arg(m_regionManager ? m_regionManager->regions().size() : 0));
                });
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

        // 用 config.ini 中保存的下车方向覆盖所有区域，确保界面设置与区域判定一致
        // （regions.json 可能残留旧方向，启动加载后必须与界面设置对齐）
        m_regionManager->setAllEntryDirection(m_currentEntryDirection);
        m_regionManager->saveRegions(regionPath);

        // 将已保存的区域 + 虚拟门线显示在 ImageView 中
        refreshRegionOverlays();
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
    initPassengerBook();
    initTripStore();
    initPersonFolder();
    initWebServer();

    // 会话状态持久化路径（与访客台账同目录）
    {
        QString dir = m_config.visitorLogPath;
        if (!QDir(dir).isAbsolute())
            dir = QCoreApplication::applicationDirPath() + "/" + dir;
        m_sessionStatePath = QDir(dir).filePath(QStringLiteral("session_state.json"));
    }
    // 恢复上次会话的人员进出记录与上车目录登记（计数已在 initTripStore 中恢复）
    loadSessionState();

    // 创建通道监控器（上车/下车检测）
    m_zoneMonitor = new ZoneMonitor(this);
    m_zoneMonitor->setReidMatchThreshold(m_config.reidMatchThreshold);
    m_zoneMonitor->setConfirmFrames(3);
    // 同步恢复的累计计数（标签显示优先使用 ZoneMonitor 的计数）
    m_zoneMonitor->setCounters(m_boardingCount, m_alightingCount);

    // ===== 下车身份比对：独立线程（避免阻塞 UI 与实时检测管线）=====
    m_matchThread = new QThread(this);
    m_matchWorker = new IdentityMatchWorker();
    m_matchWorker->moveToThread(m_matchThread);
    connect(m_matchWorker, &IdentityMatchWorker::finished, this,
            [this](quint64 jobId, int bestId, float similarity, bool matched) {
                // 回到主线程结算（台账增删仅在此线程进行）
                const PendingAlight pend = m_pendingAlights.take(jobId);
                applyAlightingMatch(pend, bestId, similarity, matched);
            });
    m_matchThread->start();
    // 下车比对线程使用独立 ReID 提取器 → 首帧前先同步一次用户配置的融合权重
    m_matchWorker->setComponentWeights(m_config.reidWeightColor,
                                       m_config.reidWeightTexture,
                                       m_config.reidWeightShape);
    LOG_INFO("[MainWindow] 下车身份比对线程已启动");

    // ===== 上/下车文件夹全图比对线程（准确率优先，独立于实时特征比对）=====
    m_folderMatchThread = new QThread(this);
    m_folderMatchWorker = new FolderMatchWorker();
    m_folderMatchWorker->moveToThread(m_folderMatchThread);
    connect(m_folderMatchWorker, &FolderMatchWorker::finished, this,
            [this](quint64 jobId, int bestPersonId, QString bestDir,
                   float similarity, bool matched) {
                onFolderMatchFinished(jobId, bestPersonId, bestDir, similarity, matched);
            });
    m_folderMatchWorker->setMaxImagesPerFolder(m_config.personFolderMaxImages);
    m_folderMatchThread->start();
    LOG_INFO("[MainWindow] 上/下车文件夹全图比对线程已启动");

    // ===== 下车比对攒批定时器：窗口内的多个下车事件合并做匈牙利全局匹配 =====
    m_alightBatchTimer = new QTimer(this);
    m_alightBatchTimer->setSingleShot(true);
    connect(m_alightBatchTimer, &QTimer::timeout, this, &MainWindow::flushAlightBatch);

    // ===== 深度 ReID（OSNet 现场微调模型）：加载并注入各使用方 =====
    initDeepReid();

    // ===== 上车事件：登记乘客 + 上车时间 + 保存「乘客ID_上车_时间」图片 =====
    connect(m_zoneMonitor, &ZoneMonitor::boardingEvent, this, [this](const ZoneEvent &evt) {
        m_boardingCount++;

        // ★ 关键诊断：上车必须「完整穿越」—— 起点在区域上边界之外(A端之上)，
        //   终点在区域下边界之外(B端之下)。
        LOG_INFO(QString("[MainWindow] ★ BOARDING 判定依据(完整穿越: 上边界外→下边界外): "
                         "V%1 track=%2 region=%3 区域[topY=%4 bottomY=%5 midY=%6] | "
                         "startY=%7 endY=%8 dy=%9 frames=%10 durationMs=%11 suppressed=%12")
                     .arg(evt.visitorId).arg(evt.trackId).arg(evt.regionId)
                     .arg(evt.zoneTopNormY, 0, 'f', 4).arg(evt.zoneBottomNormY, 0, 'f', 4)
                     .arg(evt.zoneCenterNormY, 0, 'f', 4)
                     .arg(evt.startNormY, 0, 'f', 4).arg(evt.endNormY, 0, 'f', 4)
                     .arg(evt.moveDy, 0, 'f', 4)
                     .arg(evt.trackFrames).arg(evt.durationMs)
                     .arg(evt.suppressed ? evt.suppressReason : QStringLiteral("无")));

        int pid = evt.visitorId > 0 ? evt.visitorId : m_trackToVisitor.value(evt.trackId, -1);
        const QDateTime boardTime = evt.exitTime.isValid() ? evt.exitTime
                                                           : QDateTime::currentDateTime();

        if (kLegacyPipelineEnabled) {
            // 上车整段序列特征（进入检测区域 → 离开检测区域），下车时用于整段序列比对
            QVector<cv::Mat> feats = evt.trackFeatures;
            if (feats.isEmpty()) {
                if (!evt.enterFeature.empty()) feats.append(evt.enterFeature);
                if (!evt.exitFeature.empty())  feats.append(evt.exitFeature);
            }

            // 先登记台账（必要时自动分配乘客 ID），再据此命名并保存上车图片
            if (m_passengerBook) {
                pid = m_passengerBook->registerBoarding(pid, feats, boardTime, QString());
            }

            // 保存“进入检测区域 → 离开检测区域”的整段上车图片序列
            if (m_passengerBook && !evt.trackImages.isEmpty()) {
                const QString dirPath = savePassengerEventImages(evt.trackImages, evt.trackBoxes,
                                                                 pid, true, boardTime);
                if (!dirPath.isEmpty()) m_passengerBook->updateBoardingImage(pid, dirPath);
            } else {
                const cv::Mat &snap = !evt.exitImage.empty() ? evt.exitImage : evt.enterImage;
                const cv::Rect &snapBox = !evt.exitBbox.empty() ? evt.exitBbox : evt.enterBbox;
                if (m_passengerBook && !snap.empty()) {
                    const QString imgPath = savePassengerEventImage(snap, snapBox, pid, true, boardTime);
                    m_passengerBook->updateBoardingImage(pid, imgPath);
                }
            }

            // 保存「进入检测区域 → 离开检测区域」的整段上车视频
            if (m_config.reidSaveVideo && !evt.trackFullFrames.isEmpty()) {
                savePassengerEventVideo(evt.trackFullFrames, pid, true, boardTime, evt.durationMs);
            }

            // 归档本次穿越的逐帧图像：visitor_V{id}/上车_{时间}/
            finalizeVisitorSegment(m_trackToVisitor.value(evt.trackId, -1), true, boardTime);
        }

        // 更新访客记录（上车时间 / 方向）
        if (pid > 0) {
            VisitorRecord &vr = m_visitors[pid];
            vr.visitorId = pid;
            vr.direction = QStringLiteral("boarding");
            vr.enterTime = boardTime;
            vr.regionId = evt.regionId;
        }

        if (m_visitorLog) {
            VisitorLog::EventEntry logEntry;
            logEntry.timestamp = boardTime;
            logEntry.eventType = "BOARDING";
            logEntry.visitorId = pid;
            logEntry.trackId = evt.trackId;
            logEntry.regionId = evt.regionId;
            logEntry.similarity = 0.0f;
            logEntry.details = QString("上车(从下往上 B→A) 乘客P%1, enter=%2, exit=%3")
                                   .arg(pid)
                                   .arg(evt.enterTime.toString("HH:mm:ss.zzz"))
                                   .arg(evt.exitTime.toString("HH:mm:ss.zzz"));
            m_visitorLog->logEvent(logEntry);
        }

        // ===== 数据存储：记录本站在该行程的「上车人数」并更新车厢内人数 =====
        if (m_tripStore && m_tripStore->isOpen()) {
            const QString pidStr = pid > 0 ? QString("P%1").arg(pid) : QStringLiteral("unknown");
            m_tripStore->recordBoarding(m_currentTripId, m_currentStationId, m_currentStationName,
                                        pidStr, boardTime);
            m_onboardCount = m_tripStore->onboardCount(m_currentTripId);
            updateOnboardLabel();
        }

        // ===== 按乘客分目录：写入方向判定 + 登记上车目录（供下车目录全图比对）=====
        if (m_personFolder) {
            int personId = m_personFolder->personIdForTrack(evt.trackId);
            if (personId <= 0) personId = m_personFolder->ensurePerson(evt.trackId);
            m_personFolder->setDirection(personId, true, boardTime);
            const QString boardDir = m_personFolder->cropImagesDir(personId);
            BoardFolderEntry entry;
            entry.personId = personId;
            entry.dir = boardDir;
            entry.timeMs = boardTime.toMSecsSinceEpoch();
            entry.matched = false;
            m_boardFolders.append(entry);

            PersonEventRow row;
            row.personId = personId;
            row.status = QStringLiteral("上车");
            m_personEventRows.append(row);

            LOG_INFO(QString("[MainWindow] 上车目录已登记: person_%1 → %2")
                         .arg(personId).arg(boardDir));

            // 每次穿越 person 编号 +1：后续持续保存写入新编号，避免同一 person 重复出现
            m_personFolder->rotatePerson(evt.trackId);
        }

        updateVisitorTable();
        saveSessionState();
        LOG_INFO(QString("[MainWindow] ★ BOARDING EVENT: 乘客P%1, count=%2, 上车时间=%3")
                     .arg(pid).arg(m_boardingCount).arg(boardTime.toString("yyyy-MM-dd HH:mm:ss")));
    });

    // ===== 下车事件：提交后台线程进行图像比对，判定是哪位乘客下车 =====
    connect(m_zoneMonitor, &ZoneMonitor::alightingEvent, this, [this](const ZoneEvent &evt) {
        m_alightingCount++;

        const int fallbackId = evt.visitorId > 0 ? evt.visitorId
                                                 : m_trackToVisitor.value(evt.trackId, -1);
        const QDateTime alightTime = evt.exitTime.isValid() ? evt.exitTime
                                                            : QDateTime::currentDateTime();

        // ===== 数据存储：记录本站在该行程的「下车人数」并更新车厢内人数 =====
        if (m_tripStore && m_tripStore->isOpen()) {
            const QString pidStr = evt.visitorId > 0 ? QString("P%1").arg(evt.visitorId)
                                                     : QStringLiteral("unknown");
            m_tripStore->recordAlighting(m_currentTripId, m_currentStationId, m_currentStationName,
                                         pidStr, alightTime);
            m_onboardCount = m_tripStore->onboardCount(m_currentTripId);
            updateOnboardLabel();
        }

        // ===== 按乘客分目录：写入方向判定 + 入队后台整目录全图比对 =====
        if (m_personFolder) {
            int alightPersonId = m_personFolder->personIdForTrack(evt.trackId);
            if (alightPersonId <= 0) alightPersonId = m_personFolder->ensurePerson(evt.trackId);
            m_personFolder->setDirection(alightPersonId, false, alightTime);
            const QString alightDir = m_personFolder->cropImagesDir(alightPersonId);

            PersonEventRow row;
            row.personId = alightPersonId;
            row.status = QStringLiteral("下车");
            m_personEventRows.append(row);

            // 每次穿越 person 编号 +1：后续持续保存写入新编号，避免同一 person 重复出现
            m_personFolder->rotatePerson(evt.trackId);

            // 候选：时间更早 + 未匹配 的上车目录。
            // 额外排除「上车后过短时间(30s 内)即下车」的候选：此类候选往往是
            // 模型相似度虚高导致的误匹配（真实上下车一般间隔更长）。
            const qint64 alightMs = alightTime.toMSecsSinceEpoch();
            constexpr qint64 MIN_DWELL_MS = 30000;
            QVector<BoardFolderCandidate> candidates;
            for (const BoardFolderEntry &e : m_boardFolders) {
                if (e.matched) continue;
                if (e.timeMs > alightMs) continue;
                if (alightMs - e.timeMs < MIN_DWELL_MS) continue;
                BoardFolderCandidate c;
                c.personId = e.personId;
                c.dir = e.dir;
                c.timeMs = e.timeMs;
                candidates.append(c);
            }

            if (!alightDir.isEmpty() && !candidates.isEmpty()
                && m_folderMatchWorker && m_config.reidUseDeepReid) {
                const quint64 jobId = ++m_folderMatchJobCounter;
                FolderMatchPending pend;
                pend.alightPersonId = alightPersonId;
                pend.alightDir = alightDir;
                pend.alightTime = alightTime;
                m_folderMatchPending.insert(jobId, pend);
                m_folderMatchWorker->submit(jobId, alightPersonId, alightDir,
                                            candidates, m_config.reidFolderMatchThreshold);
                LOG_INFO(QString("[MainWindow] 下车目录已入队全图比对: person_%1 → %2, 候选上车目录=%3")
                             .arg(alightPersonId).arg(alightDir).arg(candidates.size()));
            } else if (!alightDir.isEmpty() && candidates.isEmpty()) {
                LOG_INFO(QString("[MainWindow] 下车 person_%1 无可比对上车目录（无更早且未标记的上车）")
                             .arg(alightPersonId));
            }
        }

        if (kLegacyPipelineEnabled) {
            // 下车整段序列特征（进入检测区域 → 离开检测区域），与台账中
            // 「已上车未下车」乘客的整段序列逐一比对
            QVector<cv::Mat> feats = evt.trackFeatures;
            if (feats.isEmpty()) {
                if (!evt.enterFeature.empty()) feats.append(evt.enterFeature);
                if (!evt.exitFeature.empty())  feats.append(evt.exitFeature);
            }

            PendingAlight pend;
            pend.evt = evt;
            pend.alightTime = alightTime;
            pend.fallbackId = fallbackId;
            pend.feats = feats;

            // 无需比对的情形（未启用台账 / 无在车乘客 / 无可比对特征 / 无比对线程）
            // → 直接在主线程结算（按回退身份或异常处理）
            if (!m_passengerBook || !m_matchWorker || m_passengerBook->onboardCount() == 0
                || feats.isEmpty()) {
                applyAlightingMatch(pend, -1, 0.0f, false);
                return;
            }

            // ===== 攒批提交：把短时间窗口内的多个下车事件合并成一批，交给后台线程
            //  做「多模态动态加权 + 匈牙利全局匹配」，避免逐个贪心抢占导致的次优配对。=====
            const quint64 jobId = ++m_matchJobCounter;
            m_pendingAlights.insert(jobId, pend);
            m_alightBatchJobs.append(jobId);
            if (m_alightBatchTimer) m_alightBatchTimer->start(ALIGHT_BATCH_WINDOW_MS);
            LOG_INFO(QString("[MainWindow] ★ 下车事件入队待批量比对: jobId=%1, 当前批次=%2, "
                             "在车乘客=%3, 探测帧=%4")
                         .arg(jobId).arg(m_alightBatchJobs.size())
                         .arg(m_passengerBook->onboardCount()).arg(feats.size()));
        } else {
            // 旧链路停用：仅做轻量结算（访客记录 + 日志 + 表格刷新）。
            // 身份判定已由 person_N 目录 + FolderMatchWorker 全图比对完成，结果写 match.txt + reid_matches。
            const int pid = fallbackId;
            if (pid > 0) {
                VisitorRecord &vr = m_visitors[pid];
                vr.visitorId = pid;
                vr.direction = QStringLiteral("alighting");
                vr.exitTime = alightTime;
                vr.regionId = evt.regionId;
                vr.isInside = false;
            }
            if (m_visitorLog) {
                VisitorLog::EventEntry logEntry;
                logEntry.timestamp = alightTime;
                logEntry.eventType = "ALIGHTING";
                logEntry.visitorId = pid;
                logEntry.trackId = evt.trackId;
                logEntry.regionId = evt.regionId;
                logEntry.similarity = 0.0f;
                const QString alightDir = (evt.entrySide == 0 && evt.exitSide == 1)
                        ? QStringLiteral("从上往下 A→B") : QStringLiteral("从下往上 B→A");
                logEntry.details = QString("下车(%1) 乘客P%2，身份由目录全图比对确定(见 match.txt)")
                                       .arg(alightDir).arg(pid);
                m_visitorLog->logEvent(logEntry);
            }
            updateVisitorTable();
            saveSessionState();
            LOG_INFO(QString("[MainWindow] ★ ALIGHTING EVENT(旧链路停用): 乘客P%1, count=%2, 下车时间=%3")
                     .arg(pid).arg(m_alightingCount)
                     .arg(alightTime.toString("yyyy-MM-dd HH:mm:ss")));
        }
    });

    m_elapsedTimer.start();
    m_lastUiUpdate.start();
    m_fpsTimer.start();

    updateCaptureStatus();

    // 程序启动时默认自动开启采集（相机未配置/打开失败时由 onToggleCapture 弹窗提示）
    if (!m_isCapturing) {
        onToggleCapture();
    }
}

// ===== 下车身份比对结果结算（主线程执行）=====
void MainWindow::applyAlightingMatch(const PendingAlight &pend, int bestId,
                                     float similarity, bool matched)
{
    const ZoneEvent &evt = pend.evt;
    const QDateTime alightTime = pend.alightTime;
    const int fallbackId = pend.fallbackId;

    // 判定“刚上车又立即被判下车”的阈值（秒）：停留时长小于该值视为高度可疑
    static constexpr qint64 SUSPECT_DWELL_SECONDS = 30;

    // ===== 结算前快照：在车乘客及其上车时间（用于停留时长 / 误判分析）=====
    QMap<int, QDateTime> onboardBoardTime;
    QString onboardList;
    if (m_passengerBook) {
        const QVector<PassengerBook::BoardingRecord> recs = m_passengerBook->onboardRecords();
        for (const auto &rec : recs) {
            const QDateTime bt = QDateTime::fromString(rec.boardTime, Qt::ISODate);
            onboardBoardTime.insert(rec.passengerId, bt);
            if (!onboardList.isEmpty()) onboardList += QStringLiteral(", ");
            onboardList += QString("P%1@%2").arg(rec.passengerId)
                               .arg(bt.isValid() ? bt.toString("HH:mm:ss") : rec.boardTime);
        }
    }

    // ★ 关键诊断：下车必须「完整穿越」—— 起点在区域下边界之外(B端之下)，
    //   终点在区域上边界之外(A端之上)。
    LOG_INFO(QString("[MainWindow] ★ ALIGHTING 判定依据(完整穿越: 下边界外→上边界外): "
                     "V%1 track=%2 region=%3 区域[topY=%4 bottomY=%5 midY=%6] | "
                     "startY=%7 endY=%8 dy=%9 frames=%10 durationMs=%11 | "
                     "比对结果 bestId=%12 sim=%13 matched=%14 fallbackId=%15 "
                     "在车乘客=[%16] 探测帧=%17")
                 .arg(evt.visitorId).arg(evt.trackId).arg(evt.regionId)
                 .arg(evt.zoneTopNormY, 0, 'f', 4).arg(evt.zoneBottomNormY, 0, 'f', 4)
                 .arg(evt.zoneCenterNormY, 0, 'f', 4)
                 .arg(evt.startNormY, 0, 'f', 4).arg(evt.endNormY, 0, 'f', 4)
                 .arg(evt.moveDy, 0, 'f', 4)
                 .arg(evt.trackFrames).arg(evt.durationMs)
                 .arg(bestId).arg(similarity, 0, 'f', 3)
                 .arg(matched ? "true" : "false").arg(fallbackId)
                 .arg(onboardList.isEmpty() ? QStringLiteral("空") : onboardList)
                 .arg(pend.feats.size()));

    int pid = -1;
    bool matchedFlag = false;
    const float sim = similarity;

    if (m_passengerBook && matched && bestId > 0) {
        // 图像比对命中 → 判定该乘客下车
        pid = bestId;
        matchedFlag = true;
        m_passengerBook->removePassenger(pid);
    } else if (m_passengerBook && fallbackId > 0 && m_passengerBook->isOnboard(fallbackId)) {
        // 比对未达阈值，但跟踪身份指向台账中的在车乘客 → 采用回退身份
        pid = fallbackId;
        matchedFlag = false;
        m_passengerBook->removePassenger(fallbackId);
    } else {
        pid = fallbackId;   // 可能 <=0 或指向非在车乘客 → 异常下车
        matchedFlag = false;
    }

    // 组装下车整段图像（优先整段序列，其次单张快照）
    QVector<cv::Mat> alightImgs = evt.trackImages;
    if (alightImgs.isEmpty()) {
        const cv::Mat &snap = !evt.exitImage.empty() ? evt.exitImage : evt.enterImage;
        if (!snap.empty()) alightImgs.append(snap);
    }

    // 异常乘客下车：未能确定是哪位乘客
    const bool abnormalAlight = (pid <= 0) || (!matchedFlag);

    if (abnormalAlight) {
        const QString abDir = saveAbnormalAlightImages(alightImgs, evt.trackBoxes, alightTime);
        LOG_WARN(QString("[MainWindow] ★ 异常乘客下车：无法确定身份(pid=%1, matched=%2, sim=%3)，"
                         "图像已保存至: %4")
                     .arg(pid).arg(matchedFlag ? "true" : "false")
                     .arg(sim, 0, 'f', 3)
                     .arg(abDir.isEmpty() ? "(保存失败)" : abDir));
        if (m_visitorLog) {
            VisitorLog::EventEntry logEntry;
            logEntry.timestamp = alightTime;
            logEntry.eventType = "ALIGHTING_ABNORMAL";
            logEntry.visitorId = pid;
            logEntry.trackId = evt.trackId;
            logEntry.regionId = evt.regionId;
            logEntry.similarity = sim;
            logEntry.details = QString("异常乘客下车(无法确定身份) 乘客P%1, matched=%2, sim=%3, 图像=%4")
                                   .arg(pid)
                                   .arg(matchedFlag ? "命中" : "未命中")
                                   .arg(sim, 0, 'f', 3)
                                   .arg(abDir);
            m_visitorLog->logEvent(logEntry);
        }
    } else if (!alightImgs.isEmpty()) {
        savePassengerEventImages(alightImgs, evt.trackBoxes, pid, false, alightTime);
    }

    // 保存「进入检测区域 → 离开检测区域」的整段下车视频
    // （异常下车时乘客号未知，落盘为「异常乘客_下车_时间.mp4」）
    if (m_config.reidSaveVideo && !evt.trackFullFrames.isEmpty()) {
        savePassengerEventVideo(evt.trackFullFrames, abnormalAlight ? -1 : pid, false,
                                alightTime, evt.durationMs);
    }

    // 归档本次穿越的逐帧图像：visitor_V{id}/下车_{时间}/
    finalizeVisitorSegment(m_trackToVisitor.value(evt.trackId, -1), false, alightTime);

    // ===== 关键诊断：上车→下车停留时长（定位“刚上车又立即判下车”的误判）=====
    if (pid > 0 && onboardBoardTime.contains(pid)) {
        const QDateTime bt = onboardBoardTime.value(pid);
        if (bt.isValid()) {
            const qint64 dwell = bt.secsTo(alightTime);
            if (dwell >= 0 && dwell < SUSPECT_DWELL_SECONDS) {
                LOG_WARN(QString("[MainWindow] ⚠ 疑似误判：乘客P%1 于 %2 上车，仅 %3 秒后即被判下车"
                                 "（阈值<%4s），matched=%5 sim=%6。请核查 ReID 匹配/穿越判定。")
                             .arg(pid).arg(bt.toString("HH:mm:ss")).arg(dwell)
                             .arg(SUSPECT_DWELL_SECONDS)
                             .arg(matchedFlag ? "命中" : "回退/异常")
                             .arg(sim, 0, 'f', 3));
            } else {
                LOG_INFO(QString("[MainWindow] 乘客P%1 停留时长=%2 秒（上车=%3 → 下车=%4）")
                             .arg(pid).arg(dwell)
                             .arg(bt.toString("HH:mm:ss"))
                             .arg(alightTime.toString("HH:mm:ss")));
            }
        }
    } else if (pid > 0) {
        LOG_WARN(QString("[MainWindow] 下车身份 P%1 不在「上车时间快照」中"
                         "（可能由上/下车编号空间不一致导致，fallbackId=%2）")
                     .arg(pid).arg(fallbackId));
    }

    if (pid > 0) {
        VisitorRecord &vr = m_visitors[pid];
        vr.visitorId = pid;
        vr.direction = QStringLiteral("alighting");
        vr.exitTime = alightTime;
        vr.regionId = evt.regionId;
        vr.isInside = false;
    }

    if (m_visitorLog) {
        VisitorLog::EventEntry logEntry;
        logEntry.timestamp = alightTime;
        logEntry.eventType = "ALIGHTING";
        logEntry.visitorId = pid;
        logEntry.trackId = evt.trackId;
        logEntry.regionId = evt.regionId;
        logEntry.similarity = sim;
        const QString alightDir = (evt.entrySide == 0 && evt.exitSide == 1)
                ? QStringLiteral("从上往下 A→B") : QStringLiteral("从下往上 B→A");
        logEntry.details = QString("下车(%1) 乘客P%2, 与上车图像比对=%3(sim=%4), enter=%5, exit=%6")
                               .arg(alightDir)
                               .arg(pid)
                               .arg(matchedFlag ? "命中" : "回退")
                               .arg(sim, 0, 'f', 3)
                               .arg(evt.enterTime.toString("HH:mm:ss.zzz"))
                               .arg(evt.exitTime.toString("HH:mm:ss.zzz"));
        m_visitorLog->logEvent(logEntry);
    }

    updateVisitorTable();
    LOG_INFO(QString("[MainWindow] ★ ALIGHTING EVENT: 乘客P%1, count=%2, 下车时间=%3, 比对相似度=%4, %5")
                 .arg(pid).arg(m_alightingCount)
                 .arg(alightTime.toString("yyyy-MM-dd HH:mm:ss"))
                 .arg(sim, 0, 'f', 3)
                 .arg(matchedFlag ? "命中" : "回退/异常"));
}

// ===== 下车比对攒批提交：同批多探测一次做「匈牙利全局匹配」=====
void MainWindow::flushAlightBatch()
{
    if (m_alightBatchJobs.isEmpty()) return;

    const QVector<quint64> jobs = m_alightBatchJobs;
    m_alightBatchJobs.clear();

    if (!m_matchWorker || !m_passengerBook) {
        for (quint64 id : jobs)
            if (m_pendingAlights.contains(id))
                applyAlightingMatch(m_pendingAlights.take(id), -1, 0.0f, false);
        return;
    }

    // 组装本批探测（每个 job 一条时序特征序列）
    QVector<QPair<quint64, QVector<cv::Mat>>> probes;
    probes.reserve(jobs.size());
    for (quint64 id : jobs) {
        if (!m_pendingAlights.contains(id)) continue;
        probes.append(qMakePair(id, m_pendingAlights.value(id).feats));
    }
    if (probes.isEmpty()) return;

    // 在车模板快照（含时序融合特征 + 原始帧）
    const QVector<MatchTemplate> templates = m_passengerBook->onboardTemplateBundle();
    if (templates.isEmpty()) {
        for (const auto &pr : probes)
            applyAlightingMatch(m_pendingAlights.take(pr.first), -1, 0.0f, false);
        return;
    }

    LOG_INFO(QString("[MainWindow] ★ 下车身份比对(批量/匈牙利全局匹配)已提交: 探测=%1, 在车乘客=%2")
                 .arg(probes.size()).arg(templates.size()));
    m_matchWorker->submitBatch(probes, templates, m_passengerBook->matchThreshold());
}

MainWindow::~MainWindow()
{
    if (m_isCapturing) {
        onCloseCamera();
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

    // ===== 乘车数据存储：结束行程并关闭数据库 =====
    if (m_tripStore) {
        m_tripStore->endTrip(m_currentTripId, QDateTime::currentDateTime());
        m_tripStore->close();
        delete m_tripStore;
        m_tripStore = nullptr;
    }

    if (m_saveThread) { m_saveThread->quit(); m_saveThread->wait(); }

    // 停止下车身份比对线程
    if (m_matchThread) {
        m_matchThread->quit();
        m_matchThread->wait(3000);
        delete m_matchWorker;
        m_matchWorker = nullptr;
    }

    // 停止上/下车文件夹全图比对线程
    if (m_folderMatchThread) {
        m_folderMatchThread->quit();
        m_folderMatchThread->wait(3000);
        delete m_folderMatchWorker;
        m_folderMatchWorker = nullptr;
    }

    delete m_personFolder;
    m_personFolder = nullptr;

    delete m_webServer;
    m_webServer = nullptr;

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

        if (!m_camera.openCamera(m_config.rtspUrl, m_config.depthRtspUrl)) {
            QMessageBox::critical(this, "错误", "无法打开 RTSP 流:\n" + m_config.rtspUrl);
            return;
        }
        LOG_INFO("Camera opened: " + m_config.rtspUrl);
        if (m_depthView) m_depthView->setVisible(m_camera.hasDepth());
        if (ui->label_depthFps)
            ui->label_depthFps->setText(m_camera.hasDepth() ? QStringLiteral("深度FPS: --")
                                                            : QStringLiteral("深度: 未连接"));
        m_videoMode = false;
        startDetectionPipeline();
    }

    updateCaptureStatus();
}

void MainWindow::startDetectionPipeline()
{
    // 创建线程：RGB 拉流 / 深度拉流 / 检测 三者解耦，各自独立运行
    m_frameSlot = new LatestFrameSlot();
    m_grabThread = new GrabThread(&m_camera, m_frameSlot, "HX-CCD23", this);
    m_depthThread = new DepthGrabThread(&m_camera, "HX-CCD23-Depth", this);
    m_detectThread = new DetectThread(m_frameSlot, "HX-CCD23-Detect", this);

    // 创建检测管线
    m_detector = new PersonDetector();
    m_tracker = new PersonTracker(m_config.trackerIouThreshold, m_config.trackerMaxMissed);
    m_tracker->setKalmanEnabled(m_config.trackerKalmanEnabled);
    m_tracker->setReidMatchingEnabled(m_config.trackerReidEnabled);
    m_tracker->setReidMatchThreshold(m_config.trackerReidThreshold);
    m_tracker->setGalleryMaxAge(m_config.trackerGalleryMaxAgeMs);
    m_reidExtractor = new ReIDExtractor();
    m_reidExtractor->setComponentWeights(m_config.reidWeightColor,
                                         m_config.reidWeightTexture,
                                         m_config.reidWeightShape);
    m_reidExtractor->setDeepReid(m_deepReid, m_config.reidUseDeepReid);
    m_reidExtractor->setBoxExpandForReid(m_config.reidHeadBoxExpand);
    LOG_INFO(QString("[MainWindow] ReID 后端=%1 (deep=%2, headBoxExpand=%3)")
                 .arg(m_reidExtractor->deepEnabled() ? "深度OSNet(512D)" : "手工多模态")
                 .arg(m_config.reidUseDeepReid ? "on" : "off")
                 .arg(m_config.reidHeadBoxExpand, 0, 'f', 2));
    if (m_matchWorker) {
        m_matchWorker->setComponentWeights(m_config.reidWeightColor,
                                           m_config.reidWeightTexture,
                                           m_config.reidWeightShape);
    }

    if (m_config.reidUseBackground) {
        if (!m_bgModel) m_bgModel = new BackgroundModel();
        m_bgModel->reset();
        m_reidExtractor->setBackgroundModel(m_bgModel);
    } else {
        m_reidExtractor->setBackgroundModel(nullptr);
    }

    if (m_passengerBook) {
        m_passengerBook->setReidExtractor(m_reidExtractor);
        m_passengerBook->setMatchThreshold(effectiveReidThreshold());
    }

    QString modelPath = pickDetectorModelPath();
    if (!QFileInfo(modelPath).isAbsolute()) {
        modelPath = QCoreApplication::applicationDirPath() + "/" + modelPath;
    }
    m_detector->setDnnUseCuda(m_config.detectorDnnUseCuda);
    m_detector->setDnnInputSize(m_config.detectorDnnInputSize);
    m_detector->setLetterbox(m_config.detectorLetterbox);

    if (!m_config.detectorModelUrl.isEmpty()) {
        downloadDetectorModel(m_config.detectorModelUrl, m_config.detectorOnnxPath);
    }

    bool modelLoaded = m_detector->loadModel(modelPath, effectiveDetectorConf(), m_config.detectorNmsThreshold);
    if (!modelLoaded && modelPath.endsWith(".engine", Qt::CaseInsensitive)) {
        QString onnxPath = modelPath.left(modelPath.lastIndexOf('.')) + ".onnx";
        if (QFileInfo::exists(onnxPath)) {
            LOG_WARN("[MainWindow] Engine load failed, falling back to ONNX: " + onnxPath);
            modelLoaded = m_detector->loadModel(onnxPath, effectiveDetectorConf(), m_config.detectorNmsThreshold);
            if (modelLoaded) modelPath = onnxPath;
        }
    }
    if (modelLoaded) {
        QString backend = m_detector->isTrtBackend() ? "TensorRT" : "OpenCV DNN";
        LOG_INFO(QString("[MainWindow] YOLO model loaded [%1]: %2").arg(backend, modelPath));
    } else {
        LOG_WARN("[MainWindow] Failed to load any YOLO model (detection disabled)");
    }
    applyDetectorFilters(m_detector);

    m_detectThread->setDetector(m_detector);
    m_detectThread->setTracker(m_tracker);
    m_detectThread->setReidExtractor(m_reidExtractor);
    m_detectThread->setBackgroundModel(m_config.reidUseBackground ? m_bgModel : nullptr);

    {
        QString clipDir = m_config.personClipPath;
        if (!QDir(clipDir).isAbsolute())
            clipDir = QCoreApplication::applicationDirPath() + "/" + clipDir;
        m_detectThread->setPersonClipRecording(m_config.personClipEnabled, clipDir,
                                               m_config.personClipGapMs, m_config.personClipFps);
    }

    connect(m_grabThread, &GrabThread::deviceLost, this, &MainWindow::onGrabThreadDeviceLost);
    connect(m_grabThread, &GrabThread::frameGrabbed, this, &MainWindow::onFrameGrabbed);
    connect(m_detectThread, &DetectThread::frameProcessed, this, &MainWindow::onFrameProcessed);
    connect(m_detectThread, &DetectThread::rawFrameGrabbed, this, [this](const cv::Mat &frame) {
        QMutexLocker locker(&m_rawFrameMutex);
        m_currentRawFrame = frame.clone();
    });
    connect(m_depthThread, &DepthGrabThread::depthFrameGrabbed, this, &MainWindow::onDepthFrameGrabbed);
    connect(m_grabThread, &GrabThread::videoFinished, this, [this]() {
        LOG_INFO("[MainWindow] 视频检测完成");
        onCloseCamera();
        m_isCapturing = false;
        m_videoMode = false;
        restorePersonFolderFromVideo();
        ui->btnToggle->setText("启动采集");
        ui->btnSave->setEnabled(false);
        updateCaptureStatus();
    });

    m_grabThread->start();
    m_depthThread->start();
    m_detectThread->start();
    LOG_INFO("[MainWindow] RGB grab / depth grab / detect threads started");

    if (m_zoneMonitor) {
        m_zoneMonitor->setReidExtractor(m_reidExtractor);
        m_zoneMonitor->setReidMatchThreshold(effectiveReidThreshold());
        m_zoneMonitor->setChannelZones(m_regionManager->regions());
    }

    m_fpsTimer.restart();
    m_fpsFrameCount = 0;
    m_currentFps = 0.0;

    m_depthFpsTimer.restart();
    m_depthFpsFrameCount = 0;
    m_currentDepthFps = 0.0;

    m_isCapturing = true;
    ui->btnToggle->setText("停止采集");
    ui->btnSave->setEnabled(true);
    updateCameraStatus(true);
    m_elapsedTimer.start();
}

void MainWindow::onSelectVideo()
{
    const QString path = QFileDialog::getOpenFileName(
        this, QStringLiteral("选择视频文件"), QString(),
        QStringLiteral("视频文件 (*.mp4 *.avi *.mkv *.mov *.flv *.wmv *.m4v *.mpg);;所有文件 (*)"));
    if (path.isEmpty()) return;

    if (m_isCapturing) {
        onCloseCamera();
    }

    if (!m_camera.openVideoFile(path)) {
        QMessageBox::critical(this, "错误", "无法打开视频文件:\n" + path);
        return;
    }

    m_videoMode = true;
    switchPersonFolderToVideo(path);
    startDetectionPipeline();

    LOG_INFO(QString("[MainWindow] 视频检测已启动: %1").arg(path));
}

void MainWindow::switchPersonFolderToVideo(const QString &videoPath)
{
    QString base = m_config.savePath;
    if (!QDir(base).isAbsolute())
        base = QCoreApplication::applicationDirPath() + "/" + base;

    const QFileInfo fi(videoPath);
    const QString videoDir = QDir(base).filePath(
        QStringLiteral("%1_%2")
            .arg(fi.completeBaseName())
            .arg(QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd_HHmmss"))));

    if (m_personFolder) {
        delete m_personFolder;
        m_personFolder = nullptr;
    }
    m_personFolder = new PersonFolder(videoDir);
    m_personFolder->setSaveEveryN(m_config.personSaveEveryN);

    // 视频模式从空会话开始记录（不继承实时会话状态）
    m_personEventRows.clear();
    m_boardFolders.clear();
    updateVisitorTable();

    LOG_INFO(QString("[MainWindow] 视频模式人员归档目录: %1").arg(videoDir));
}

void MainWindow::restorePersonFolderFromVideo()
{
    if (!m_videoMode) return;

    QString base = m_config.savePath;
    if (!QDir(base).isAbsolute())
        base = QCoreApplication::applicationDirPath() + "/" + base;

    if (m_personFolder) {
        delete m_personFolder;
        m_personFolder = nullptr;
    }
    m_personFolder = new PersonFolder(base);
    m_personFolder->setSaveEveryN(m_config.personSaveEveryN);

    // 恢复实时会话的人员记录与上车目录登记
    loadSessionState();

    LOG_INFO("[MainWindow] 视频检测结束，已恢复实时人员归档目录");
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
    if (m_depthThread) {
        m_depthThread->stop();
        m_depthThread->wait();
        delete m_depthThread;
        m_depthThread = nullptr;
    }
    if (m_detectThread) {
        m_detectThread->stop();
        m_detectThread->wait();
        delete m_detectThread;
        m_detectThread = nullptr;
    }
    delete m_frameSlot;
    m_frameSlot = nullptr;

    // 清理检测管线
    delete m_detector;
    m_detector = nullptr;
    delete m_tracker;
    m_tracker = nullptr;
    delete m_reidExtractor;
    m_reidExtractor = nullptr;

    delete m_deepReid;
    m_deepReid = nullptr;

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

void MainWindow::onDepthFrameGrabbed(QImage img)
{
    // 更新深度 FPS（仅统计有效帧）
    if (!img.isNull()) {
        m_depthFpsFrameCount++;
        const qint64 fpsElapsed = m_depthFpsTimer.elapsed();
        if (fpsElapsed >= 1000) {
            m_currentDepthFps = m_depthFpsFrameCount * 1000.0 / fpsElapsed;
            m_depthFpsFrameCount = 0;
            m_depthFpsTimer.restart();
        }
        if (ui->label_depthFps)
            ui->label_depthFps->setText(QString("深度FPS: %1").arg(m_currentDepthFps, 0, 'f', 1));
    }

    if (!m_depthView || img.isNull())
        return;
    if (!m_depthView->isVisible())
        m_depthView->setVisible(true);
    m_depthView->setImage(img);

    // 缓存最新深度伪彩帧，供「保存 RGB 图像时同步保存深度图」使用
    {
        QMutexLocker lk(&m_depthImageMutex);
        m_currentDepthImage = img;
    }
    m_depthView->resetView();
}

void MainWindow::saveCorrespondingDepth(const QString &dirPath, const QString &baseName)
{
    QMutexLocker lk(&m_depthImageMutex);
    if (m_currentDepthImage.isNull()) return;
    const QString depthDir = QDir(dirPath).filePath(QStringLiteral("深度图"));
    if (!QDir().mkpath(depthDir)) {
        LOG_WARN("[MainWindow] 无法创建深度图目录: " + depthDir);
        return;
    }
    const QString depthPath = QDir(depthDir).filePath(baseName + ".jpg");
    if (!m_currentDepthImage.save(depthPath, "JPG", 95)) {
        LOG_WARN("[MainWindow] 深度图保存失败: " + depthPath);
    }
}

void MainWindow::onFrameGrabbed(QImage img)
{
    Q_UNUSED(img);

    // 以实际拉流帧率为准统计 RGB FPS（来自 RGB 采集线程，反映真实拉流速率约 30fps）
    m_fpsFrameCount++;
    qint64 fpsElapsed = m_fpsTimer.elapsed();
    if (fpsElapsed >= 1000) {
        m_currentFps = m_fpsFrameCount * 1000.0 / fpsElapsed;
        m_fpsFrameCount = 0;
        m_fpsTimer.restart();
    }
    ui->label_fps->setText(QString("FPS: %1").arg(m_currentFps, 0, 'f', 1));
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
            fl.visitorId = m_trackToVisitor.value(p.trackId, -1);
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

    // ID 标签格式：显示 visitorId（如果有）否则显示 trackId
    int displayId = m_trackToVisitor.value(person.trackId, person.trackId);
    std::string label = "V" + std::to_string(displayId);

    // 位置：边界框左上方
    int x = person.bbox.x;
    int y = person.bbox.y - 10;
    if (y < 20) y = person.bbox.y + 20;

    double fontScale = 0.8;
    int thickness = 2;
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);

    // 绘制背景矩形（半透明黑色）
    cv::Rect bgRect(x - 2, y - textSize.height - 4, textSize.width + 8, textSize.height + 8);
    // 确保不超出图像范围
    bgRect &= cv::Rect(0, 0, frame.cols, frame.rows);
    // 关键防护：bgRect 可能为空（人员在图像边缘外），此时 frame(bgRect) 为空 ROI，
    // cv::addWeighted 会因尺寸不匹配抛 cv::Exception 导致崩溃，必须跳过。
    if (bgRect.width > 0 && bgRect.height > 0) {
        cv::Mat roi = frame(bgRect);
        if (!roi.empty()) {
            cv::Mat overlay = roi.clone();
            cv::rectangle(overlay, cv::Point(0, 0), cv::Point(bgRect.width, bgRect.height),
                          cv::Scalar(0, 0, 0), -1);
            if (overlay.size() == roi.size() && overlay.type() == roi.type())
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

    // 直接使用 alpha 混合（尺寸/类型不一致时跳过，避免抛异常崩溃）
    if (textLayer.size() == frame.size() && textLayer.type() == frame.type())
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
        // 【视角反转】默认 TopToBottom 下「上车」= 从下往上：
        // 前一帧在区域下方边界之外，当前帧在区域内或上方
        return (prevNormY > maxY) && (currNormY <= maxY);
    } else {
        // BottomToTop 取反：上车 = 从上往下：
        // 前一帧在区域上方边界之外，当前帧在区域内或下方
        return (prevNormY < minY) && (currNormY >= minY);
    }
}

// ===== ReID 匹配或创建新访客 =====
int MainWindow::matchOrCreateVisitor(const TrackedPerson &person, const cv::Mat &rawFrame, int regionId)
{
    // 如果已有 visitorId 映射，直接返回
    if (m_trackToVisitor.contains(person.trackId)) {
        return m_trackToVisitor[person.trackId];
    }

    // 提取 ReID 特征
    cv::Mat reidFeature = person.reidFeature;
    if (reidFeature.empty() && !rawFrame.empty() && m_reidExtractor) {
        reidFeature = m_reidExtractor->extract(rawFrame, person.bbox);
    }

    // 尝试与历史离开的访客匹配
    int matchedVisitorId = -1;
    float bestSimilarity = 0.0f;

    for (auto it = m_visitors.begin(); it != m_visitors.end(); ++it) {
        VisitorRecord &vr = it.value();
        // 只匹配不在区域内且有关联特征的访客
        if (!vr.isInside && !vr.reidFeature.empty() && !reidFeature.empty()) {
            float similarity = m_reidExtractor->compare(reidFeature, vr.reidFeature);
            if (similarity > bestSimilarity && similarity >= m_config.reidMatchThreshold) {
                bestSimilarity = similarity;
                matchedVisitorId = vr.visitorId;
            }
        }
    }

    if (matchedVisitorId > 0) {
        // 匹配到历史访客 → 重入
        VisitorRecord &vr = m_visitors[matchedVisitorId];
        vr.isInside = true;
        vr.currentTrackId = person.trackId;
        vr.enterTime = QDateTime::currentDateTime();
        vr.exitTime = QDateTime();
        if (!rawFrame.empty()) {
            vr.enterImage = rawFrame.clone();
        }
        // 更新 ReID 特征（取最新特征）
        if (!reidFeature.empty()) {
            vr.reidFeature = reidFeature.clone();
        }

        m_trackToVisitor[person.trackId] = matchedVisitorId;

        LOG_INFO(QString("[MainWindow] ReID RE-ENTRY: Visitor V%1 matched (similarity=%2) with track %3")
                     .arg(matchedVisitorId).arg(bestSimilarity, 0, 'f', 2).arg(person.trackId));

        // 记录重入事件
        if (m_visitorLog) {
            VisitorLog::EventEntry evt;
            evt.timestamp = QDateTime::currentDateTime();
            evt.eventType = "REENTRY";
            evt.visitorId = matchedVisitorId;
            evt.trackId = person.trackId;
            evt.regionId = regionId;
            evt.similarity = bestSimilarity;
            evt.details = QString("ReID re-entry match");
            m_visitorLog->logEvent(evt);
        }

        return matchedVisitorId;
    }

    // 创建新访客
    int newVisitorId = m_nextVisitorId++;
    VisitorRecord vr;
    vr.visitorId = newVisitorId;
    vr.reidFeature = reidFeature.empty() ? cv::Mat() : reidFeature.clone();
    vr.enterTime = QDateTime::currentDateTime();
    vr.isInside = true;
    vr.currentTrackId = person.trackId;
    vr.regionId = regionId;
    if (!rawFrame.empty()) {
        vr.enterImage = rawFrame.clone();
    }

    m_visitors[newVisitorId] = vr;
    m_trackToVisitor[person.trackId] = newVisitorId;

    LOG_INFO(QString("[MainWindow] New visitor V%1 created (track %2)").arg(newVisitorId).arg(person.trackId));

    // 记录新访客事件
    if (m_visitorLog) {
        VisitorLog::EventEntry evt;
        evt.timestamp = QDateTime::currentDateTime();
        evt.eventType = "ENTER";
        evt.visitorId = newVisitorId;
        evt.trackId = person.trackId;
        evt.regionId = regionId;
        evt.similarity = 0.0f;
        evt.details = "New visitor created";
        m_visitorLog->logEvent(evt);
    }

    return newVisitorId;
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
    vr.currentTrackId = -1;

    // 更新 ReID 特征（离开时保留最新特征用于后续匹配）
    // 特征已在进入时更新

    LOG_INFO(QString("[MainWindow] Visitor V%1 EXITED at %2")
                 .arg(visitorId).arg(vr.exitTime.toString("HH:mm:ss.zzz")));

    // 记录离开事件
    if (m_visitorLog) {
        VisitorLog::EventEntry evt;
        evt.timestamp = QDateTime::currentDateTime();
        evt.eventType = "EXIT";
        evt.visitorId = visitorId;
        evt.trackId = vr.currentTrackId;
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
    for (const PersonEventRow &r : m_personEventRows) {
        if (row >= ui->tableVisitors->rowCount()) {
            ui->tableVisitors->insertRow(row);
        }

        // 乘客ID（与图片目录 person_N_yyyyMMdd_HHmmss 对应）
        const QString personLabel = m_personFolder
                ? m_personFolder->personName(r.personId)
                : QStringLiteral("person_%1").arg(r.personId);
        QTableWidgetItem *idItem = ui->tableVisitors->item(row, 0);
        if (!idItem) {
            idItem = new QTableWidgetItem();
            ui->tableVisitors->setItem(row, 0, idItem);
        }
        idItem->setText(personLabel);
        idItem->setTextAlignment(Qt::AlignCenter);
        idItem->setForeground(QBrush(QColor(0, 229, 255)));

        // 状态：只有 上车 / 下车
        QTableWidgetItem *statusItem = ui->tableVisitors->item(row, 1);
        if (!statusItem) {
            statusItem = new QTableWidgetItem();
            ui->tableVisitors->setItem(row, 1, statusItem);
        }
        const bool boarding = (r.status == QStringLiteral("上车"));
        statusItem->setText(boarding ? "上车" : "下车");
        statusItem->setTextAlignment(Qt::AlignCenter);
        statusItem->setForeground(QBrush(boarding ? QColor(0, 230, 118) : QColor(255, 171, 64)));

        // 乘客匹配ID：只显示匹配到的对方 person 完整名称（内容过长用 tooltip 补全）
        QTableWidgetItem *matchItem = ui->tableVisitors->item(row, 2);
        if (!matchItem) {
            matchItem = new QTableWidgetItem();
            ui->tableVisitors->setItem(row, 2, matchItem);
        }
        if (r.matchedPersonId > 0) {
            const QString matchedLabel = m_personFolder
                    ? m_personFolder->personName(r.matchedPersonId)
                    : QStringLiteral("person_%1").arg(r.matchedPersonId);
            matchItem->setText(matchedLabel);
            matchItem->setToolTip(matchedLabel);
            matchItem->setForeground(QBrush(QColor(186, 104, 200)));
        } else {
            matchItem->setText("--");
            matchItem->setForeground(QBrush(QColor(120, 120, 120)));
        }
        matchItem->setTextAlignment(Qt::AlignCenter);

        // 相似度得分
        QTableWidgetItem *simItem = ui->tableVisitors->item(row, 3);
        if (!simItem) {
            simItem = new QTableWidgetItem();
            ui->tableVisitors->setItem(row, 3, simItem);
        }
        if (r.matchedPersonId > 0) {
            // 相似度按百分制（0~100）显示，保留一位小数
            simItem->setText(QString::number(r.similarity * 100.0f, 'f', 1));
            simItem->setForeground(QBrush(QColor(255, 213, 79)));
        } else {
            simItem->setText("--");
            simItem->setForeground(QBrush(QColor(120, 120, 120)));
        }
        simItem->setTextAlignment(Qt::AlignCenter);

        row++;
    }

    // 移除多余的行
    while (ui->tableVisitors->rowCount() > row) {
        ui->tableVisitors->removeRow(ui->tableVisitors->rowCount() - 1);
    }

    // 乘客ID / 状态 / 匹配ID 列宽随内容自适应（ID 已改为 person_N_yyyyMMdd_HHmmss 长格式）
    ui->tableVisitors->resizeColumnToContents(0);
    ui->tableVisitors->resizeColumnToContents(1);
    ui->tableVisitors->resizeColumnToContents(2);

    ui->tableVisitors->blockSignals(false);
}

void MainWindow::refreshRegionOverlays()
{
    if (!m_imageView || !m_regionManager) return;

    QVector<QVector<QPointF>> polys;
    QVector<QPair<QPointF, QPointF>> lines;
    for (const auto &r : m_regionManager->regions()) {
        if (r.polygon.size() >= 3) polys.append(r.polygon);
        QPointF a, b;
        if (RegionManager::regionDoorLine(r, a, b)) lines.append(qMakePair(a, b));
    }
    m_imageView->setRegionPolygons(polys);
    m_imageView->setDoorLines(lines);
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

// 是否绘制某条轨迹的检测框。
// 背景：PersonTracker::update() 每帧会把「未匹配到真实检测」的轨迹用卡尔曼预测
//       外推 bbox 并在 maxMissed 帧内保留；旧绘制逻辑无差别地画这些预测框，
//       导致实时画面比同一段视频的离线判定多出一个框（日志 dets=1, persons=2）。
bool MainWindow::shouldDrawTrack(const TrackedPerson &person) const
{
    if (person.bbox.width <= 0 || person.bbox.height <= 0)
        return false;

    // 1) 确认帧：轨迹累计命中达到阈值后才绘制，抑制新建/偶发轨迹的闪现
    if (m_config.displayMinConfirmFrames > 1 &&
        person.age < m_config.displayMinConfirmFrames)
        return false;

    // 2) 本帧被真实检测到 → 绘制
    if (person.detected)
        return true;

    // 3) 本帧未匹配到检测（bbox 为卡尔曼预测外推）：
    //    默认不绘制，彻底消除“同一人两个框”的幽灵框；
    //    若显式允许，则仅在丢失帧数不超过阈值时短暂保留。
    if (m_config.displayOnlyDetectedTracks)
        return false;
    return person.missedFrames <= m_config.displayMaxMissedFrames;
}

void MainWindow::onFrameProcessed(QImage img, QVector<TrackedPerson> persons)
{
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

    // ===== 乘客出现即分配 person_N 并持续保存原始整帧（供后台整目录全图比对）=====
    savePersonAppearanceFrames(persons, rawFrame);

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
                    int visitorId = matchOrCreateVisitor(p, rawFrame, currentRegionId);
                    p.visitorId = visitorId;

                    // 保存进入图像（仅保留内存快照；不再落盘 reid 目录，该目录不参与比对）
                    if (!rawFrame.empty()) {
                        p.enterImage = rawFrame.clone();
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
                if (kLegacyPipelineEnabled && !rawFrame.empty() && p.visitorId > 0) {
                    saveVisitorFrame(rawFrame, p.bbox, p.visitorId, p.trackId);
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

                    // 更新访客离开记录
                    int visitorId = m_trackToVisitor.value(p.trackId, -1);
                    if (visitorId > 0) {
                        updateVisitorExit(visitorId, rawFrame);
                        p.visitorId = visitorId;
                    }

                    // 保存离开图像（仅保留内存快照；不再落盘 reid 目录，该目录不参与比对）
                    if (!rawFrame.empty()) {
                        p.exitImage = rawFrame.clone();
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

    // 仅当到达刷新间隔时才做“重绘 + 显示”。
    // 否则跳过全部全帧绘制（区域/框/标签），直接返回并解除采集线程背压，
    // 避免每帧都做昂贵绘制导致 UI 队列积压、画面越来越滞后。
    if (m_lastUiUpdate.elapsed() < UI_UPDATE_INTERVAL_MS) {
        m_currentPersonCount = personInRegionCount;
        if (m_detectThread) m_detectThread->setUiBusy(false);
        return;
    }

    // ===== 在 cv::Mat 上绘制叠加信息 =====
    // 性能：img 为队列连接传入的深拷贝副本，可直接在其缓冲上就地转换与绘制，
    //       省去一次整帧 clone（1280x720 约 2.7MB / 1080p 约 6MB）。
    QImage rgbImg = img.convertToFormat(QImage::Format_RGB888);
    cv::Mat displayFrame(rgbImg.height(), rgbImg.width(), CV_8UC3,
                         const_cast<uchar*>(rgbImg.bits()), rgbImg.bytesPerLine());
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
        if (overlay.size() == displayFrame.size() && overlay.type() == displayFrame.type())
            cv::addWeighted(overlay, 0.2, displayFrame, 0.8, 0, displayFrame);

        // 绿色边框
        cv::polylines(displayFrame, std::vector<std::vector<cv::Point>>{polyPoints},
                      true, cv::Scalar(0, 255, 0), 2);

        // 绘制区域名称 + 方向标识
        if (!region.name.isEmpty()) {
            cv::Point namePos(polyPoints[0].x + 5, polyPoints[0].y + 15);
            QString dirStr = (region.entryDirection == EntryDirection::TopToBottom) ? " [↓]" : " [↑]";
            // 用 Qt 渲染（支持中文）；cv::putText 会把中文渲染成 "？？？"
            OverlayText::draw(displayFrame, region.name + dirStr, namePos,
                              QColor(255, 255, 255), 16, true);
        }

        // 绘制方向箭头
        drawDirectionArrowOnFrame(displayFrame, region, imgW, imgH);

        // ===== 绘制虚拟门线（DoorCounter v6 判定基准）=====
        {
            QPointF la, lb;
            if (RegionManager::regionDoorLine(region, la, lb)) {
                const cv::Point pa(static_cast<int>(la.x() * imgW), static_cast<int>(la.y() * imgH));
                const cv::Point pb(static_cast<int>(lb.x() * imgW), static_cast<int>(lb.y() * imgH));
                const cv::Scalar lineColor(0, 220, 255);   // BGR: 亮青

                // 外发光底色 + 主体实线
                cv::line(displayFrame, pa, pb, cv::Scalar(120, 120, 0), 7, cv::LINE_AA);
                cv::line(displayFrame, pa, pb, lineColor, 3, cv::LINE_AA);

                // 两端点标记
                cv::circle(displayFrame, pa, 5, lineColor, -1, cv::LINE_AA);
                cv::circle(displayFrame, pb, 5, lineColor, -1, cv::LINE_AA);

                // 中点法向刻度（示意“跨越门线”）
                const cv::Point2f d(static_cast<float>(pb.x - pa.x),
                                    static_cast<float>(pb.y - pa.y));
                const float len = std::sqrt(d.x * d.x + d.y * d.y);
                if (len > 1e-3f) {
                    const cv::Point2f nrm(-d.y / len, d.x / len);
                    const cv::Point2f mid((pa.x + pb.x) * 0.5f, (pa.y + pb.y) * 0.5f);
                    cv::line(displayFrame,
                             cv::Point(mid.x - nrm.x * 12, mid.y - nrm.y * 12),
                             cv::Point(mid.x + nrm.x * 12, mid.y + nrm.y * 12),
                             lineColor, 2, cv::LINE_AA);
                }

                // 标注：门线 + 下车方向（TopToBottom=向下穿越=下车；BottomToTop=向上穿越=下车）
                const QString lbl = (region.entryDirection == EntryDirection::TopToBottom)
                                        ? QStringLiteral("DOOR 下车↓") : QStringLiteral("DOOR 下车↑");
                // 用 Qt 渲染（支持中文）；cv::putText 会把「上车/下车」渲染成 "？？？"
                OverlayText::draw(displayFrame, lbl, cv::Point(pa.x + 8, pa.y - 8),
                                  QColor(0, 220, 255), 20, true);
            }
        }
    }

    // 2. 绘制活跃人员边界框 / 3. ID 标签 / 4. 淡出标签
    // 统一以异常兜底，任何 OpenCV 绘制异常都不得逃逸导致程序崩溃。
    try {
        for (const auto &p : persons) {
            // 显示过滤：跳过非真实检测的卡尔曼幽灵框（详见 shouldDrawTrack）
            if (!shouldDrawTrack(p)) continue;

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
    } catch (const cv::Exception &e) {
        LOG_ERR(QString("[MainWindow] cv::Exception while drawing overlays: %1")
                    .arg(QString::fromStdString(e.what())));
    }

    // 转换回 QImage 用于显示
    // 性能：就地在同一缓冲上做 BGR→RGB（省去一次整帧分配），随后 copy() 得到
    //       自有内存的 QImage（当前帧缓存 / 异步保存需要数据在函数返回后仍有效）。
    cv::cvtColor(displayFrame, displayFrame, cv::COLOR_BGR2RGB);
    QImage displayImg(displayFrame.data, displayFrame.cols, displayFrame.rows,
                      static_cast<int>(displayFrame.step), QImage::Format_RGB888);
    displayImg = displayImg.copy();

    // 显示（已在上方节流判定通过）
    if (m_imageView) m_imageView->setImage(displayImg);
    m_lastUiUpdate.restart();

    // 更新人数显示（FPS 已由 onFrameGrabbed 按拉流帧率更新）
    int boardingCnt = m_zoneMonitor ? m_zoneMonitor->boardingCount() : m_boardingCount;
    int alightingCnt = m_zoneMonitor ? m_zoneMonitor->alightingCount() : m_alightingCount;
    ui->label_personCount->setText(QString("区域人数: %1 | 上车: %2 | 下车: %3 | 车厢内: %4")
                                       .arg(personInRegionCount)
                                       .arg(boardingCnt)
                                       .arg(alightingCnt)
                                       .arg(m_onboardCount));
    m_currentPersonCount = personInRegionCount;

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

    // 解除采集线程背压标志，允许投递下一帧显示
    if (m_detectThread) m_detectThread->setUiBusy(false);
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
        saveCorrespondingDepth(personDir, QString("%1_%2").arg(type.toLower()).arg(timestamp));
    } else {
        LOG_WARN(QString("[MainWindow] Failed to save ReID image: %1").arg(filePath));
    }
}

void MainWindow::saveVisitorFrame(const cv::Mat &rawFrame, const cv::Rect &personBbox,
                                  int visitorId, int trackId)
{
    if (rawFrame.empty() || visitorId <= 0) return;

    // ===== 有效性校验：剔除「无人 / 无效框 / 框大部分越界」的空白存档 =====
    if (personBbox.width < 20 || personBbox.height < 40) return;   // 过小框（误检/远景）
    const cv::Rect frameRect(0, 0, rawFrame.cols, rawFrame.rows);
    const cv::Rect safeBbox = personBbox & frameRect;
    if (safeBbox.empty()) return;
    const double insideRatio = static_cast<double>(safeBbox.area()) /
                               std::max(1.0, static_cast<double>(personBbox.area()));
    if (insideRatio < 0.6) return;                                 // 框主体在画面外

    // ===== 节流：每 3 帧仅存 1 张，避免一次穿越产生成百上千张重复帧 =====
    static constexpr int SAVE_EVERY_N = 3;
    int &tick = m_visitorFrameTick[visitorId];
    if ((tick++ % SAVE_EVERY_N) != 0) return;

    // ===== 目录 =====
    // save_images/{日期}/visitor_V{id}/_pending_HHmmss_t{track}/
    // Inside 期间先落到「待定」目录，上/下车事件确定方向后由
    // finalizeVisitorSegment() 重命名为 上车_时间 / 下车_时间，
    // 这样同一次穿越独立成目录，不会把不同乘客混在一起。
    QString basePath = m_config.savePath;
    if (!QDir(basePath).isAbsolute()) {
        basePath = QCoreApplication::applicationDirPath() + "/" + basePath;
    }
    const QString dateStr = QDateTime::currentDateTime().toString("yyyyMMdd");
    const QString visitorDir = QString("%1/%2/visitor_V%3")
                                   .arg(basePath).arg(dateStr).arg(visitorId);

    QString &pending = m_visitorPendingDir[visitorId];
    if (pending.isEmpty() || !QDir(pending).exists()) {
        const QString seg = QString("_pending_%1_t%2")
                                .arg(QDateTime::currentDateTime().toString("HHmmss"))
                                .arg(trackId);
        pending = QDir(visitorDir).filePath(seg);
        if (!QDir().mkpath(pending)) {
            pending.clear();
            LOG_WARN(QString("[MainWindow] 无法创建访客图像目录: %1").arg(pending));
            return;
        }
        m_visitorSegmentSeq[visitorId] = 0;
    }

    // 裁剪人物区域（带 padding）
    const int padX = static_cast<int>(safeBbox.width * 0.3);
    const int padY = static_cast<int>(safeBbox.height * 0.3);
    const int x1 = std::max(0, safeBbox.x - padX);
    const int y1 = std::max(0, safeBbox.y - padY);
    const int x2 = std::min(rawFrame.cols, safeBbox.x + safeBbox.width + padX);
    const int y2 = std::min(rawFrame.rows, safeBbox.y + safeBbox.height + padY);
    if (x2 <= x1 || y2 <= y1) return;

    cv::Mat crop = rawFrame(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
    if (crop.empty()) return;

    // 在裁剪图上绘制 visitor ID / trackId / 时间戳
    const QString ts = QDateTime::currentDateTime().toString("HHmmss_zzz");
    const std::string label = QString("V%1 t%2 %3").arg(visitorId).arg(trackId).arg(ts).toStdString();
    cv::putText(crop, label, cv::Point(10, 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

    // 文件名：V{id}_t{track}_{序号}_{时间}.jpg（方向由所在目录体现）
    const int seq = ++m_visitorSegmentSeq[visitorId];
    const QString baseName = QString("V%1_t%2_%3_%4")
                                 .arg(visitorId).arg(trackId)
                                 .arg(seq, 4, 10, QChar('0')).arg(ts);
    const QString fileName = QDir(pending).filePath(baseName + ".jpg");
    if (!cv::imwrite(fileName.toStdString(), crop)) {
        LOG_WARN(QString("[MainWindow] 访客帧保存失败: %1").arg(fileName));
    } else {
        saveCorrespondingDepth(pending, baseName);
    }
}

void MainWindow::finalizeVisitorSegment(int visitorId, bool boarding, const QDateTime &time)
{
    if (visitorId <= 0) return;

    auto it = m_visitorPendingDir.find(visitorId);
    if (it == m_visitorPendingDir.end() || it.value().isEmpty()) return;

    const QString pending = it.value();
    m_visitorPendingDir.erase(it);          // 先清理状态，避免后续异常导致悬挂
    m_visitorFrameTick.remove(visitorId);
    m_visitorSegmentSeq.remove(visitorId);

    QDir pendingDir(pending);
    if (!pendingDir.exists()) return;

    const QDateTime t = time.isValid() ? time : QDateTime::currentDateTime();
    const QString prefix = boarding ? QStringLiteral("上车") : QStringLiteral("下车");
    const QString baseName = QString("%1_%2").arg(prefix).arg(t.toString("yyyyMMdd_HHmmss"));

    const QString parent = QFileInfo(pending).absolutePath();
    QString target = QDir(parent).filePath(baseName);
    int suffix = 1;
    while (QFileInfo::exists(target)) {
        target = QDir(parent).filePath(QString("%1_%2").arg(baseName).arg(suffix++));
    }

    if (QDir().rename(pending, target)) {
        const int count = QDir(target).entryList(QDir::Files).size();
        LOG_INFO(QString("[MainWindow] 访客V%1 本次穿越图像已归档: %2 (%3 张)")
                     .arg(visitorId).arg(target).arg(count));
    } else {
        LOG_WARN(QString("[MainWindow] 归档访客图像目录失败: %1 -> %2")
                     .arg(pending, target));
    }
}

void MainWindow::initPassengerBook()
{
    m_passengerBook = new PassengerBook();
    m_passengerBook->setMatchThreshold(m_config.reidMatchThreshold);

    // 台账与访客记录同目录，便于统一管理/清理
    QString dir = m_config.visitorLogPath;
    if (!QDir(dir).isAbsolute()) {
        dir = QCoreApplication::applicationDirPath() + "/" + dir;
    }
    m_passengerBook->setStorageDir(dir);
    m_passengerBook->load();
}

// ===== 乘车数据存储（每站上下车人数 + 车厢内人数）=====
void MainWindow::initTripStore()
{
    // 默认 SQLite（零依赖、随程序落盘）；如需大规模车队可切换 "postgresql" 并传入连接参数。
    m_tripStore = TripDataStore::create(QStringLiteral("sqlite"));

    // 允许通过配置覆盖后端（app_config 可选字段；缺省 sqlite）
    const QString backend = m_config.tripDbBackend.isEmpty()
                                ? QStringLiteral("sqlite") : m_config.tripDbBackend;
    if (backend.compare(m_tripStore->backendName(), Qt::CaseInsensitive) != 0) {
        delete m_tripStore;
        m_tripStore = TripDataStore::create(backend);
    }

    QVariantMap opts;
    if (m_tripStore->backendName() == QLatin1String("sqlite")) {
        QString dbPath = m_config.tripDbPath;
        if (dbPath.isEmpty()) dbPath = QStringLiteral("trip_data/bus_trips.db");
        if (!QDir(dbPath).isAbsolute())
            dbPath = QCoreApplication::applicationDirPath() + "/" + dbPath;
        opts.insert(QStringLiteral("path"), dbPath);
    } else {
        opts.insert(QStringLiteral("host"), m_config.tripDbHost);
        opts.insert(QStringLiteral("port"), m_config.tripDbPort);
        opts.insert(QStringLiteral("dbname"), m_config.tripDbName);
        opts.insert(QStringLiteral("user"), m_config.tripDbUser);
        opts.insert(QStringLiteral("password"), m_config.tripDbPassword);
    }

    QString err;
    if (!m_tripStore->open(opts, &err)) {
        LOG_WARN("[MainWindow] 乘车数据库打开失败（将不记录站点统计）: " + err);
        delete m_tripStore;
        m_tripStore = nullptr;
        return;
    }

    // 重启后恢复最近一次行程（计数不清零）；无历史行程时才新建
    const QString lastTrip = m_tripStore->lastTrip();
    if (!lastTrip.isEmpty()) {
        m_currentTripId = lastTrip;
    } else {
        m_currentTripId = QStringLiteral("TRIP-%1")
                              .arg(QDateTime::currentDateTime().toString("yyyyMMdd-HHmmss"));
    }
    m_currentStationId = m_config.currentStationId;
    m_currentStationName = m_config.currentStationName;
    m_tripStore->beginTrip(m_currentTripId, m_config.vehicleId, QDateTime::currentDateTime());
    m_onboardCount = m_tripStore->onboardCount(m_currentTripId);

    // 恢复累计上车/下车（跨站点求和），并同步到 ZoneMonitor 供界面显示
    int totalBoard = 0, totalAlight = 0;
    for (const StopStat &s : m_tripStore->stopStats(m_currentTripId)) {
        totalBoard += s.boarding;
        totalAlight += s.alighting;
    }
    m_boardingCount = totalBoard;
    m_alightingCount = totalAlight;
    if (m_zoneMonitor) m_zoneMonitor->setCounters(totalBoard, totalAlight);

    LOG_INFO(QString("[MainWindow] 乘车数据存储已启动: backend=%1, trip=%2, station=%3")
                 .arg(m_tripStore->backendName(), m_currentTripId, m_currentStationName));
    updateOnboardLabel();
}

void MainWindow::setCurrentStation(const QString &stationId, const QString &stationName)
{
    m_currentStationId = stationId;
    m_currentStationName = stationName;
    LOG_INFO(QString("[MainWindow] 切换当前站点: %1 (%2)").arg(stationName, stationId));
}

void MainWindow::updateOnboardLabel()
{
    if (!ui || !ui->label_personCount) return;
    const int boardingCnt = m_zoneMonitor ? m_zoneMonitor->boardingCount() : m_boardingCount;
    const int alightingCnt = m_zoneMonitor ? m_zoneMonitor->alightingCount() : m_alightingCount;
    ui->label_personCount->setText(QString("区域人数: %1 | 车厢内: %2")
                                       .arg(m_currentPersonCount)
                                       .arg(m_onboardCount));
    if (ui->label_boardCount)
        ui->label_boardCount->setText(QString("上车: %1").arg(boardingCnt));
    if (ui->label_alightCount)
        ui->label_alightCount->setText(QString("下车: %1").arg(alightingCnt));

    pushWebStatus();
}

// ===== 按乘客分目录的图像归档 =====
void MainWindow::initPersonFolder()
{
    QString base = m_config.savePath;
    if (!QDir(base).isAbsolute())
        base = QCoreApplication::applicationDirPath() + "/" + base;
    m_personFolder = new PersonFolder(base);
    m_personFolder->setSaveEveryN(m_config.personSaveEveryN);
    LOG_INFO("[MainWindow] 按乘客分目录归档已就绪: " + base);
}

void MainWindow::saveSessionState()
{
    QJsonObject root;
    if (m_personFolder)
        root[QStringLiteral("personFolder")] = m_personFolder->stateToJson();

    QJsonArray events;
    for (const PersonEventRow &r : m_personEventRows) {
        QJsonObject e;
        e[QStringLiteral("personId")] = r.personId;
        e[QStringLiteral("status")] = r.status;
        e[QStringLiteral("matchedPersonId")] = r.matchedPersonId;
        e[QStringLiteral("similarity")] = static_cast<double>(r.similarity);
        events.append(e);
    }
    root[QStringLiteral("personEvents")] = events;

    QJsonArray boards;
    for (const BoardFolderEntry &e : m_boardFolders) {
        QJsonObject b;
        b[QStringLiteral("personId")] = e.personId;
        b[QStringLiteral("dir")] = e.dir;
        b[QStringLiteral("timeMs")] = static_cast<double>(e.timeMs);
        b[QStringLiteral("matched")] = e.matched;
        boards.append(b);
    }
    root[QStringLiteral("boardFolders")] = boards;

    if (m_sessionStatePath.isEmpty()) return;
    QDir().mkpath(QFileInfo(m_sessionStatePath).absolutePath());
    QFile f(m_sessionStatePath);
    if (f.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        f.write(QJsonDocument(root).toJson(QJsonDocument::Compact));
        f.close();
    }
}

void MainWindow::loadSessionState()
{
    if (m_sessionStatePath.isEmpty()) return;
    QFile f(m_sessionStatePath);
    if (!f.exists() || !f.open(QIODevice::ReadOnly)) return;
    const QJsonObject root = QJsonDocument::fromJson(f.readAll()).object();
    f.close();

    if (m_personFolder && root.contains(QStringLiteral("personFolder")))
        m_personFolder->stateFromJson(root.value(QStringLiteral("personFolder")).toObject());

    m_personEventRows.clear();
    for (const QJsonValue &v : root.value(QStringLiteral("personEvents")).toArray()) {
        const QJsonObject e = v.toObject();
        PersonEventRow r;
        r.personId = e.value(QStringLiteral("personId")).toInt();
        r.status = e.value(QStringLiteral("status")).toString();
        r.matchedPersonId = e.value(QStringLiteral("matchedPersonId")).toInt(-1);
        r.similarity = static_cast<float>(e.value(QStringLiteral("similarity")).toDouble());
        m_personEventRows.append(r);
    }

    m_boardFolders.clear();
    for (const QJsonValue &v : root.value(QStringLiteral("boardFolders")).toArray()) {
        const QJsonObject b = v.toObject();
        BoardFolderEntry e;
        e.personId = b.value(QStringLiteral("personId")).toInt();
        e.dir = b.value(QStringLiteral("dir")).toString();
        e.timeMs = static_cast<qint64>(b.value(QStringLiteral("timeMs")).toDouble());
        e.matched = b.value(QStringLiteral("matched")).toBool();
        m_boardFolders.append(e);
    }

    updateVisitorTable();
    LOG_INFO(QString("[MainWindow] 会话状态已恢复: 人员记录=%1, 上车目录=%2")
                 .arg(m_personEventRows.size()).arg(m_boardFolders.size()));
}

void MainWindow::initWebServer()
{
    m_webServer = new StatusWebServer(this);
    if (m_config.webServerEnabled) {
        if (!m_webServer->start(static_cast<quint16>(m_config.webServerPort))) {
            LOG_WARN("[MainWindow] Web 状态服务启动失败（端口可能被占用）");
        }
    } else {
        LOG_INFO("[MainWindow] Web 状态服务未启用");
    }
    pushWebStatus();
}

void MainWindow::pushWebStatus()
{
    if (!m_webServer) return;

    QJsonObject st;
    st["time"] = QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss");
    st["boarding"] = m_zoneMonitor ? m_zoneMonitor->boardingCount() : m_boardingCount;
    st["alighting"] = m_zoneMonitor ? m_zoneMonitor->alightingCount() : m_alightingCount;
    st["onboard"] = m_onboardCount;
    st["regionPersons"] = m_currentPersonCount;
    st["station"] = m_currentStationName;
    st["matches"] = m_recentMatches;
    m_webServer->setStatus(st);
}

void MainWindow::savePersonAppearanceFrames(const QVector<TrackedPerson> &persons,
                                            const cv::Mat &rawFrame)
{
    if (!m_personFolder || rawFrame.empty()) return;

    QSet<int> currentIds;
    for (const TrackedPerson &p : persons) {
        if (p.age < m_config.personMinTrackFrames) continue;          // 建档预热

        const int pid = m_personFolder->ensurePerson(p.trackId);
        if (pid <= 0) continue;
        currentIds.insert(pid);

        if (!p.detected) continue;   // 卡尔曼外推帧不存图，但保持视频打开，避免短暂遮挡就切段

        // 人体裁剪（带 30% padding），与原始整帧一并保存，均不叠加标注
        const cv::Rect safe = p.bbox & cv::Rect(0, 0, rawFrame.cols, rawFrame.rows);
        cv::Mat crop;
        if (!safe.empty()) {
            const int padX = static_cast<int>(safe.width * 0.3);
            const int padY = static_cast<int>(safe.height * 0.3);
            const int x1 = std::max(0, safe.x - padX);
            const int y1 = std::max(0, safe.y - padY);
            const int x2 = std::min(rawFrame.cols, safe.x + safe.width + padX);
            const int y2 = std::min(rawFrame.rows, safe.y + safe.height + padY);
            if (x2 > x1 && y2 > y1)
                crop = rawFrame(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
        }
        if (crop.empty()) crop = rawFrame.clone();

        m_personFolder->saveFrames(pid, rawFrame, crop, QDateTime::currentDateTime());

        // 同时保存对应深度图（伪彩，节流为每 3 帧 1 张）
        static int s_depthTick = 0;
        if ((s_depthTick++ % 3) == 0) {
            saveCorrespondingDepth(m_personFolder->personDir(pid),
                                   QString("depth_%1").arg(QDateTime::currentDateTime().toString("HHmmss_zzz")));
        }
    }

    // 乘客消失（本帧不再出现在跟踪列表）→ 关闭其视频，写入索引使其可被 VLC 播放
    for (const int pid : m_activePersonIds) {
        if (!currentIds.contains(pid)) {
            m_personFolder->closeVideo(pid);
        }
    }
    m_activePersonIds = currentIds;
}

void MainWindow::onFolderMatchFinished(quint64 jobId, int bestPersonId, QString bestDir,
                                       float similarity, bool matched)
{
    const FolderMatchPending pend = m_folderMatchPending.take(jobId);

    if (matched && bestPersonId > 0 && m_personFolder) {
        // 写 match.txt（方向 + 匹配对象 + 相似度）+ 标记去重
        m_personFolder->markMatched(bestPersonId, pend.alightPersonId,
                                    similarity, pend.alightTime);

        // 上车目录登记表置为已匹配（后续下车不再与此目录比对）
        for (BoardFolderEntry &e : m_boardFolders) {
            if (e.personId == bestPersonId) e.matched = true;
        }

        if (m_tripStore && m_tripStore->isOpen()) {
            const QString boardDir = bestDir.isEmpty()
                    ? m_personFolder->cropImagesDir(bestPersonId) : bestDir;
            m_tripStore->recordReidMatch(m_currentTripId,
                                         QStringLiteral("person_%1").arg(bestPersonId), boardDir,
                                         QStringLiteral("person_%1").arg(pend.alightPersonId),
                                         pend.alightDir,
                                         static_cast<double>(similarity), pend.alightTime);
        }

        // 回填人员记录的「乘客匹配ID」
        for (PersonEventRow &r : m_personEventRows) {
            if (r.personId == bestPersonId && r.status == QStringLiteral("上车")) {
                r.matchedPersonId = pend.alightPersonId;
                r.similarity = similarity;
            } else if (r.personId == pend.alightPersonId && r.status == QStringLiteral("下车")) {
                r.matchedPersonId = bestPersonId;
                r.similarity = similarity;
            }
        }
        updateVisitorTable();
        saveSessionState();

        // 推送到 Web 状态（手机端查看匹配记录）
        QJsonObject m;
        m["board"] = QStringLiteral("person_%1").arg(bestPersonId);
        m["alight"] = QStringLiteral("person_%1").arg(pend.alightPersonId);
        m["similarity"] = static_cast<double>(similarity);
        m_recentMatches.append(m);
        if (m_recentMatches.size() > 50) m_recentMatches.removeFirst();
        pushWebStatus();

        LOG_INFO(QString("[MainWindow] ★ 目录全图比对命中: person_%1(上车) ↔ person_%2(下车) sim=%3，已写 marker 并落库")
                     .arg(bestPersonId).arg(pend.alightPersonId).arg(similarity, 0, 'f', 4));
    } else {
        LOG_WARN(QString("[MainWindow] 目录全图比对未命中: 下车 person_%1 (best=person_%2 sim=%3)，保留待后续比对")
                     .arg(pend.alightPersonId).arg(bestPersonId).arg(similarity, 0, 'f', 3));
    }
}

// ===== 深度 ReID（OSNet 现场微调模型）=====
void MainWindow::initDeepReid()
{
    if (!m_config.reidUseDeepReid) {
        LOG_INFO("[MainWindow] 深度 ReID 未启用（使用手工多模态特征）");
        return;
    }

    if (!m_deepReid) m_deepReid = new DeepReID();
    // 识别推理卸载到 GPU：优先 CUDA 后端（不可用自动回退 CPU），降低 CPU 占用
    m_deepReid->setPreferableBackend(m_config.reidDeepUseCuda);

    QString modelPath = m_config.reidDeepModelPath;
    if (modelPath.isEmpty()) modelPath = QStringLiteral("./model/osnet_x1_0_head_finetuned.onnx");
    if (!QDir(modelPath).isAbsolute())
        modelPath = QCoreApplication::applicationDirPath() + "/" + modelPath;

    bool ok = m_deepReid->load(modelPath);
    if (!ok) {
        // 回退链：现场微调模型 → 原始 MSMT17 模型（源码目录，开发环境一并尝试）
        const QString appDir = QCoreApplication::applicationDirPath();
        const QStringList fallbacks = {
            appDir + QStringLiteral("/model/osnet_x1_0_head_finetuned.onnx"),
            appDir + QStringLiteral("/model/osnet_x1_0_msmt17.onnx"),
            appDir + QStringLiteral("/../CameraViewer/model/osnet_x1_0_head_finetuned.onnx"),
            appDir + QStringLiteral("/../CameraViewer/model/osnet_x1_0_msmt17.onnx"),
        };
        for (const QString &alt : fallbacks) {
            if (!QFileInfo::exists(alt)) continue;
            if (m_deepReid->load(alt)) { ok = true; modelPath = alt; break; }
        }
    }
    if (!ok) {
        LOG_WARN("[MainWindow] 深度 ReID 模型加载失败，回退手工多模态特征: " + modelPath);
        delete m_deepReid;
        m_deepReid = nullptr;
        return;
    }

    // 注入已存在的使用方（后台比对线程）；Tracker/ZoneMonitor 共用的 m_reidExtractor
    // 在采集启动时创建（见 onToggleCapture），届时再绑定同一 m_deepReid。
    if (m_matchWorker) m_matchWorker->setDeepReid(m_deepReid, true);
    if (m_folderMatchWorker) m_folderMatchWorker->setDeepReid(m_deepReid, true);
    if (m_passengerBook) m_passengerBook->setMatchThreshold(effectiveReidThreshold());

    LOG_INFO(QString("[MainWindow] 深度 ReID 已加载: %1 | 推理后端=%2 (生效阈值=%3)，手工 ReID 自动改走深度特征")
                 .arg(m_deepReid->modelPath())
                 .arg(m_deepReid->backendName())
                 .arg(effectiveReidThreshold(), 0, 'f', 2));
}

float MainWindow::effectiveReidThreshold() const
{
    if (m_deepReid && m_deepReid->ready() && m_config.reidUseDeepReid)
        return m_config.reidDeepMatchThreshold;
    return m_config.reidMatchThreshold;
}

float MainWindow::effectiveDetectorConf() const
{
    const bool head = (m_config.detectorModelKind.compare(QStringLiteral("head"),
                                                          Qt::CaseInsensitive) == 0);
    return head ? m_config.detectorHeadConfThreshold : m_config.detectorConfThreshold;
}

void MainWindow::applyDetectorFilters(PersonDetector *det) const
{
    if (!det) return;

    det->setMinSizeFilterEnabled(m_config.detectorSizeFilterEnabled);

    const bool head = (m_config.detectorModelKind.compare(QStringLiteral("head"),
                                                          Qt::CaseInsensitive) == 0);
    if (head) {
        // 头部检测：框小且近似方形，使用独立的宽松过滤（否则会被人体参数全部误杀）
        det->setMinSizeFilter(m_config.detectorHeadMinHeightRatio,
                              m_config.detectorHeadMinAreaRatio);
        det->setMinBoxSizePx(m_config.detectorHeadMinBoxWidthPx,
                             m_config.detectorHeadMinBoxHeightPx);
        det->setBoxAspectRange(m_config.detectorHeadMinBoxAspect,
                               m_config.detectorHeadMaxBoxAspect);
        LOG_INFO(QString("[MainWindow] 检测类型=head（头部检测），过滤 hRatio>=%1 aRatio>=%2 "
                         "minPx=%3x%4 aspect∈[%5,%6]")
                     .arg(m_config.detectorHeadMinHeightRatio, 0, 'f', 4)
                     .arg(m_config.detectorHeadMinAreaRatio, 0, 'f', 5)
                     .arg(m_config.detectorHeadMinBoxWidthPx).arg(m_config.detectorHeadMinBoxHeightPx)
                     .arg(m_config.detectorHeadMinBoxAspect, 0, 'f', 2)
                     .arg(m_config.detectorHeadMaxBoxAspect, 0, 'f', 2));
    } else {
        det->setMinSizeFilter(m_config.detectorMinPersonHeightRatio,
                              m_config.detectorMinPersonAreaRatio);
        det->setMinBoxSizePx(m_config.detectorMinBoxWidthPx, m_config.detectorMinBoxHeightPx);
        det->setBoxAspectRange(m_config.detectorMinBoxAspect, m_config.detectorMaxBoxAspect);
    }

    // 最大尺寸上限：head 用较严参数抑制「超大静态误检」；person 默认不限制
    if (head) {
        det->setMaxSizeFilter(m_config.detectorHeadMaxBoxWidthRatio,
                              m_config.detectorHeadMaxBoxHeightRatio,
                              m_config.detectorHeadMaxBoxAreaRatio);
    } else {
        det->setMaxSizeFilter(m_config.detectorMaxBoxWidthRatio,
                              m_config.detectorMaxBoxHeightRatio,
                              m_config.detectorMaxBoxAreaRatio);
    }

    // 静态误检抑制：同位置、近零位移的持续检测判为背景误检并忽略
    det->setStaticSuppression(m_config.detectorStaticSuppressEnabled,
                              m_config.detectorStaticConfirmFrames,
                              m_config.detectorStaticMaxMoveRatio,
                              m_config.detectorStaticIouThreshold,
                              m_config.detectorStaticMissTolerance);
}

QString MainWindow::savePassengerEventImage(const cv::Mat &frame, const cv::Rect &bbox,
                                            int passengerId, bool boarding, const QDateTime &time)
{
    if (frame.empty()) return QString();

    // 目录：save_images/reid/passenger/
    QString basePath = m_config.reidSavePath;
    if (!QDir(basePath).isAbsolute()) {
        basePath = QCoreApplication::applicationDirPath() + "/" + basePath;
    }
    const QString dir = QDir(basePath).filePath("passenger");
    if (!QDir().mkpath(dir)) {
        LOG_WARN("[MainWindow] cannot create passenger image dir: " + dir);
        return QString();
    }

    // 裁剪人物（带 30% padding），无有效框时保留整帧
    cv::Mat img = frame.clone();
    const cv::Rect safe = bbox & cv::Rect(0, 0, frame.cols, frame.rows);
    if (!safe.empty()) {
        const int padX = static_cast<int>(safe.width * 0.3);
        const int padY = static_cast<int>(safe.height * 0.3);
        const int x1 = std::max(0, safe.x - padX);
        const int y1 = std::max(0, safe.y - padY);
        const int x2 = std::min(frame.cols, safe.x + safe.width + padX);
        const int y2 = std::min(frame.rows, safe.y + safe.height + padY);
        if (x2 > x1 && y2 > y1)
            img = frame(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
    }

    // 文件名：乘客ID + 上车/下车 + 时间，例如 P3_上车_20260929_153012.jpg
    const QDateTime t = time.isValid() ? time : QDateTime::currentDateTime();
    const QString dirText = boarding ? QStringLiteral("上车") : QStringLiteral("下车");
    const QString fileName = QString("P%1_%2_%3.jpg")
                                 .arg(passengerId).arg(dirText).arg(t.toString("yyyyMMdd_HHmmss"));
    const QString fullPath = QDir(dir).filePath(fileName);

    // 在图上标注乘客信息
    const std::string label = QString("P%1 %2 %3")
                                  .arg(passengerId).arg(dirText)
                                  .arg(t.toString("yyyy-MM-dd HH:mm:ss"))
                                  .toStdString();
    cv::putText(img, label, cv::Point(10, 28), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                boarding ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);

    if (cv::imwrite(fullPath.toStdString(), img)) {
        LOG_INFO(QString("[MainWindow] 乘客%1图片已保存: %2").arg(dirText, fullPath));
        return fullPath;
    }
    LOG_WARN("[MainWindow] 乘客图片保存失败: " + fullPath);
    return QString();
}

QString MainWindow::savePassengerEventImages(const QVector<cv::Mat> &crops,
                                             const QVector<cv::Rect> &boxes,
                                             int passengerId, bool boarding, const QDateTime &time)
{
    Q_UNUSED(boxes);
    if (crops.isEmpty()) return QString();

    QString basePath = m_config.reidSavePath;
    if (!QDir(basePath).isAbsolute()) {
        basePath = QCoreApplication::applicationDirPath() + "/" + basePath;
    }

    const QDateTime t = time.isValid() ? time : QDateTime::currentDateTime();
    const QString dirText = boarding ? QStringLiteral("上车") : QStringLiteral("下车");
    const QString folderName = QString("P%1_%2_%3")
                                   .arg(passengerId).arg(dirText).arg(t.toString("yyyyMMdd_HHmmss"));
    const QString dir = QDir(basePath).filePath("passenger/" + folderName);
    if (!QDir().mkpath(dir)) {
        LOG_WARN("[MainWindow] cannot create passenger sequence dir: " + dir);
        return QString();
    }

    // 限制单次保存数量，避免目录过大
    const int maxSave = qMin(crops.size(), 30);
    int saved = 0;
    for (int i = 0; i < maxSave; ++i) {
        const cv::Mat &c = crops[i];
        if (c.empty()) continue;

        cv::Mat img = c.clone();
        const std::string label = QString("P%1 %2  #%3/%4")
                                      .arg(passengerId).arg(dirText)
                                      .arg(i + 1).arg(crops.size())
                                      .toStdString();
        cv::putText(img, label, cv::Point(6, 22), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    boarding ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);

        const QString fp = QDir(dir).filePath(QString("frame_%1.jpg").arg(i, 3, 10, QChar('0')));
        if (cv::imwrite(fp.toStdString(), img)) saved++;
    }

    if (saved == 0) return QString();

    LOG_INFO(QString("[MainWindow] 乘客%1整段序列图片已保存: %2 (%3/%4 张)")
                 .arg(dirText, dir).arg(saved).arg(crops.size()));
    return dir;
}

QString MainWindow::savePassengerEventVideo(const QVector<cv::Mat> &frames,
                                            int passengerId, bool boarding,
                                            const QDateTime &time, qint64 durationMs)
{
    if (frames.isEmpty()) return QString();

    QString basePath = m_config.reidSavePath;
    if (!QDir(basePath).isAbsolute()) {
        basePath = QCoreApplication::applicationDirPath() + "/" + basePath;
    }
    const QString dir = QDir(basePath).filePath("passenger");
    if (!QDir().mkpath(dir)) {
        LOG_WARN("[MainWindow] cannot create passenger video dir: " + dir);
        return QString();
    }

    const QDateTime t = time.isValid() ? time : QDateTime::currentDateTime();
    const QString dirText = boarding ? QStringLiteral("上车") : QStringLiteral("下车");
    const QString fileBase = (passengerId > 0)
            ? QString("P%1_%2_%3").arg(passengerId).arg(dirText).arg(t.toString("yyyyMMdd_HHmmss"))
            : QString("异常乘客_%1_%2").arg(dirText).arg(t.toString("yyyyMMdd_HHmmss"));

    // 帧率：采样间隔为 2 帧，故按“帧数 / 实际耗时”推算（限制 5~30fps）
    double fps = 12.0;
    if (durationMs > 0 && frames.size() > 1) {
        fps = frames.size() * 1000.0 / static_cast<double>(durationMs);
    }
    fps = qBound(5.0, fps, 30.0);

    const cv::Size sz = frames.first().size();
    if (sz.width <= 0 || sz.height <= 0) return QString();

    // 优先 mp4(mp4v)，打不开则回退 avi(MJPG)，兼容不同 OpenCV 后端
    QString fullPath = QDir(dir).filePath(fileBase + ".mp4");
    cv::VideoWriter writer(fullPath.toStdString(),
                           cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, sz, true);
    if (!writer.isOpened()) {
        fullPath = QDir(dir).filePath(fileBase + ".avi");
        writer.open(fullPath.toStdString(),
                    cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, sz, true);
    }
    if (!writer.isOpened()) {
        LOG_WARN("[MainWindow] 视频写入器打开失败(编码不可用): " + fullPath);
        return QString();
    }

    int written = 0;
    for (const cv::Mat &f : frames) {
        if (f.empty()) continue;
        cv::Mat out = f;
        if (f.size() != sz) cv::resize(f, out, sz);
        writer.write(out);
        written++;
    }
    writer.release();

    if (written == 0) return QString();

    LOG_INFO(QString("[MainWindow] 乘客%1整段视频已保存: %2 (帧数=%3, fps=%4, 耗时=%5ms)")
                 .arg(dirText, fullPath).arg(written).arg(fps, 0, 'f', 1).arg(durationMs));
    return fullPath;
}

QString MainWindow::saveAbnormalAlightImages(const QVector<cv::Mat> &crops,
                                             const QVector<cv::Rect> &boxes,
                                             const QDateTime &time)
{
    Q_UNUSED(boxes);
    if (crops.isEmpty()) return QString();

    QString basePath = m_config.reidSavePath;
    if (!QDir(basePath).isAbsolute()) {
        basePath = QCoreApplication::applicationDirPath() + "/" + basePath;
    }

    const QDateTime t = time.isValid() ? time : QDateTime::currentDateTime();
    const QString folderName = QString("异常乘客_下车_%1").arg(t.toString("yyyyMMdd_HHmmss"));
    const QString dir = QDir(basePath).filePath("passenger/" + folderName);
    if (!QDir().mkpath(dir)) {
        LOG_WARN("[MainWindow] cannot create abnormal-alight dir: " + dir);
        return QString();
    }

    const int maxSave = qMin(crops.size(), 30);
    int saved = 0;
    for (int i = 0; i < maxSave; ++i) {
        const cv::Mat &c = crops[i];
        if (c.empty()) continue;

        cv::Mat img = c.clone();
        const std::string label = QString("异常乘客下车  #%1/%2  %3")
                                      .arg(i + 1).arg(crops.size())
                                      .arg(t.toString("HH:mm:ss"))
                                      .toStdString();
        // 黄色标注，区别于正常上车(绿)/下车(红)
        cv::putText(img, label, cv::Point(6, 22), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 255, 255), 2);

        const QString fp = QDir(dir).filePath(QString("frame_%1.jpg").arg(i, 3, 10, QChar('0')));
        if (cv::imwrite(fp.toStdString(), img)) saved++;
    }

    if (saved == 0) return QString();

    LOG_WARN(QString("[MainWindow] 异常乘客下车整段图片已保存: %1 (%2/%3 张)")
                 .arg(dir).arg(saved).arg(crops.size()));
    return dir;
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
        QString dirStr = (m_currentEntryDirection == EntryDirection::TopToBottom)
                             ? "↓从上往下（下车）" : "↑从下往上（下车）";
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

void MainWindow::onClearCounts()
{
    // 只清「人员记录 + 上下车人数」，保留已绘制的上下车分界线
    m_visitors.clear();
    m_trackToVisitor.clear();
    m_nextVisitorId = 1;
    m_visitorPendingDir.clear();
    m_visitorFrameTick.clear();
    m_visitorSegmentSeq.clear();

    // 重置通道监控器（计数与轨迹状态归零，门线定义保留）
    if (m_zoneMonitor) {
        m_zoneMonitor->reset();
    }
    // 清零时结束当前行程并新建，保证重启后不会恢复清零前的计数
    if (m_tripStore && m_tripStore->isOpen()) {
        m_tripStore->endTrip(m_currentTripId, QDateTime::currentDateTime());
        m_currentTripId = QStringLiteral("TRIP-%1")
                              .arg(QDateTime::currentDateTime().toString("yyyyMMdd-HHmmss"));
        m_tripStore->beginTrip(m_currentTripId, m_config.vehicleId, QDateTime::currentDateTime());
    }
    m_boardingCount = 0;
    m_alightingCount = 0;
    m_onboardCount = 0;

    // 清空乘客台账（上车记录）
    if (m_passengerBook) m_passengerBook->clear();

    // 按乘客分目录归档：会话序号归 1、上车目录登记表清空、关闭视频写入
    if (m_personFolder) m_personFolder->reset();
    m_boardFolders.clear();
    m_folderMatchPending.clear();
    m_personEventRows.clear();

    if (ui->tableVisitors) {
        ui->tableVisitors->setRowCount(0);
    }

    updateOnboardLabel();
    saveSessionState();   // 清零后立即落盘，重启不再恢复旧记录
    LOG_INFO("[MainWindow] 计数与人员记录已清除（保留上下车分界线）");
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

    // 只保留一个检测框：最新绘制的覆盖之前的
    if (!m_regionManager->regions().isEmpty()) {
        m_regionManager->clearRegions();
        if (m_imageView) m_imageView->clearRegionPolygons();
    }

    // 创建区域
    DetectionRegion region;
    region.polygon = normalizedPoints;
    region.name = QStringLiteral("区域1");
    region.entryDirection = m_currentEntryDirection;
    region.color = QColor(0, 255, 0);

    m_regionManager->addRegion(region);

    // 更新 ImageView 显示（区域多边形 + 虚拟门线）
    refreshRegionOverlays();

    // 自动保存区域
    QString regionPath = m_config.regionSavePath;
    if (!QDir(regionPath).isAbsolute()) {
        regionPath = QCoreApplication::applicationDirPath() + "/" + regionPath;
    }
    m_regionManager->saveRegions(regionPath);

    // 更新采集状态和引导提示
    updateCaptureStatus();

    QString dirStr = (region.entryDirection == EntryDirection::TopToBottom)
                         ? "从上往下↓（下车）" : "从下往上↑（下车）";
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

    // 停止旧线程（防止 openCamera 释放 VideoCapture 时仍在读取）
    if (m_depthThread) {
        m_depthThread->stop();
        m_depthThread->wait();
        delete m_depthThread;
        m_depthThread = nullptr;
    }
    if (m_detectThread) {
        m_detectThread->stop();
        m_detectThread->wait();
        delete m_detectThread;
        m_detectThread = nullptr;
    }
    if (m_grabThread) {
        m_grabThread->stop();
        m_grabThread->wait();
        delete m_grabThread;
        m_grabThread = nullptr;
    }
    delete m_frameSlot;
    m_frameSlot = nullptr;

    if (m_camera.openCamera(m_config.rtspUrl, m_config.depthRtspUrl)) {
        m_reconnectTimer->stop();
        if (m_depthView) m_depthView->setVisible(m_camera.hasDepth());

        // 重新创建线程：RGB 拉流 / 深度拉流 / 检测
        m_frameSlot = new LatestFrameSlot();
        m_grabThread = new GrabThread(&m_camera, m_frameSlot, "HX-CCD23", this);
        m_depthThread = new DepthGrabThread(&m_camera, "HX-CCD23-Depth", this);
        m_detectThread = new DetectThread(m_frameSlot, "HX-CCD23-Detect", this);

        if (m_detector) m_detectThread->setDetector(m_detector);
        if (m_tracker) m_detectThread->setTracker(m_tracker);
        if (m_reidExtractor) m_detectThread->setReidExtractor(m_reidExtractor);
        m_detectThread->setBackgroundModel(m_config.reidUseBackground ? m_bgModel : nullptr);

        connect(m_grabThread, &GrabThread::deviceLost, this, &MainWindow::onGrabThreadDeviceLost);
        connect(m_grabThread, &GrabThread::frameGrabbed, this, &MainWindow::onFrameGrabbed);
        connect(m_detectThread, &DetectThread::frameProcessed, this, &MainWindow::onFrameProcessed);
        connect(m_detectThread, &DetectThread::rawFrameGrabbed, this, [this](const cv::Mat &frame) {
            QMutexLocker locker(&m_rawFrameMutex);
            m_currentRawFrame = frame.clone();
        });
        connect(m_depthThread, &DepthGrabThread::depthFrameGrabbed, this, &MainWindow::onDepthFrameGrabbed);

        // 重置 FPS 计算
        m_fpsTimer.restart();
        m_fpsFrameCount = 0;
        m_currentFps = 0.0;

        m_depthFpsTimer.restart();
        m_depthFpsFrameCount = 0;
        m_currentDepthFps = 0.0;

        m_grabThread->start();
        m_depthThread->start();
        m_detectThread->start();
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

QString MainWindow::pickDetectorModelPath() const
{
    const QString appDir = QCoreApplication::applicationDirPath();

    if (m_config.detectorPreferOnnx) {
        const QString cfg = m_config.detectorOnnxPath;
        QStringList candidates;
        if (QFileInfo(cfg).isAbsolute()) {
            candidates << cfg;
        } else {
            candidates << appDir + "/" + cfg
                       << appDir + "/../CameraViewer/" + cfg   // 开发/构建目录回退
                       << QDir::currentPath() + "/" + cfg;
        }
        for (const QString &c : candidates) {
            if (QFileInfo::exists(c)) return c;
        }
        LOG_WARN(QString("[MainWindow] 未找到 ONNX 小模型(%1)，回退到 %2")
                     .arg(cfg, m_config.modelPath));
    }
    return m_config.modelPath;
}

void MainWindow::downloadDetectorModel(const QString &url, const QString &targetPath)
{
    if (url.isEmpty() || !m_netMgr) return;

    QString target = targetPath;
    if (!QFileInfo(target).isAbsolute()) {
        target = QCoreApplication::applicationDirPath() + "/" + target;
    }
    if (QFileInfo::exists(target)) return;   // 已存在，无需下载

    QDir().mkpath(QFileInfo(target).absolutePath());

    QNetworkRequest req{QUrl(url)};
    req.setAttribute(QNetworkRequest::RedirectPolicyAttribute,
                     QNetworkRequest::NoLessSafeRedirectPolicy);
    QNetworkReply *reply = m_netMgr->get(req);
    LOG_INFO("[MainWindow] 开始下载检测模型: " + url);

    connect(reply, &QNetworkReply::finished, this, [this, reply, target]() {
        reply->deleteLater();
        if (reply->error() != QNetworkReply::NoError) {
            LOG_WARN(QString("[MainWindow] 检测模型下载失败: %1").arg(reply->errorString()));
            return;
        }
        const QByteArray data = reply->readAll();
        if (data.size() < 1024 * 1024) {
            LOG_WARN(QString("[MainWindow] 检测模型文件过小(%1 字节)，已忽略").arg(data.size()));
            return;
        }
        QFile f(target);
        if (!f.open(QIODevice::WriteOnly)) {
            LOG_WARN("[MainWindow] 无法写入模型文件: " + target);
            return;
        }
        f.write(data);
        f.close();

        // 校验：必须能被 OpenCV DNN 读取，否则删除，避免以后误用损坏文件
        try {
            cv::dnn::Net net = cv::dnn::readNetFromONNX(target.toStdString());
            if (net.empty()) throw cv::Exception();
            LOG_INFO(QString("[MainWindow] 检测模型下载并校验成功: %1 (%2 MB)，"
                             "下次启动采集/离线判定将优先使用")
                         .arg(target).arg(data.size() / 1024.0 / 1024.0, 0, 'f', 1));
        } catch (const cv::Exception &e) {
            QFile::remove(target);
            LOG_WARN(QString("[MainWindow] 检测模型校验失败，已删除: %1")
                         .arg(QString::fromStdString(e.what())));
        }
    });
}

void MainWindow::onOfflineDetect()
{
    // ⚠ 不可复用采集线程正在使用的检测器：TensorRT 推理不是线程安全的，
    //    采集线程与离线判定并发调用同一 detector 会导致进程崩溃（已复现）。
    //    因此这里始终新建独立检测器；若采集正在运行，会有额外显存/性能开销。
    PersonDetector *offlineDetector = nullptr;
    bool ownsDetector = true;

    if (m_isCapturing) {
        LOG_WARN("[OfflineDetect] 采集运行中执行离线判定：新建独立检测器，显存/性能占用翻倍");
    }

    {
        // 尝试加载模型（独立实例，避免与采集线程并发使用 TensorRT）
        offlineDetector = new PersonDetector();
        // 同样优先轻量 ONNX 小模型
        QString modelPath = pickDetectorModelPath();
        if (!QFileInfo(modelPath).isAbsolute()) {
            modelPath = QCoreApplication::applicationDirPath() + "/" + modelPath;
        }
        offlineDetector->setDnnUseCuda(m_config.detectorDnnUseCuda);
        offlineDetector->setDnnInputSize(m_config.detectorDnnInputSize);
        offlineDetector->setLetterbox(m_config.detectorLetterbox);   // 与在线一致

        LOG_INFO("[OfflineDetect] Trying to load model: " + modelPath);

        bool loaded = offlineDetector->loadModel(modelPath, effectiveDetectorConf(), m_config.detectorNmsThreshold);

        // 如果 .engine 加载失败，尝试回退到 .onnx
        if (!loaded && modelPath.endsWith(".engine", Qt::CaseInsensitive)) {
            QString onnxPath = modelPath.left(modelPath.lastIndexOf('.')) + ".onnx";
            LOG_INFO("[OfflineDetect] Engine failed, trying ONNX fallback: " + onnxPath);
            if (QFileInfo::exists(onnxPath)) {
                loaded = offlineDetector->loadModel(onnxPath, effectiveDetectorConf(), m_config.detectorNmsThreshold);
            } else {
                LOG_WARN("[OfflineDetect] ONNX fallback not found: " + onnxPath);
            }
        }

        // 如果 .onnx 加载失败，尝试回退到 .engine
        if (!loaded && modelPath.endsWith(".onnx", Qt::CaseInsensitive)) {
            QString enginePath = modelPath.left(modelPath.lastIndexOf('.')) + ".engine";
            LOG_INFO("[OfflineDetect] ONNX failed, trying engine fallback: " + enginePath);
            if (QFileInfo::exists(enginePath)) {
                loaded = offlineDetector->loadModel(enginePath, effectiveDetectorConf(), m_config.detectorNmsThreshold);
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

        // 尺寸/长宽比过滤（离线判定同样适用 person/head 自适应参数）
        applyDetectorFilters(offlineDetector);

        QString backend = offlineDetector->isTrtBackend() ? "TensorRT" : "OpenCV DNN";
        LOG_INFO(QString("[OfflineDetect] Model loaded [%1]: %2").arg(backend, modelPath));
    }

    {
        // 先让表单析构，再释放检测器，避免表单成员悬空引用
        OfflineDetectForm form(offlineDetector, m_regionManager,
                               m_config.reidUseDeepReid,
                               m_config.reidDeepModelPath,
                               m_config.reidDeepMatchThreshold,
                               this);
        form.exec();
    }

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
    }

    // 应用尺寸/长宽比过滤参数（运行时生效；依据 person/head 自动选择）
    applyDetectorFilters(m_detector);

    // 深度 ReID：若由关→开且尚未加载，则立即加载
    if (m_config.reidUseDeepReid && !m_deepReid) initDeepReid();

    // 应用 ReID 多特征融合权重（颜色/纹理/身形）
    if (m_reidExtractor) {
        m_reidExtractor->setComponentWeights(m_config.reidWeightColor,
                                             m_config.reidWeightTexture,
                                             m_config.reidWeightShape);
        // 深度后端开关与头框扩展
        m_reidExtractor->setDeepReid(m_deepReid, m_config.reidUseDeepReid);
        m_reidExtractor->setBoxExpandForReid(m_config.reidHeadBoxExpand);
        // 应用背景扣除开关
        if (m_config.reidUseBackground) {
            if (!m_bgModel) m_bgModel = new BackgroundModel();
            m_reidExtractor->setBackgroundModel(m_bgModel);
        } else {
            m_reidExtractor->setBackgroundModel(nullptr);
        }
    }
    // 下车比对线程使用独立提取器 → 同步同一套融合权重（线程安全队列投递）
    if (m_matchWorker) {
        m_matchWorker->setComponentWeights(m_config.reidWeightColor,
                                           m_config.reidWeightTexture,
                                           m_config.reidWeightShape);
        m_matchWorker->setDeepReid(m_deepReid, m_config.reidUseDeepReid);
    }
    if (m_detectThread) {
        m_detectThread->setBackgroundModel(m_config.reidUseBackground ? m_bgModel : nullptr);

        // 人物片段录制参数同步（运行中修改配置即刻生效）
        QString clipDir = m_config.personClipPath;
        if (!QDir(clipDir).isAbsolute())
            clipDir = QCoreApplication::applicationDirPath() + "/" + clipDir;
        m_detectThread->setPersonClipRecording(m_config.personClipEnabled, clipDir,
                                               m_config.personClipGapMs, m_config.personClipFps);
    }

    // 应用 ReID 匹配阈值（运行时生效；深度后端用 DeepMatchThreshold）
    if (m_zoneMonitor) {
        m_zoneMonitor->setReidMatchThreshold(effectiveReidThreshold());
    }
    // 乘客台账比对阈值同步
    if (m_passengerBook) {
        m_passengerBook->setMatchThreshold(effectiveReidThreshold());
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

// ===== 版本 / 更新 =====

QString MainWindow::appVersionText() const
{
    // 简版：vX.Y.Z（完整信息含构建日期，见 onAbout / 日志）
    return QString(APP_VERSION_DISPLAY);
}

int MainWindow::parseVersionCode(const QString &ver)
{
    QString v = ver.trimmed();
    if (v.startsWith('v') || v.startsWith('V')) v = v.mid(1);
    if (v.isEmpty()) return -1;

    const QStringList parts = v.split('.');
    bool ok = false;
    const int major = parts.value(0).toInt(&ok);
    if (!ok) return -1;
    const int minor = parts.size() > 1 ? parts.value(1).toInt() : 0;
    const int patch = parts.size() > 2 ? parts.value(2).toInt() : 0;
    return major * 10000 + minor * 100 + patch;
}

void MainWindow::onAbout()
{
    QMessageBox::about(
        this,
        QString("关于 %1").arg(APP_PRODUCT_NAME),
        QString(
            "<b>%1</b><br><br>"
            "版本号：<b>%2</b>（code=%3）<br>"
            "构建日期：%4 %5<br>"
            "更新地址：%6")
            .arg(APP_PRODUCT_NAME,
                 APP_VERSION_DISPLAY)
            .arg(APP_VERSION_CODE)
            .arg(APP_BUILD_DATE, APP_BUILD_TIME)
            .arg(m_config.updateUrl.trimmed().isEmpty()
                     ? QStringLiteral("（未配置，支持程序目录下 update.json）")
                     : m_config.updateUrl.trimmed()));
}

void MainWindow::onCheckUpdate()
{
    checkForUpdates();
}

void MainWindow::checkForUpdates()
{
    const int curCode = APP_VERSION_CODE;
    const QString curVer = appVersionText();

    // 公共处理：解析更新清单并比较版本
    auto handleManifest = [this, curCode, curVer](const QByteArray &data, const QString &from) {
        QJsonParseError err;
        const QJsonDocument doc = QJsonDocument::fromJson(data, &err);
        if (err.error != QJsonParseError::NoError || !doc.isObject()) {
            QMessageBox::warning(this, "检查更新",
                QString("更新清单解析失败\n来源：%1\n错误：%2").arg(from, err.errorString()));
            return;
        }
        const QJsonObject o = doc.object();
        const QString remoteVer = o.value("version").toString();
        const QString dlUrl = o.value("url").toString();
        const QString notes = o.value("notes").toString();

        const int remoteCode = parseVersionCode(remoteVer);
        if (remoteCode < 0) {
            QMessageBox::warning(this, "检查更新",
                QString("更新清单中的版本号无效：\"%1\"").arg(remoteVer));
            return;
        }

        if (remoteCode > curCode) {
            QString msg = QString(
                "<b>发现新版本！</b><br><br>"
                "当前版本：%1<br>最新版本：<b>v%2</b><br>")
                .arg(curVer, remoteVer);
            if (!notes.isEmpty())
                msg += QString("<br>更新说明：%1").arg(notes.toHtmlEscaped());
            if (!dlUrl.isEmpty())
                msg += QString("<br><br>下载地址：<a href='%1'>%1</a>").arg(dlUrl);

            QMessageBox box(this);
            box.setWindowTitle("检查更新");
            box.setIcon(QMessageBox::Information);
            box.setTextFormat(Qt::RichText);
            box.setText(msg);
            QPushButton *openBtn = dlUrl.isEmpty()
                ? nullptr : box.addButton("打开下载地址", QMessageBox::AcceptRole);
            box.addButton("关闭", QMessageBox::RejectRole);
            box.exec();
            if (openBtn && box.clickedButton() == openBtn) {
                QDesktopServices::openUrl(QUrl(dlUrl));
            }
            LOG_INFO(QString("[Update] 发现新版本: %1 -> %2").arg(curVer, remoteVer));
        } else {
            QMessageBox::information(this, "检查更新",
                QString("当前已是最新版本。\n\n当前版本：%1\n服务器版本：v%2")
                    .arg(curVer, remoteVer));
            LOG_INFO(QString("[Update] 已是最新版本: %1 (远端 %2)").arg(curVer, remoteVer));
        }
    };

    const QString url = m_config.updateUrl.trimmed();

    // 1) 远端更新地址（http/https）
    if (!url.isEmpty()) {
        if (!m_netMgr) m_netMgr = new QNetworkAccessManager(this);
        QNetworkRequest req{QUrl(url)};
        req.setAttribute(QNetworkRequest::RedirectPolicyAttribute,
                         QNetworkRequest::NoLessSafeRedirectPolicy);
        QNetworkReply *reply = m_netMgr->get(req);
        LOG_INFO("[Update] 正在检查远端更新清单: " + url);
        connect(reply, &QNetworkReply::finished, this,
                [this, reply, url, handleManifest]() {
            reply->deleteLater();
            if (reply->error() != QNetworkReply::NoError) {
                QMessageBox::warning(this, "检查更新",
                    QString("无法访问更新地址：\n%1\n\n错误：%2\n\n"
                            "→ 结论：当前环境下无法在线更新。")
                        .arg(url, reply->errorString()));
                LOG_WARN(QString("[Update] 拉取失败: %1").arg(reply->errorString()));
                return;
            }
            handleManifest(reply->readAll(), url);
        });
        return;
    }

    // 2) 本地 update.json（离线/U盘升级）
    const QString localPath = QCoreApplication::applicationDirPath() + "/update.json";
    if (!QFileInfo::exists(localPath)) {
        QMessageBox::information(this, "检查更新",
            QString("当前程序版本：%1\n\n"
                    "未配置更新地址（config.ini 中 [Update]/UpdateUrl），"
                    "程序目录下也没有 update.json。\n\n"
                    "→ 结论：当前环境不进行在线更新，可手动替换程序完成升级。")
                .arg(curVer));
        LOG_WARN("[Update] 未配置更新来源（无 updateUrl，也无 update.json）");
        return;
    }

    QFile f(localPath);
    if (!f.open(QIODevice::ReadOnly)) {
        QMessageBox::warning(this, "检查更新",
            QString("无法读取本地更新清单：%1").arg(localPath));
        return;
    }
    const QByteArray data = f.readAll();
    f.close();
    handleManifest(data, localPath);
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
        // 人员进出记录：(乘客ID, 上车时间, 下车时间)
        entry.boardTime = vr.enterTime.isValid() ? vr.enterTime.toString(Qt::ISODate) : QString();
        entry.alightTime = vr.exitTime.isValid() ? vr.exitTime.toString(Qt::ISODate) : QString();
        entry.isInside = vr.isInside;
        entry.regionId = vr.regionId;
        entry.currentTrackId = vr.currentTrackId;
        entry.reidFeatureBase64 = VisitorLog::matToBase64(vr.reidFeature);
        // 进入/离开图像路径由 saveReidImage 管理，这里记录目录
        entry.enterImagePath = QString("person_%1").arg(vr.visitorId);
        entry.exitImagePath = QString("person_%1").arg(vr.visitorId);
        entry.direction = vr.direction;
        entries[vr.visitorId] = entry;
    }

    m_visitorLog->saveVisitors(entries);
    LOG_INFO(QString("[MainWindow] Visitor records saved: %1 entries").arg(entries.size()));

    // 乘客台账（上车记录）一并持久化，重启后仍可做下车比对
    if (m_passengerBook) m_passengerBook->save();
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

        // 恢复时间：上车时间 / 下车时间
        if (!entry.boardTime.isEmpty()) {
            vr.enterTime = QDateTime::fromString(entry.boardTime, Qt::ISODate);
        }
        if (!entry.alightTime.isEmpty()) {
            vr.exitTime = QDateTime::fromString(entry.alightTime, Qt::ISODate);
        }

        // 恢复 ReID 特征
        if (!entry.reidFeatureBase64.isEmpty()) {
            vr.reidFeature = VisitorLog::base64ToMat(entry.reidFeatureBase64);
        }

        // 恢复运动方向
        vr.direction = entry.direction;

        m_visitors[entry.visitorId] = vr;

        // 更新 m_nextVisitorId
        if (entry.visitorId >= m_nextVisitorId) {
            m_nextVisitorId = entry.visitorId + 1;
        }

        restoredCount++;
    }

    // 更新表格显示
    updateVisitorTable();

    LOG_INFO(QString("[MainWindow] Loaded %1 visitor records from previous session (nextVisitorId=%2)")
                 .arg(restoredCount).arg(m_nextVisitorId));
}
