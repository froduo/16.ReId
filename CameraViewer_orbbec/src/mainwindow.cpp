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
#include "overlaytext.h"

#include <cmath>

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

    // ===== 深度伪彩视图（Orbbec Gemini 335）=====
    // 与彩色视图左右放入同一水平分屏器：尺寸自适应、可拖动分隔条；
    // 非 Orbbec 模式或相机无深度输出时隐藏该视图。
    if (m_imageView) {
        if (QHBoxLayout* imageLayout = ui->imageLayout) {
            m_depthView = new ImageView(this);
            m_depthView->setObjectName("depthViewMain");
            m_depthView->setDrawMode(ImageView::DrawMode::View); // 只读：不接受区域绘制
            m_depthView->setToolTip("深度伪彩视图（近处偏暖色；按有效深度 5%~95% 分位自动拉伸；黑=无有效深度）");
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
                m_mediaSplitter->setStretchFactor(0, 3);   // 彩色：深度 ≈ 3:2
                m_mediaSplitter->setStretchFactor(1, 2);
                imageLayout->insertWidget(idx, m_mediaSplitter);
                imageLayout->setStretch(idx, 14);          // 图像区 14 : 右侧面板 11
                if (idx + 1 < imageLayout->count())
                    imageLayout->setStretch(idx + 1, 11);
            } else {
                imageLayout->addWidget(m_mediaSplitter);   // 兜底
            }

            // 初始可见性：仅 Orbbec 模式默认显示；相机实际有无深度在启动采集后确认
            m_depthView->setVisible(m_config.useOrbbec);
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
    connect(ui->btnSave, &QPushButton::clicked, this, &MainWindow::onSaveCurrentFrame);
    connect(ui->chkAutoSave, &QCheckBox::toggled, this, &MainWindow::onAutoSaveToggled);

    // 区域绘制相关信号槽
    connect(ui->btnDrawRegion, &QPushButton::clicked, this, &MainWindow::onDrawRegion);
    connect(ui->btnClearRegion, &QPushButton::clicked, this, &MainWindow::onClearRegions);
    connect(ui->btnOfflineDetect, &QPushButton::clicked, this, &MainWindow::onOfflineDetect);
    connect(ui->btnSettings, &QPushButton::clicked, this, &MainWindow::onSetParams);

    // ImageView 多边形绘制完成信号
    if (m_imageView) {
        connect(m_imageView, &ImageView::polygonCompleted, this, &MainWindow::onPolygonCompleted);
    }

    // 初始化访客记录表格
    if (ui->tableVisitors) {
        // 人员进出记录格式：(乘客ID, 上车时间, 下车时间) + 停留/状态
        ui->tableVisitors->setColumnCount(5);
        ui->tableVisitors->setHorizontalHeaderLabels({"乘客ID", "上车时间", "下车时间", "停留", "状态"});
        ui->tableVisitors->horizontalHeader()->setStretchLastSection(true);
        ui->tableVisitors->setEditTriggers(QAbstractItemView::NoEditTriggers);
        ui->tableVisitors->setSelectionBehavior(QAbstractItemView::SelectRows);
        ui->tableVisitors->setSelectionMode(QAbstractItemView::SingleSelection);
        // 设置列宽（紧凑布局）
        ui->tableVisitors->setColumnWidth(0, 54);   // 乘客ID
        ui->tableVisitors->setColumnWidth(1, 68);   // 上车时间
        ui->tableVisitors->setColumnWidth(2, 68);   // 下车时间
        ui->tableVisitors->setColumnWidth(3, 46);   // 停留
        // 第5列(状态)自动拉伸
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

    // 创建通道监控器（上车/下车检测）
    m_zoneMonitor = new ZoneMonitor(this);
    m_zoneMonitor->setReidMatchThreshold(m_config.reidMatchThreshold);
    m_zoneMonitor->setConfirmFrames(3);

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
            logEntry.details = QString("上车(从上往下 A→B) 乘客P%1, enter=%2, exit=%3")
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

        updateVisitorTable();
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
    });

    m_elapsedTimer.start();
    m_lastUiUpdate.start();
    m_fpsTimer.start();

    updateCaptureStatus();
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
        logEntry.details = QString("下车(从下往上) 乘客P%1, 与上车图像比对=%2(sim=%3), enter=%4, exit=%5")
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

        // 按配置选择帧源：Orbbec Gemini 335 深度相机 / RTSP 网络相机
        if (m_config.useOrbbec) {
            if (!m_camera.openOrbbec(m_config.orbbecColorWidth,
                                     m_config.orbbecColorHeight,
                                     m_config.orbbecFps)) {
                QMessageBox::critical(this, "错误",
                    QString("无法打开 Orbbec 深度相机。\n%1\n\n请确认：\n"
                            "  1) 相机已连接（lsusb 可见 2bc5:0800）\n"
                            "  2) 已安装 udev 规则 99-obsensor-libusb.rules\n"
                            "  3) 当前用户对 /dev/bus/usb 有读写权限")
                        .arg(m_camera.lastError()));
                return;
            }
            LOG_INFO("Orbbec camera opened: " + m_camera.deviceName());
            if (m_depthView) m_depthView->setVisible(m_camera.hasDepth());
        } else {
            if (!m_camera.openCamera(m_config.rtspUrl)) {
                QMessageBox::critical(this, "错误", "无法打开 RTSP 流:\n" + m_config.rtspUrl);
                return;
            }
            LOG_INFO("Camera opened: " + m_config.rtspUrl);
            if (m_depthView) m_depthView->setVisible(false);   // RTSP 无深度输出
        }

        // 创建采集线程
        m_grabThread = new GrabThread(&m_camera, "HX-CCD23", this);

        // 创建检测管线
        m_detector = new PersonDetector();
        m_tracker = new PersonTracker(m_config.trackerIouThreshold, m_config.trackerMaxMissed);
        m_tracker->setKalmanEnabled(m_config.trackerKalmanEnabled);
        m_tracker->setReidMatchingEnabled(m_config.trackerReidEnabled);
        m_tracker->setReidMatchThreshold(m_config.trackerReidThreshold);
        m_tracker->setGalleryMaxAge(m_config.trackerGalleryMaxAgeMs);
        m_reidExtractor = new ReIDExtractor();
        // 多特征融合权重（颜色/纹理/身形），权重内部自动归一化
        m_reidExtractor->setComponentWeights(m_config.reidWeightColor,
                                             m_config.reidWeightTexture,
                                             m_config.reidWeightShape);
        // 深度 ReID（OSNet 现场微调）：启用后 extract/compare 自动改走 512 维深度特征
        m_reidExtractor->setDeepReid(m_deepReid, m_config.reidUseDeepReid);
        m_reidExtractor->setBoxExpandForReid(m_config.reidHeadBoxExpand);
        LOG_INFO(QString("[MainWindow] ReID 后端=%1 (deep=%2, headBoxExpand=%3)")
                     .arg(m_reidExtractor->deepEnabled() ? "深度OSNet(512D)" : "手工多模态")
                     .arg(m_config.reidUseDeepReid ? "on" : "off")
                     .arg(m_config.reidHeadBoxExpand, 0, 'f', 2));
        // 下车比对线程的独立提取器同步同一套权重
        if (m_matchWorker) {
            m_matchWorker->setComponentWeights(m_config.reidWeightColor,
                                               m_config.reidWeightTexture,
                                               m_config.reidWeightShape);
        }

        // 背景扣除（固定摄像头）：绑定背景模型，特征提取时剔除背景像素
        if (m_config.reidUseBackground) {
            if (!m_bgModel) m_bgModel = new BackgroundModel();
            m_bgModel->reset();   // 每次启动采集重新学习背景
            m_reidExtractor->setBackgroundModel(m_bgModel);
            LOG_INFO(QString("[MainWindow] 背景扣除已启用（ReID 仅统计人体前景像素），加速后端=%1")
                         .arg(m_bgModel->usingGpu() ? "CUDA(GPU)" : "CPU"));
        } else {
            m_reidExtractor->setBackgroundModel(nullptr);
        }

        // 乘客台账：绑定 ReID 提取器与匹配阈值，用于上下车图像比对
        if (m_passengerBook) {
            m_passengerBook->setReidExtractor(m_reidExtractor);
            m_passengerBook->setMatchThreshold(effectiveReidThreshold());
        }

        // 加载检测模型：优先轻量 ONNX 小模型（若存在），否则使用配置的 engine
        QString modelPath = pickDetectorModelPath();
        if (!QFileInfo(modelPath).isAbsolute()) {
            modelPath = QCoreApplication::applicationDirPath() + "/" + modelPath;
        }

        // ONNX(DNN) 后端配置必须在 loadModel 之前设置
        m_detector->setDnnUseCuda(m_config.detectorDnnUseCuda);
        m_detector->setDnnInputSize(m_config.detectorDnnInputSize);
        // TensorRT 预处理模式（head 模型建议 squash：Letterbox=false）
        m_detector->setLetterbox(m_config.detectorLetterbox);

        // 可选：配置了下载地址且目标文件缺失时，后台拉取（下载完成后下次生效）
        if (!m_config.detectorModelUrl.isEmpty()) {
            downloadDetectorModel(m_config.detectorModelUrl, m_config.detectorOnnxPath);
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

        // 尺寸/长宽比过滤：依据 detectorModelKind（person/head）自动选择参数
        applyDetectorFilters(m_detector);

        // 设置检测管线到采集线程
        m_grabThread->setDetector(m_detector);
        m_grabThread->setTracker(m_tracker);
        m_grabThread->setReidExtractor(m_reidExtractor);
        m_grabThread->setBackgroundModel(m_config.reidUseBackground ? m_bgModel : nullptr);

        // 人物片段录制：画面出现人即录制，连续无人超过间隔后分段保存
        {
            QString clipDir = m_config.personClipPath;
            if (!QDir(clipDir).isAbsolute())
                clipDir = QCoreApplication::applicationDirPath() + "/" + clipDir;
            m_grabThread->setPersonClipRecording(m_config.personClipEnabled, clipDir,
                                                 m_config.personClipGapMs, m_config.personClipFps);
        }

        // 连接信号
        connect(m_grabThread, &GrabThread::deviceLost, this, &MainWindow::onGrabThreadDeviceLost);
        connect(m_grabThread, &GrabThread::frameProcessed, this, &MainWindow::onFrameProcessed);
        connect(m_grabThread, &GrabThread::frameGrabbed, this, &MainWindow::onFrameGrabbed);
        connect(m_grabThread, &GrabThread::depthFrameGrabbed, this, &MainWindow::onDepthFrameGrabbed);
        connect(m_grabThread, &GrabThread::rawFrameGrabbed, this, [this](const cv::Mat &frame) {
            QMutexLocker locker(&m_rawFrameMutex);
            m_currentRawFrame = frame.clone();
        });

        m_grabThread->start();
        LOG_INFO("[MainWindow] GrabThread started with detection pipeline");

        // 配置通道监控器
        if (m_zoneMonitor) {
            m_zoneMonitor->setReidExtractor(m_reidExtractor);
            m_zoneMonitor->setReidMatchThreshold(effectiveReidThreshold());
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
    if (!m_depthView || img.isNull())
        return;
    // 深度视图仅显示伪彩，不做检测叠加；首次收到深度帧时确保可见
    if (!m_depthView->isVisible())
        m_depthView->setVisible(true);
    m_depthView->setImage(img);

    // 同步一份原始深度（毫米）供深度视图鼠标悬停查询：像素坐标 → 距离/高度(m)
    if (m_grabThread) {
        cv::Mat d = m_grabThread->lastDepth();
        if (!d.empty() && d.type() == CV_16U)
            m_depthView->setDepthData(d.cols, d.rows, d.ptr<quint16>());
    }
    // 尺寸自适应：ImageView 仅在首帧自动 fit，这里每帧重新适配，
    // 使深度图随分屏器/窗口尺寸变化保持等比铺满（深度视图为只读，无缩放交互冲突）。
    m_depthView->resetView();
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
    // 注意：检测管线启用时，画面统一由 onFrameProcessed 输出（其已叠加检测框/区域），
    // 这里若再用“原始帧”覆盖显示，会把刚绘制好的区域覆盖掉（表现为“绘制后不显示”）。
    const bool pipelineActive = (m_detector && m_detector->isLoaded());
    const bool shouldRender = m_lastUiUpdate.elapsed() >= UI_UPDATE_INTERVAL_MS;
    if (shouldRender && !pipelineActive) {
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

    // 记录格式：(乘客ID, 上车时间, 下车时间) + 停留 / 状态
    int row = 0;
    for (auto it = m_visitors.constBegin(); it != m_visitors.constEnd(); ++it) {
        const VisitorRecord &vr = it.value();

        if (row >= ui->tableVisitors->rowCount()) {
            ui->tableVisitors->insertRow(row);
        }

        // 乘客ID（P + 编号）
        QTableWidgetItem *idItem = ui->tableVisitors->item(row, 0);
        if (!idItem) {
            idItem = new QTableWidgetItem();
            ui->tableVisitors->setItem(row, 0, idItem);
        }
        idItem->setText(QString("P%1").arg(vr.visitorId));
        idItem->setTextAlignment(Qt::AlignCenter);
        idItem->setForeground(QBrush(QColor(0, 229, 255)));

        // 上车时间（从上往下 = 上车）
        QTableWidgetItem *enterItem = ui->tableVisitors->item(row, 1);
        if (!enterItem) {
            enterItem = new QTableWidgetItem();
            ui->tableVisitors->setItem(row, 1, enterItem);
        }
        enterItem->setText(vr.enterTime.isValid() ? vr.enterTime.toString("HH:mm:ss") : "--");
        enterItem->setTextAlignment(Qt::AlignCenter);
        enterItem->setForeground(QBrush(QColor(0, 200, 255))); // 青色

        // 下车时间（从下往上 = 下车）
        QTableWidgetItem *exitItem = ui->tableVisitors->item(row, 2);
        if (!exitItem) {
            exitItem = new QTableWidgetItem();
            ui->tableVisitors->setItem(row, 2, exitItem);
        }
        exitItem->setText(vr.exitTime.isValid() ? vr.exitTime.toString("HH:mm:ss") : "--");
        exitItem->setTextAlignment(Qt::AlignCenter);
        exitItem->setForeground(QBrush(QColor(255, 171, 64))); // 橙色

        // 停留时长
        QTableWidgetItem *durationItem = ui->tableVisitors->item(row, 3);
        if (!durationItem) {
            durationItem = new QTableWidgetItem();
            ui->tableVisitors->setItem(row, 3, durationItem);
        }
        if (vr.enterTime.isValid()) {
            QDateTime endTime = vr.exitTime.isValid() ? vr.exitTime : QDateTime::currentDateTime();
            qint64 secs = vr.enterTime.secsTo(endTime);
            int mins = secs / 60;
            int s = secs % 60;
            durationItem->setText(QString("%1:%2").arg(mins, 2, 10, QChar('0')).arg(s, 2, 10, QChar('0')));
        } else {
            durationItem->setText("--");
        }
        durationItem->setTextAlignment(Qt::AlignCenter);

        // 状态：已上车(在途) / 已下车 / 已离开
        QTableWidgetItem *statusItem = ui->tableVisitors->item(row, 4);
        if (!statusItem) {
            statusItem = new QTableWidgetItem();
            ui->tableVisitors->setItem(row, 4, statusItem);
        }
        if (vr.isInside) {
            // 注意：isInside 仅表示「人在检测区域内（跟踪中）」，与上/下车方向无关；
            // 方向由 ZoneMonitor 的完整穿越判定给出（vr.direction）。
            // 之前此处写死为「已上车」，导致“下车过程中的人”也被显示成“已上车”。
            if (vr.direction == QStringLiteral("alighting")) {
                statusItem->setText("● 下车中");
                statusItem->setForeground(QBrush(QColor(255, 171, 64)));  // 橙色
            } else if (vr.direction == QStringLiteral("boarding")) {
                statusItem->setText("● 上车中");
                statusItem->setForeground(QBrush(QColor(0, 230, 118)));   // 绿色
            } else {
                statusItem->setText("● 区域内");
                statusItem->setForeground(QBrush(QColor(0, 230, 118)));   // 绿色
            }
        } else if (vr.exitTime.isValid()) {
            statusItem->setText("○ 已下车");
            statusItem->setForeground(QBrush(QColor(176, 190, 197))); // 灰色
        } else {
            statusItem->setText("○ 已离开");
            statusItem->setForeground(QBrush(QColor(120, 120, 120)));
        }
        statusItem->setTextAlignment(Qt::AlignCenter);

        row++;
    }

    // 移除多余的行
    while (ui->tableVisitors->rowCount() > row) {
        ui->tableVisitors->removeRow(ui->tableVisitors->rowCount() - 1);
    }

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
                    int visitorId = matchOrCreateVisitor(p, rawFrame, currentRegionId);
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

    // 仅当到达刷新间隔时才做“重绘 + 显示”。
    // 否则跳过全部全帧绘制（区域/框/标签），直接返回并解除采集线程背压，
    // 避免每帧都做昂贵绘制导致 UI 队列积压、画面越来越滞后。
    if (m_lastUiUpdate.elapsed() < UI_UPDATE_INTERVAL_MS) {
        m_currentPersonCount = personInRegionCount;
        if (m_grabThread) m_grabThread->setUiBusy(false);
        return;
    }

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

                // 标注：门线 + 方向（↑下车 / ↓上车）
                const QString lbl = (region.entryDirection == EntryDirection::TopToBottom)
                                        ? QStringLiteral("DOOR 上车↓") : QStringLiteral("DOOR 下车↑");
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
    cv::Mat rgbDisplay;
    cv::cvtColor(displayFrame, rgbDisplay, cv::COLOR_BGR2RGB);
    QImage displayImg(rgbDisplay.data, rgbDisplay.cols, rgbDisplay.rows,
                      static_cast<int>(rgbDisplay.step), QImage::Format_RGB888);
    displayImg = displayImg.copy();

    // 显示（已在上方节流判定通过）
    if (m_imageView) m_imageView->setImage(displayImg);
    m_lastUiUpdate.restart();

    // 更新 FPS 和人数显示
    ui->label_fps->setText(QString("FPS: %1").arg(m_currentFps, 0, 'f', 1));
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
    if (m_grabThread) m_grabThread->setUiBusy(false);
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
    const QString fileName = QString("%1/V%2_t%3_%4_%5.jpg")
                                 .arg(pending).arg(visitorId).arg(trackId)
                                 .arg(seq, 4, 10, QChar('0')).arg(ts);
    if (!cv::imwrite(fileName.toStdString(), crop)) {
        LOG_WARN(QString("[MainWindow] 访客帧保存失败: %1").arg(fileName));
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

    // 一次采集会话 = 一个行程
    m_currentTripId = QStringLiteral("TRIP-%1")
                          .arg(QDateTime::currentDateTime().toString("yyyyMMdd-HHmmss"));
    m_currentStationId = m_config.currentStationId;
    m_currentStationName = m_config.currentStationName;
    m_tripStore->beginTrip(m_currentTripId, m_config.vehicleId, QDateTime::currentDateTime());
    m_onboardCount = m_tripStore->onboardCount(m_currentTripId);

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
    ui->label_personCount->setText(QString("区域人数: %1 | 上车: %2 | 下车: %3 | 车厢内: %4")
                                       .arg(m_currentPersonCount)
                                       .arg(boardingCnt)
                                       .arg(alightingCnt)
                                       .arg(m_onboardCount));
}

// ===== 深度 ReID（OSNet 现场微调模型）=====
void MainWindow::initDeepReid()
{
    if (!m_config.reidUseDeepReid) {
        LOG_INFO("[MainWindow] 深度 ReID 未启用（使用手工多模态特征）");
        return;
    }

    if (!m_deepReid) m_deepReid = new DeepReID();

    QString modelPath = m_config.reidDeepModelPath;
    if (modelPath.isEmpty()) modelPath = QStringLiteral("./model/osnet_x1_0_head_finetuned.onnx");
    if (!QDir(modelPath).isAbsolute())
        modelPath = QCoreApplication::applicationDirPath() + "/" + modelPath;

    // 深度 ReID 推理后端：优先 CUDA(GPU)，不可用时自动回退 CPU（见 reiddeep.cpp）
    m_deepReid->setUseCuda(m_config.reidDeepUseCuda);

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
    if (m_passengerBook) m_passengerBook->setMatchThreshold(effectiveReidThreshold());

    LOG_INFO(QString("[MainWindow] 深度 ReID 已加载: %1 | backend=%2 (生效阈值=%3)，手工 ReID 自动改走深度特征")
                 .arg(m_deepReid->modelPath())
                 .arg(m_deepReid->cudaActive() ? QStringLiteral("CUDA(GPU)") : QStringLiteral("CPU"))
                 .arg(effectiveReidThreshold(), 0, 'f', 2));
}

float MainWindow::effectiveReidThreshold() const
{
    if (m_deepReid && m_deepReid->ready() && m_config.reidUseDeepReid)
        return m_config.reidDeepMatchThreshold;
    return m_config.reidMatchThreshold;
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
        msgBox.setWindowTitle("绘制上下车分界线 - 操作引导");
        msgBox.setText("⚠️ 当前没有画面，绘制的分界线坐标可能不准确。\n\n"
                       "📋 推荐操作顺序：\n"
                       "  ① 先点击「启动采集」获取实时画面\n"
                       "  ② 然后点击「绘制上下车分界线」\n"
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
        ui->btnDrawRegion->setText("绘制上下车分界线");
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

    // 清除访客记录
    m_visitors.clear();
    m_trackToVisitor.clear();
    m_nextVisitorId = 1;
    m_visitorPendingDir.clear();
    m_visitorFrameTick.clear();
    m_visitorSegmentSeq.clear();

    // 重置通道监控器
    if (m_zoneMonitor) {
        m_zoneMonitor->reset();
    }
    m_boardingCount = 0;
    m_alightingCount = 0;

    // 清空乘客台账（上车记录）
    if (m_passengerBook) m_passengerBook->clear();

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

    const bool camReopened = m_config.useOrbbec
        ? m_camera.openOrbbec(m_config.orbbecColorWidth, m_config.orbbecColorHeight, m_config.orbbecFps)
        : m_camera.openCamera(m_config.rtspUrl);
    if (camReopened) {
        m_reconnectTimer->stop();
        if (m_depthView) m_depthView->setVisible(m_camera.hasDepth());

        // 重新创建采集线程（含检测管线）
        m_grabThread = new GrabThread(&m_camera, "HX-CCD23", this);

        if (m_detector) m_grabThread->setDetector(m_detector);
        if (m_tracker) m_grabThread->setTracker(m_tracker);
        if (m_reidExtractor) m_grabThread->setReidExtractor(m_reidExtractor);

        connect(m_grabThread, &GrabThread::deviceLost, this, &MainWindow::onGrabThreadDeviceLost);
        connect(m_grabThread, &GrabThread::frameProcessed, this, &MainWindow::onFrameProcessed);
        connect(m_grabThread, &GrabThread::frameGrabbed, this, &MainWindow::onFrameGrabbed);
        connect(m_grabThread, &GrabThread::depthFrameGrabbed, this, &MainWindow::onDepthFrameGrabbed);
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
    if (m_grabThread) {
        m_grabThread->setBackgroundModel(m_config.reidUseBackground ? m_bgModel : nullptr);

        // 人物片段录制参数同步（运行中修改配置即刻生效）
        QString clipDir = m_config.personClipPath;
        if (!QDir(clipDir).isAbsolute())
            clipDir = QCoreApplication::applicationDirPath() + "/" + clipDir;
        m_grabThread->setPersonClipRecording(m_config.personClipEnabled, clipDir,
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
