#include "offlinedetectform.h"
#include "logger.h"
#include <QMessageBox>
#include <QTimer>
#include <QJsonArray>
#include <QJsonObject>
#include <QJsonDocument>
#include <QFile>
#include <QHeaderView>
#include <QFileInfo>
#include <QDir>
#include <QCoreApplication>
#include <QMap>
#include <QStringList>
#include <algorithm>

// 方向判定约定（与在线一致）——【相机装反 / 视角反转后的新定义】：
//   人从上往下穿过检测区域：进入侧=A(顶) → 离开侧=B(底)  => 下车
//   人从下往上穿过检测区域：进入侧=B(底) → 离开侧=A(顶)  => 上车

OfflineDetectForm::OfflineDetectForm(PersonDetector *detector,
                                     RegionManager *regionManager,
                                     bool useDeepReid,
                                     const QString &deepModelPath,
                                     float deepMatchThreshold,
                                     QWidget *parent)
    : QDialog(parent)
    , m_detector(detector)
    , m_regionManager(regionManager)
{
    // ===== 深度 ReID（OSNet/ONNX）初始化（需在 setupUI 前确定可用性）=====
    m_deepThreshold = deepMatchThreshold;
    QString deepPath = deepModelPath;
    if (deepPath.isEmpty()) {
        deepPath = QCoreApplication::applicationDirPath() + "/model/osnet_x1_0_msmt17.onnx";
    } else if (!QDir(deepPath).isAbsolute()) {
        deepPath = QCoreApplication::applicationDirPath() + "/" + deepPath;
    }
    bool deepOk = m_deepReid.load(deepPath);
    if (!deepOk) {
        // 回退：源码目录（开发环境）
        const QString alt = QCoreApplication::applicationDirPath()
                            + "/../CameraViewer_passenger/model/osnet_x1_0_msmt17.onnx";
        deepOk = m_deepReid.load(alt);
    }
    m_useDeepReid = useDeepReid && deepOk;
    if (useDeepReid && !deepOk) {
        LOG_WARN("[OfflineDetect] 深度 ReID 模型不可用，回退到手工特征");
    }

    setupUI();
    setWindowTitle("离线图像检测（含上车/下车方向判定）");
    resize(1280, 880);

    // 创建独立的 tracker 和 reid extractor 用于离线检测
    m_tracker = new PersonTracker(0.3f, 30);
    m_reidExtractor = new ReIDExtractor();

    // 复用通道状态机做方向判定；离线图片间隔较大，进入/离开确认帧数设为 1
    m_zoneMonitor = new ZoneMonitor(this);
    m_zoneMonitor->setReidExtractor(m_reidExtractor);
    m_zoneMonitor->setConfirmFrames(1);
    // 离线图像/视频间隔大，单帧即可确认启动与离开；并关闭“同人反向冷却”，
    // 使一段视频里若同时出现上/下车都能被统计出来（便于判定视频性质）。
    m_zoneMonitor->setStartConfirmFrames(1);
    m_zoneMonitor->setReverseCooldownMs(0);

    // 上车：B→A（从下往上）
    connect(m_zoneMonitor, &ZoneMonitor::boardingEvent, this, [this](const ZoneEvent &evt) {
        m_boardCount++;
        m_trackDirection[evt.trackId] = QStringLiteral("上车 ↑(从下往上)");
        LOG_INFO(QString("[OfflineDetect] ★ 上车: track=%1, B(底)→A(顶), enter=%2, exit=%3")
                     .arg(evt.trackId)
                     .arg(evt.enterTime.toString("HH:mm:ss.zzz"))
                     .arg(evt.exitTime.toString("HH:mm:ss.zzz")));
        if (m_labelInfo)
            m_labelInfo->setText(QString("方向判定：上车 %1 次，下车 %2 次")
                                     .arg(m_boardCount).arg(m_alightCount));
    });

    // 下车：A→B（从上往下）
    connect(m_zoneMonitor, &ZoneMonitor::alightingEvent, this, [this](const ZoneEvent &evt) {
        m_alightCount++;
        m_trackDirection[evt.trackId] = QStringLiteral("下车 ↓(从上往下)");
        LOG_INFO(QString("[OfflineDetect] ★ 下车: track=%1, A(顶)→B(底), enter=%2, exit=%3")
                     .arg(evt.trackId)
                     .arg(evt.enterTime.toString("HH:mm:ss.zzz"))
                     .arg(evt.exitTime.toString("HH:mm:ss.zzz")));
        if (m_labelInfo)
            m_labelInfo->setText(QString("方向判定：上车 %1 次，下车 %2 次")
                                     .arg(m_boardCount).arg(m_alightCount));
    });

    // ===== 视频判定定时器：边播放边检测，避免阻塞界面 =====
    m_videoTimer = new QTimer(this);
    connect(m_videoTimer, &QTimer::timeout, this, &OfflineDetectForm::onVideoTick);
}

OfflineDetectForm::~OfflineDetectForm()
{
    if (m_videoTimer) m_videoTimer->stop();
    if (m_videoCap.isOpened()) m_videoCap.release();
    delete m_tracker;
    delete m_reidExtractor;
}

void OfflineDetectForm::setupUI()
{
    auto *mainLayout = new QHBoxLayout(this);
    mainLayout->setSpacing(8);
    mainLayout->setContentsMargins(8, 8, 8, 8);

    // 左侧：图像显示区
    auto *leftLayout = new QVBoxLayout();
    leftLayout->setSpacing(6);

    // 按钮行
    auto *btnLayout = new QHBoxLayout();
    m_btnLoad = new QPushButton("📂 加载图像(可多选)");
    m_btnLoad->setMinimumSize(150, 32);
    m_btnDetect = new QPushButton("🔍 检测当前图片");
    m_btnDetect->setMinimumSize(130, 32);
    m_btnDetect->setEnabled(false);
    m_btnBatch = new QPushButton("▶ 连续检测全部");
    m_btnBatch->setMinimumSize(130, 32);
    m_btnBatch->setEnabled(false);
    m_btnVideo = new QPushButton("🎬 加载视频判定上/下车");
    m_btnVideo->setMinimumSize(175, 32);
    m_btnVideo->setToolTip("选择一段视频：窗口内即时播放画面，同时逐帧检测并判定「上车/下车」");
    m_btnVideoStop = new QPushButton("⏹ 停止判定");
    m_btnVideoStop->setMinimumSize(100, 32);
    m_btnVideoStop->setEnabled(false);
    m_btnVideoStop->setToolTip("停止当前视频播放/检测并给出结论");
    m_labelInfo = new QLabel("请加载图像（可多选，用于连续方向判定）");
    m_labelInfo->setStyleSheet("color: #78ceff; font-family: 'Consolas'; font-size: 12px;");

    btnLayout->addWidget(m_btnLoad);
    btnLayout->addWidget(m_btnDetect);
    btnLayout->addWidget(m_btnBatch);
    btnLayout->addWidget(m_btnVideo);
    btnLayout->addWidget(m_btnVideoStop);
    btnLayout->addWidget(m_labelInfo);
    btnLayout->addStretch();

    leftLayout->addLayout(btnLayout);

    // 第二行：上/下车图像相似度比对（判定是否同一乘客）
    auto *row2 = new QHBoxLayout();
    m_btnLoadBoard = new QPushButton("⬆ 加载上车图像(可多选)");
    m_btnLoadBoard->setMinimumSize(175, 30);
    m_btnLoadAlight = new QPushButton("⬇ 加载下车图像(可多选)");
    m_btnLoadAlight->setMinimumSize(175, 30);
    m_btnCompare = new QPushButton("🔍 比较相似度");
    m_btnCompare->setMinimumSize(130, 30);
    m_labelBoardCount = new QLabel("上车: 0 张");
    m_labelAlightCount = new QLabel("下车: 0 张");
    m_labelBoardCount->setStyleSheet("color:#81c784;font-family:'Consolas';font-size:12px;");
    m_labelAlightCount->setStyleSheet("color:#ffb74d;font-family:'Consolas';font-size:12px;");
    row2->addWidget(m_btnLoadBoard);
    row2->addWidget(m_btnLoadAlight);
    row2->addWidget(m_btnCompare);
    m_checkDeepReid = new QCheckBox("深度ReID(OSNet)");
    m_checkDeepReid->setChecked(m_useDeepReid);
    m_checkDeepReid->setEnabled(m_deepReid.ready());
    m_checkDeepReid->setStyleSheet("color:#b0bec5;font-family:'Consolas';font-size:12px;");
    m_checkDeepReid->setToolTip("勾选：使用深度外观嵌入(OSNet/MSMT17)比对；取消：使用颜色+纹理+身形手工特征");
    row2->addWidget(m_checkDeepReid);
    row2->addWidget(m_labelBoardCount);
    row2->addWidget(m_labelAlightCount);
    row2->addStretch();
    leftLayout->addLayout(row2);

    // ImageView
    m_imageView = new ImageView(this);
    m_imageView->setMinimumSize(640, 480);
    leftLayout->addWidget(m_imageView, 1);

    // ===== 上车 / 下车 图像预览窗体（多张时默认显示第一张，可翻页）=====
    auto *previewRow = new QHBoxLayout();
    previewRow->setSpacing(8);

    auto makeSide = [this](const QString &title, const QColor &color,
                           ImageView *&view, QLabel *&info,
                           QPushButton *&prev, QPushButton *&next) {
        auto *box = new QVBoxLayout();
        box->setSpacing(4);

        auto *head = new QHBoxLayout();
        auto *lbl = new QLabel(title);
        lbl->setStyleSheet(QString("color:%1;font-family:'Consolas';font-size:12px;font-weight:bold;")
                               .arg(color.name()));
        info = new QLabel(QStringLiteral("(未加载)"));
        info->setStyleSheet("color:#90a4ae;font-family:'Consolas';font-size:11px;");
        info->setTextInteractionFlags(Qt::TextSelectableByMouse);
        prev = new QPushButton(QStringLiteral("◀"));
        next = new QPushButton(QStringLiteral("▶"));
        const QList<QPushButton *> navBtns{prev, next};
        for (QPushButton *b : navBtns) {
            b->setFixedSize(22, 20);
            b->setToolTip(QStringLiteral("上一张 / 下一张"));
        }
        head->addWidget(lbl);
        head->addWidget(info, 1);
        head->addWidget(prev);
        head->addWidget(next);
        box->addLayout(head);

        view = new ImageView(this);
        view->setMinimumSize(220, 150);
        view->setMaximumHeight(190);
        box->addWidget(view);
        return box;
    };

    previewRow->addLayout(makeSide(QStringLiteral("上车图像"), QColor(129, 199, 132),
                                   m_boardView, m_labelBoardImage, m_btnBoardPrev, m_btnBoardNext), 1);
    previewRow->addLayout(makeSide(QStringLiteral("下车图像"), QColor(255, 183, 77),
                                   m_alightView, m_labelAlightImage, m_btnAlightPrev, m_btnAlightNext), 1);
    leftLayout->addLayout(previewRow);

    mainLayout->addLayout(leftLayout, 3);

    // 右侧：检测结果面板
    auto *rightPanel = new QFrame();
    rightPanel->setMinimumWidth(280);
    rightPanel->setMaximumWidth(340);
    rightPanel->setStyleSheet("QFrame { background-color: rgba(6, 14, 26, 0.96); border: 1px solid #1a3860; border-radius: 8px; }");

    auto *rightLayout = new QVBoxLayout(rightPanel);
    rightLayout->setSpacing(6);
    rightLayout->setContentsMargins(6, 6, 6, 6);

    auto *titleLabel = new QLabel("◆ 检测结果（含方向）");
    titleLabel->setStyleSheet("color: #00e5ff; font-family: 'Consolas'; font-size: 13px; font-weight: bold;");
    rightLayout->addWidget(titleLabel);

    m_tableResults = new QTableWidget();
    m_tableResults->setColumnCount(4);
    m_tableResults->setHorizontalHeaderLabels({"ID", "方向", "位置", "区域内"});
    m_tableResults->horizontalHeader()->setStretchLastSection(true);
    m_tableResults->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_tableResults->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_tableResults->verticalHeader()->setVisible(false);
    m_tableResults->verticalHeader()->setDefaultSectionSize(22);
    m_tableResults->setColumnWidth(0, 40);
    m_tableResults->setColumnWidth(1, 96);
    m_tableResults->setColumnWidth(2, 90);
    m_tableResults->setStyleSheet(
        "QTableWidget { background-color: #080f1e; border: 1px solid #1a3058; border-radius: 4px; "
        "color: #b8d4f0; font-family: 'Consolas'; font-size: 11px; gridline-color: #122040; }"
        "QTableWidget::item { padding: 2px 4px; }"
        "QHeaderView::section { background-color: #0a1a30; color: #78ceff; font-weight: bold; "
        "border: 1px solid #1a3058; padding: 3px 4px; }");

    rightLayout->addWidget(m_tableResults, 1);

    m_labelCompare = new QLabel("相似度比对：\n请分别加载“上车图像”与“下车图像”，再点「比较相似度」。");
    m_labelCompare->setWordWrap(true);
    m_labelCompare->setMinimumHeight(72);
    m_labelCompare->setStyleSheet("color:#b0bec5;font-family:'Consolas';font-size:12px;"
                                  "background:rgba(0,0,0,0.25);border:1px solid #1a3058;"
                                  "border-radius:4px;padding:6px;");
    rightLayout->addWidget(m_labelCompare);

    m_labelVideoResult = new QLabel("视频判定：\n（未运行）");
    m_labelVideoResult->setWordWrap(true);
    m_labelVideoResult->setMinimumHeight(56);
    m_labelVideoResult->setStyleSheet("color:#81c784;font-family:'Consolas';font-size:12px;"
                                      "background:rgba(0,0,0,0.25);border:1px solid #1a3058;"
                                      "border-radius:4px;padding:6px;");
    rightLayout->addWidget(m_labelVideoResult);

    mainLayout->addWidget(rightPanel, 1);

    // 连接信号
    connect(m_btnLoad, &QPushButton::clicked, this, &OfflineDetectForm::onLoadImage);
    connect(m_btnDetect, &QPushButton::clicked, this, &OfflineDetectForm::onRunDetect);
    connect(m_btnBatch, &QPushButton::clicked, this, &OfflineDetectForm::onBatchDetect);
    connect(m_btnVideo, &QPushButton::clicked, this, &OfflineDetectForm::onVideoDetect);
    connect(m_btnVideoStop, &QPushButton::clicked, this, &OfflineDetectForm::onVideoStop);
    connect(m_btnLoadBoard, &QPushButton::clicked, this, &OfflineDetectForm::onLoadBoardingImages);
    connect(m_btnLoadAlight, &QPushButton::clicked, this, &OfflineDetectForm::onLoadAlightingImages);
    connect(m_btnCompare, &QPushButton::clicked, this, &OfflineDetectForm::onCompareGroups);
    connect(m_checkDeepReid, &QCheckBox::toggled, this, &OfflineDetectForm::onDeepReidToggled);
    connect(m_btnBoardPrev, &QPushButton::clicked, this, &OfflineDetectForm::onBoardPrev);
    connect(m_btnBoardNext, &QPushButton::clicked, this, &OfflineDetectForm::onBoardNext);
    connect(m_btnAlightPrev, &QPushButton::clicked, this, &OfflineDetectForm::onAlightPrev);
    connect(m_btnAlightNext, &QPushButton::clicked, this, &OfflineDetectForm::onAlightNext);
}

void OfflineDetectForm::refreshZoneMonitor()
{
    if (m_zoneMonitor && m_regionManager) {
        m_zoneMonitor->setChannelZones(m_regionManager->regions());
    }
}

void OfflineDetectForm::onLoadImage()
{
    QStringList files = QFileDialog::getOpenFileNames(
        this, "选择图像文件（可多选，按文件名顺序连续判定）", QString(),
        "图像文件 (*.jpg *.jpeg *.png *.bmp *.tiff);;所有文件 (*)");

    if (files.isEmpty()) return;

    m_imagePaths = files;
    std::sort(m_imagePaths.begin(), m_imagePaths.end()); // 按名称排序，保证时间顺序
    m_currentIndex = 0;

    m_currentImagePath = m_imagePaths.first();
    m_currentImage = cv::imread(m_currentImagePath.toStdString());
    if (m_currentImage.empty()) {
        QMessageBox::warning(this, "错误", "无法加载图像: " + m_currentImagePath);
        m_btnDetect->setEnabled(false);
        m_btnBatch->setEnabled(false);
        return;
    }

    // 显示图像
    cv::Mat rgb;
    cv::cvtColor(m_currentImage, rgb, cv::COLOR_BGR2RGB);
    QImage img(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888);
    m_imageView->setImage(img.copy());

    // 设置区域多边形到 ImageView 并刷新通道定义
    refreshZoneMonitor();
    if (m_regionManager) {
        QVector<QVector<QPointF>> polys;
        for (const auto &r : m_regionManager->regions()) {
            polys.append(r.polygon);
        }
        m_imageView->setRegionPolygons(polys);
    }

    m_btnDetect->setEnabled(true);
    m_btnBatch->setEnabled(m_imagePaths.size() >= 1);
    m_labelInfo->setText(QString("已加载 %1 张，当前: %2 (%3x%4)")
                             .arg(m_imagePaths.size())
                             .arg(QFileInfo(m_currentImagePath).fileName())
                             .arg(m_currentImage.cols).arg(m_currentImage.rows));

    m_tableResults->setRowCount(0);
    LOG_INFO(QString("[OfflineDetect] Loaded %1 image(s), first=%2")
                 .arg(m_imagePaths.size()).arg(m_currentImagePath));
}

void OfflineDetectForm::onRunDetect()
{
    if (m_currentImage.empty() || !m_detector || !m_detector->isLoaded()) {
        QMessageBox::warning(this, "错误", "未加载图像或检测模型未就绪");
        return;
    }
    processFrameInternal(m_currentImage);
}

void OfflineDetectForm::onBatchDetect()
{
    if (m_imagePaths.isEmpty() || !m_detector || !m_detector->isLoaded()) {
        QMessageBox::warning(this, "错误", "未加载图像序列或检测模型未就绪");
        return;
    }

    // 重置序列状态，从头开始连续判定
    m_zoneMonitor->reset();
    delete m_tracker;
    m_tracker = new PersonTracker(0.3f, 30);
    m_trackDirection.clear();
    m_boardCount = 0;
    m_alightCount = 0;
    m_tableResults->setRowCount(0);

    LOG_INFO(QString("[OfflineDetect] ===== 连续检测开始: %1 张 =====").arg(m_imagePaths.size()));

    for (int i = 0; i < m_imagePaths.size(); ++i) {
        cv::Mat img = cv::imread(m_imagePaths[i].toStdString());
        if (img.empty()) {
            LOG_WARN(QString("[OfflineDetect] 跳过无法读取的图片: %1").arg(m_imagePaths[i]));
            continue;
        }
        m_currentIndex = i;
        m_currentImagePath = m_imagePaths[i];
        m_currentImage = img;

        LOG_INFO(QString("[OfflineDetect] [%1/%2] %3")
                     .arg(i + 1).arg(m_imagePaths.size())
                     .arg(QFileInfo(m_imagePaths[i]).fileName()));

        processFrameInternal(img);
    }

    LOG_INFO(QString("[OfflineDetect] ===== 连续检测完成: 上车 %1 次, 下车 %2 次 =====")
                 .arg(m_boardCount).arg(m_alightCount));

    QMessageBox::information(this, "连续检测完成",
        QString("连续检测完成。\n\n上车（从下往上）：%1 次\n下车（从上往下）：%2 次")
            .arg(m_boardCount).arg(m_alightCount));
}

void OfflineDetectForm::onVideoDetect()
{
    if (!m_detector || !m_detector->isLoaded()) {
        QMessageBox::warning(this, "错误", "检测模型未就绪，无法进行视频判定");
        return;
    }

    // 若上一次判定仍在进行，先收尾
    if (m_videoActive) finishVideoDetect(QStringLiteral("重新选择视频"));

    const QString path = QFileDialog::getOpenFileName(
        this, "选择待判定的视频（判断整段视频是上车还是下车）", QString(),
        "视频文件 (*.mp4 *.avi *.mkv *.mov *.flv *.wmv *.m4v *.mpg);;所有文件 (*)");
    if (path.isEmpty()) return;

    if (!m_videoCap.open(path.toStdString())) {
        QMessageBox::warning(this, "错误", "无法打开视频文件:\n" + path);
        return;
    }

    // ===== 重置判定状态 =====
    m_zoneMonitor->reset();
    m_zoneMonitor->setConfirmFrames(1);
    m_zoneMonitor->setStartConfirmFrames(1);
    m_zoneMonitor->setReverseCooldownMs(0);   // 允许视频内上/下穿越都被统计
    // 离线视频中目标可能较小/较远，放宽“开始保存”的最小重叠门槛，
    // 避免因门槛过高而始终不进入采样（表现为 0 in region、无任何穿越日志）
    m_zoneMonitor->setMinOverlapArea(0.0002f);
    refreshZoneMonitor();

    delete m_tracker;
    m_tracker = new PersonTracker(0.3f, 75);
    // 视频判定的关联策略：放宽 IoU、降低 ReID 关联阈值，使同一人在短暂丢失后
    // 更容易复联到原轨迹（减少 ID 切换），从而提高完整穿越的判定成功率。
    m_tracker->setIouThreshold(0.15f);
    m_tracker->setReidMatchThreshold(0.62f);
    m_trackDirection.clear();
    m_boardCount = 0;
    m_alightCount = 0;
    m_tableResults->setRowCount(0);

    m_videoPath = path;
    m_videoFrameIdx = 0;
    m_videoProcessed = 0;

    // 轨迹范围统计复位
    m_trajMinCx = 2.0f; m_trajMaxCx = -1.0f;
    m_trajMinCy = 2.0f; m_trajMaxCy = -1.0f;
    m_trackYRange.clear();
    m_trackFirstLast.clear();

    m_videoFps = m_videoCap.get(cv::CAP_PROP_FPS);
    if (!(m_videoFps > 1.0)) m_videoFps = 25.0;

    // 抽帧：把“画面刷新 + 检测”限制在约 12fps，兼顾可见性与处理速度
    m_videoStep = qMax(1, static_cast<int>(m_videoFps / 12.0 + 0.5));

    // 先显示视频首帧，确认选对了文件
    cv::Mat first;
    if (m_videoCap.read(first) && !first.empty()) {
        m_videoFrameIdx++;
        m_currentImage = first;
        cv::Mat rgb;
        cv::cvtColor(first, rgb, cv::COLOR_BGR2RGB);
        QImage qimg(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888);
        m_imageView->setImage(qimg.copy());
    }

    m_videoActive = true;
    if (m_btnVideoStop) m_btnVideoStop->setEnabled(true);
    if (m_labelVideoResult)
        m_labelVideoResult->setText(QString("视频判定中…\n%1").arg(QFileInfo(path).fileName()));

    // 定时器间隔取 0：每处理完一帧即刻让出事件循环，使画面持续刷新、
    // 按钮可响应；实际节奏由“检测耗时”决定（与检测能力匹配）。
    m_videoTimer->start(0);

    LOG_INFO(QString("[OfflineDetect] ===== 视频判定开始(边播边检测): %1 "
                     "(fps=%2, step=%3) =====")
                 .arg(QFileInfo(path).fileName()).arg(m_videoFps, 0, 'f', 2)
                 .arg(m_videoStep));
}

void OfflineDetectForm::onVideoTick()
{
    if (!m_videoActive) return;

    // 读取 step 帧（丢弃中间帧以控制检测频率），对最后一帧做检测+显示
    cv::Mat frame;
    bool okRead = false;
    for (int i = 0; i < m_videoStep; ++i) {
        if (!m_videoCap.read(frame) || frame.empty()) {
            okRead = false;
            break;
        }
        m_videoFrameIdx++;
        okRead = true;
    }
    if (!okRead) {
        finishVideoDetect(QStringLiteral("视频播放结束"));
        return;
    }
    if (frame.cols < 8 || frame.rows < 8) return;

    m_currentImage = frame;
    m_currentImagePath = m_videoPath;

    // 检测 + 跟踪 + 区域方向判定 + 画面显示（每帧都会刷新 ImageView → 可见视频画面）
    try {
        processFrameInternal(frame);
    } catch (const cv::Exception &e) {
        LOG_ERR(QString("[OfflineDetect] 视频判定 cv::Exception: %1")
                    .arg(QString::fromStdString(e.what())));
    } catch (const std::exception &e) {
        LOG_ERR(QString("[OfflineDetect] 视频判定 std::exception: %1")
                    .arg(QString::fromUtf8(e.what())));
    }
    m_videoProcessed++;

    if (m_labelInfo)
        m_labelInfo->setText(QString("视频判定中… 帧 %1 | 上车 %2, 下车 %3")
                                 .arg(m_videoFrameIdx).arg(m_boardCount).arg(m_alightCount));
}

void OfflineDetectForm::onVideoStop()
{
    finishVideoDetect(QStringLiteral("手动停止"));
}

void OfflineDetectForm::finishVideoDetect(const QString &reason)
{
    if (!m_videoActive) return;
    m_videoActive = false;

    if (m_videoTimer) m_videoTimer->stop();
    if (m_videoCap.isOpened()) m_videoCap.release();
    if (m_btnVideoStop) m_btnVideoStop->setEnabled(false);

    // ===== 结论：严格「完整穿越」优先；否则用方向判定（宽松）=====
    QString strictVerdict;
    if (m_boardCount > 0 && m_alightCount == 0)
        strictVerdict = QStringLiteral("上车（完整穿越检测区域）");
    else if (m_alightCount > 0 && m_boardCount == 0)
        strictVerdict = QStringLiteral("下车（完整穿越检测区域）");
    else if (m_boardCount > 0 && m_alightCount > 0)
        strictVerdict = QStringLiteral("多次穿越（上 %1 / 下 %2）")
                            .arg(m_boardCount).arg(m_alightCount);

    // 严格判据无结果 → 方向判定：许多「人物片段」只包含穿越的部分过程
    // （起止点落在区域内部/边界附近），此时按轨迹净位移方向给出结论。
    QString looseDetail;
    if (strictVerdict.isEmpty() && !m_trackFirstLast.isEmpty()) {
        float zoneTop = 0.0f, zoneBot = 1.0f;
        if (m_regionManager) {
            const auto rs = m_regionManager->regions();
            if (!rs.isEmpty()) {
                zoneTop = 2.0f; zoneBot = -1.0f;
                for (const QPointF &pt : rs.first().polygon) {
                    zoneTop = std::min(zoneTop, static_cast<float>(pt.y()));
                    zoneBot = std::max(zoneBot, static_cast<float>(pt.y()));
                }
            }
        }
        const float zoneH = std::max(0.05f, zoneBot - zoneTop);
        const float needDy = std::max(0.08f, zoneH * 0.45f);

        int bestTid = -1;
        float bestDy = 0.0f;
        for (auto it = m_trackFirstLast.constBegin(); it != m_trackFirstLast.constEnd(); ++it) {
            const float dy = it.value().second - it.value().first;
            if (std::fabs(dy) > std::fabs(bestDy)) { bestDy = dy; bestTid = it.key(); }
        }
        if (bestTid > 0 && std::fabs(bestDy) >= needDy) {
            strictVerdict = (bestDy < 0.0f) ? QStringLiteral("上车（方向判定）")
                                            : QStringLiteral("下车（方向判定）");
            looseDetail = QString("%1, ID%2 首末 y %3→%4 (dy=%5)")
                              .arg(bestDy < 0.0f ? QStringLiteral("由下往上")
                                                 : QStringLiteral("由上往下"))
                              .arg(bestTid)
                              .arg(m_trackFirstLast.value(bestTid).first, 0, 'f', 3)
                              .arg(m_trackFirstLast.value(bestTid).second, 0, 'f', 3)
                              .arg(bestDy, 0, 'f', 3);
        }
    }
    const QString finalVerdict = strictVerdict.isEmpty()
            ? QStringLiteral("无法判定（方向位移不足或未检测到人体）")
            : strictVerdict;

    // ===== 人物轨迹范围 vs 检测区域范围（辅助判断区域是否覆盖通道）=====
    QString coverage;
    if (m_trajMaxCy >= 0.0f) {
        coverage += QString("人物轨迹范围(归一化)：x[%1~%2] y[%3~%4]\n")
                        .arg(m_trajMinCx, 0, 'f', 3).arg(m_trajMaxCx, 0, 'f', 3)
                        .arg(m_trajMinCy, 0, 'f', 3).arg(m_trajMaxCy, 0, 'f', 3);
    } else {
        coverage += QStringLiteral("人物轨迹范围：未检测到人体\n");
    }

    // 各轨迹(trackId)的 y 范围：若出现多个 ID，说明轨迹被拆断（很可能导致判不出）
    if (!m_trackYRange.isEmpty()) {
        coverage += QString("检测到的轨迹：%1 条\n").arg(m_trackYRange.size());
        int shown = 0;
        for (auto it = m_trackYRange.constBegin();
             it != m_trackYRange.constEnd() && shown < 6; ++it, ++shown) {
            coverage += QString("  ID%1: y[%2~%3]\n")
                            .arg(it.key())
                            .arg(it.value().first, 0, 'f', 3)
                            .arg(it.value().second, 0, 'f', 3);
        }
        if (m_trackYRange.size() > 6) coverage += QStringLiteral("  …\n");
        if (m_trackYRange.size() > 1) {
            coverage += QStringLiteral("提示：轨迹被拆成多个 ID（检测抖动/小框干扰）。"
                                       "已启用最小框尺寸+宽高比过滤；若仍出现，"
                                       "请调大「最小框高度」或降低检测置信度阈值\n");
        }
    }

    if (m_regionManager) {
        const auto regions = m_regionManager->regions();
        if (regions.isEmpty()) {
            coverage += QStringLiteral("上下车分界线：未配置（请先在主界面绘制上下车分界线）\n");
        }
        for (const auto &r : regions) {
            float minX = 2.0f, maxX = -1.0f, minY = 2.0f, maxY = -1.0f;
            for (const QPointF &pt : r.polygon) {
                minX = std::min(minX, static_cast<float>(pt.x()));
                maxX = std::max(maxX, static_cast<float>(pt.x()));
                minY = std::min(minY, static_cast<float>(pt.y()));
                maxY = std::max(maxY, static_cast<float>(pt.y()));
            }
            coverage += QString("检测区域(归一化)：x[%1~%2] y[%3~%4]\n")
                            .arg(minX, 0, 'f', 3).arg(maxX, 0, 'f', 3)
                            .arg(minY, 0, 'f', 3).arg(maxY, 0, 'f', 3);

            if (m_boardCount == 0 && m_alightCount == 0 && m_trajMaxCy >= 0.0f) {
                const float zoneH2 = std::max(0.05f, maxY - minY);
                const float overlapY2 = std::min(m_trajMaxCy, maxY) - std::max(m_trajMinCy, minY);
                const bool overlapX = (m_trajMaxCx >= minX) && (m_trajMinCx <= maxX);
                if (!overlapX || overlapY2 < zoneH2 * 0.2f) {
                    // 区域与人物活动区间几乎不重叠 → 给出可直接采用的区域建议
                    const float sugTop = std::max(0.0f, m_trajMinCy - 0.03f);
                    const float sugBot = std::min(1.0f, m_trajMaxCy + 0.03f);
                    const float sugLeft = std::max(0.0f, m_trajMinCx - 0.05f);
                    const float sugRight = std::min(1.0f, m_trajMaxCx + 0.05f);
                    coverage += QString("⚠ 人物活动区间 y[%1~%2] 与检测区域 y[%3~%4] 几乎不重叠"
                                        "（重叠 %5）→ 建议把区域 y 调整为 [%6~%7]（x 覆盖 [%8~%9]）\n")
                                    .arg(m_trajMinCy, 0, 'f', 3).arg(m_trajMaxCy, 0, 'f', 3)
                                    .arg(minY, 0, 'f', 3).arg(maxY, 0, 'f', 3)
                                    .arg(std::max(0.0f, overlapY2), 0, 'f', 3)
                                    .arg(sugTop, 0, 'f', 3).arg(sugBot, 0, 'f', 3)
                                    .arg(sugLeft, 0, 'f', 3).arg(sugRight, 0, 'f', 3);

                    // 落盘建议区域：把 regions_suggested.json 直接改名为 regions.json 即可生效
                    QJsonArray poly;
                    const QVector<QPointF> pts{QPointF(sugLeft, sugTop), QPointF(sugRight, sugTop),
                                               QPointF(sugRight, sugBot), QPointF(sugLeft, sugBot)};
                    for (const QPointF &qp : pts) {
                        QJsonObject o;
                        o["x"] = qp.x();
                        o["y"] = qp.y();
                        poly.append(o);
                    }
                    QJsonObject reg;
                    reg["id"] = r.id;
                    reg["name"] = QString("区域1_建议");
                    reg["color"] = "#00ff00";
                    reg["entryDirection"] = "TopToBottom";
                    reg["polygon"] = poly;
                    QJsonArray arr;
                    arr.append(reg);

                    const QString sugPath = QDir(QCoreApplication::applicationDirPath())
                                                .filePath("regions_suggested.json");
                    QFile sf(sugPath);
                    if (sf.open(QIODevice::WriteOnly)) {
                        sf.write(QJsonDocument(arr).toJson());
                        sf.close();
                        coverage += QString("（已生成建议区域文件：%1 —— 可直接替换 regions.json 使用）\n")
                                        .arg(sugPath);
                    }
                } else {
                    coverage += QStringLiteral("提示：人物轨迹与区域有重叠但未构成完整穿越 → "
                                               "起点/终点未越出区域上下边界；"
                                               "已按「方向+位移」给出宽松判定\n");
                }
            }
        }
    }

    // 结论置顶，避免用户把「严格完整穿越 0 次」误读为判定失败
    QString summary;
    summary += QString("★ 判定结果：%1\n\n").arg(finalVerdict);
    summary += QString("视频：%1\n结束原因：%2\n已读取帧：%3（其中检测 %4 帧）\n")
                   .arg(QFileInfo(m_videoPath).fileName())
                   .arg(reason)
                   .arg(m_videoFrameIdx).arg(m_videoProcessed);
    summary += QString("严格完整穿越：上车 %1 次，下车 %2 次\n")
                   .arg(m_boardCount).arg(m_alightCount);
    if (!looseDetail.isEmpty())
        summary += QString("方向判定依据：%1\n").arg(looseDetail);
    summary += "\n";
    summary += coverage;

    if (m_labelVideoResult)
        m_labelVideoResult->setText(QString("视频判定：\n★ %1\n(%2)")
                                        .arg(finalVerdict)
                                        .arg(QFileInfo(m_videoPath).fileName()));
    LOG_INFO(QString("[OfflineDetect] ★ 视频判定结论(%1): 判定=%2 | %3")
                 .arg(reason, finalVerdict, summary.simplified()));
    QMessageBox::information(this, "视频判定结果", summary);
}

void OfflineDetectForm::processFrameInternal(const cv::Mat &frame)
{
    if (frame.empty() || !m_detector || !m_detector->isLoaded()) return;

    // ===== 离线检测日志（诊断）：输出“检测当前图片”的完整过程 =====
    const QString imgName = m_currentImagePath.isEmpty()
                                ? QStringLiteral("(内嵌图像)")
                                : QFileInfo(m_currentImagePath).fileName();
    LOG_INFO(QString("[OfflineDetect] ===== 检测当前图片: %1 (%2x%3) | 置信度阈值=%4 =====")
                 .arg(imgName).arg(frame.cols).arg(frame.rows)
                 .arg(m_detector->confThreshold(), 0, 'f', 2));

    // 1) 原始检测（低阈值 0.05、含全部类别）：用于定位“为何检测不到人”
    const QVector<Detection> rawDets = m_detector->detectAll(frame);
    LOG_INFO(QString("[OfflineDetect] 原始检测(score>=0.05，未过滤): %1 个").arg(rawDets.size()));
    for (const Detection &d : rawDets) {
        const cv::Rect box(d.x, d.y, d.w, d.h);
        const bool pass = (d.class_id == 0) && m_detector->passSizeFilter(box, frame.cols, frame.rows);
        const double asp = (d.h > 0) ? static_cast<double>(d.w) / d.h : 0.0;
        LOG_INFO(QString("[OfflineDetect]   raw: class=%1 score=%2 box=(%3,%4,%5x%6) w/h=%7 %8")
                     .arg(d.class_id).arg(d.score, 0, 'f', 3)
                     .arg(d.x).arg(d.y).arg(d.w).arg(d.h)
                     .arg(asp, 0, 'f', 2)
                     .arg(d.class_id == 0
                              ? (pass ? QStringLiteral("→可保留")
                                      : QStringLiteral("→被尺寸/宽高比过滤"))
                              : QStringLiteral("→非 person 类别，忽略")));
    }

    // 2) 过滤后检测（person 类别 + 尺寸/宽高比过滤 + 静态误检抑制）
    QVector<cv::Rect> detections = m_detector->detect(frame);
    LOG_INFO(QString("[OfflineDetect] 过滤后有效人体框: %1 个").arg(detections.size()));
    for (const cv::Rect &d : detections) {
        LOG_INFO(QString("[OfflineDetect]   keep: box=(%1,%2,%3x%4)")
                     .arg(d.x).arg(d.y).arg(d.width).arg(d.height));
    }
    if (detections.isEmpty()) {
        LOG_WARN(QString("[OfflineDetect] 当前图片 0 个有效人体框：可能因置信度低于阈值 / "
                         "类别非 person / 尺寸或宽高比被过滤 / 静态误检抑制"));
    }

    // 统计人物轨迹范围（归一化中心点），用于结论中与检测区域对比
    for (const cv::Rect &d : detections) {
        if (d.width <= 0 || d.height <= 0) continue;
        const float cx = (d.x + d.width * 0.5f) / static_cast<float>(frame.cols);
        const float cy = (d.y + d.height * 0.5f) / static_cast<float>(frame.rows);
        m_trajMinCx = std::min(m_trajMinCx, cx);
        m_trajMaxCx = std::max(m_trajMaxCx, cx);
        m_trajMinCy = std::min(m_trajMinCy, cy);
        m_trajMaxCy = std::max(m_trajMaxCy, cy);
    }

    // 跟踪 + ReID 特征（3参重载会维护 prevCenter 与 reidFeature）
    QVector<TrackedPerson> persons = m_tracker->update(detections, frame, m_reidExtractor);

    // 3) 跟踪轨迹日志：区分真实检测框与卡尔曼预测外推框
    LOG_INFO(QString("[OfflineDetect] 跟踪轨迹: %1 条").arg(persons.size()));
    for (const TrackedPerson &tp : persons) {
        LOG_INFO(QString("[OfflineDetect]   track ID=%1 detected=%2 bbox=(%3,%4,%5x%6) "
                         "age=%7 missed=%8 center=(%9,%10)")
                     .arg(tp.trackId)
                     .arg(tp.detected ? QStringLiteral("true") : QStringLiteral("false"))
                     .arg(tp.bbox.x).arg(tp.bbox.y).arg(tp.bbox.width).arg(tp.bbox.height)
                     .arg(tp.age).arg(tp.missedFrames)
                     .arg(tp.center.x, 0, 'f', 1).arg(tp.center.y, 0, 'f', 1));
    }

    // 统计每条轨迹的 y 范围与首/末 y（只统计真实检测框，忽略卡尔曼预测漂移）
    for (const TrackedPerson &tp : persons) {
        if (tp.trackId <= 0 || frame.rows <= 0) continue;
        if (!tp.detected) continue;
        const float cy = tp.center.y / static_cast<float>(frame.rows);
        auto it = m_trackYRange.find(tp.trackId);
        if (it == m_trackYRange.end()) {
            m_trackYRange.insert(tp.trackId, qMakePair(cy, cy));
            m_trackFirstLast.insert(tp.trackId, qMakePair(cy, cy));
        } else {
            it.value().first = std::min(it.value().first, cy);
            it.value().second = std::max(it.value().second, cy);
            m_trackFirstLast[tp.trackId].second = cy;   // 记录最后一次出现位置
        }
    }

    int imgW = frame.cols;
    int imgH = frame.rows;

    // 通道方向判定（与在线一致：A→B 下车，B→A 上车）
    if (m_zoneMonitor && m_regionManager) {
        refreshZoneMonitor();
        m_zoneMonitor->processFrame(persons, frame, m_regionManager, imgW, imgH);
    }

    // 区域内统计
    int inRegionCount = 0;
    for (auto &p : persons) {
        if (p.isInsideRegion) inRegionCount++;
    }

    // 显示结果
    displayResult(frame, detections, persons);

    m_labelInfo->setText(QString("检测: %1 人, 区域内: %2 | 上车 %3, 下车 %4")
                             .arg(persons.size()).arg(inRegionCount)
                             .arg(m_boardCount).arg(m_alightCount));

    LOG_INFO(QString("[OfflineDetect] Detection done: %1 persons, %2 in region, board=%3, alight=%4")
                 .arg(persons.size()).arg(inRegionCount).arg(m_boardCount).arg(m_alightCount));
}

void OfflineDetectForm::displayResult(const cv::Mat &frame,
                                      const QVector<cv::Rect> &detections,
                                      const QVector<TrackedPerson> &trackedPersons)
{
    Q_UNUSED(detections);
    cv::Mat display = frame.clone();
    int imgW = display.cols;
    int imgH = display.rows;

    // 绘制区域叠加
    drawRegionOverlay(display, imgW, imgH);

    // 绘制检测框和 ID / 方向
    for (const auto &p : trackedPersons) {
        const QString dir = m_trackDirection.value(p.trackId, QString());
        cv::Scalar boxColor;
        if (dir.startsWith("上车"))      boxColor = cv::Scalar(0, 255, 0);   // 绿=上车
        else if (dir.startsWith("下车")) boxColor = cv::Scalar(0, 165, 255); // 橙=下车
        else boxColor = p.isInsideRegion ? cv::Scalar(0, 255, 0) : cv::Scalar(180, 180, 180);
        cv::rectangle(display, p.bbox, boxColor, 2);

        // ID / 方向标签
        std::string label = "ID:" + std::to_string(p.trackId);
        if (!dir.isEmpty()) label += " " + dir.toStdString();
        int x = p.bbox.x;
        int y = p.bbox.y - 10;
        if (y < 20) y = p.bbox.y + 20;

        int baseline = 0;
        cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
        cv::Rect bg(x - 2, y - ts.height - 4, ts.width + 8, ts.height + 8);
        bg &= cv::Rect(0, 0, imgW, imgH);
        if (bg.width > 0 && bg.height > 0) {
            cv::Mat roi = display(bg);
            cv::Mat overlay = roi.clone();
            cv::rectangle(overlay, cv::Point(0, 0), cv::Point(bg.width, bg.height),
                          cv::Scalar(0, 0, 0), -1);
            cv::addWeighted(overlay, 0.6, roi, 0.4, 0, roi);
        }
        cv::putText(display, label, cv::Point(x, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, boxColor, 2);

        if (p.isInsideRegion) {
            cv::putText(display, "IN REGION", cv::Point(x, p.bbox.y + p.bbox.height + 18),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }
    }

    // 显示到 ImageView
    cv::Mat rgb;
    cv::cvtColor(display, rgb, cv::COLOR_BGR2RGB);
    QImage img(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888);
    m_imageView->setImage(img.copy());

    // 更新结果表格（ID / 方向 / 位置 / 区域内）
    m_tableResults->setRowCount(trackedPersons.size());

    for (int i = 0; i < trackedPersons.size(); ++i) {
        const auto &p = trackedPersons[i];

        auto *idItem = new QTableWidgetItem(QString("P%1").arg(p.trackId));
        idItem->setTextAlignment(Qt::AlignCenter);
        m_tableResults->setItem(i, 0, idItem);

        auto *dirItem = new QTableWidgetItem();
        const QString dir = m_trackDirection.value(p.trackId, QString());
        if (dir.isEmpty()) {
            dirItem->setText("--");
            dirItem->setForeground(QBrush(QColor(120, 120, 120)));
        } else {
            dirItem->setText(dir);
            dirItem->setForeground(dir.startsWith("上车")
                                       ? QBrush(QColor(0, 230, 118))
                                       : QBrush(QColor(255, 171, 64)));
        }
        dirItem->setTextAlignment(Qt::AlignCenter);
        m_tableResults->setItem(i, 1, dirItem);

        auto *posItem = new QTableWidgetItem(QString("(%1,%2)").arg(p.bbox.x).arg(p.bbox.y));
        posItem->setTextAlignment(Qt::AlignCenter);
        m_tableResults->setItem(i, 2, posItem);

        auto *regionItem = new QTableWidgetItem();
        if (p.isInsideRegion) {
            regionItem->setText("✓ 是");
            regionItem->setForeground(QBrush(QColor(0, 230, 118)));
        } else {
            regionItem->setText("✗ 否");
            regionItem->setForeground(QBrush(QColor(176, 190, 197)));
        }
        regionItem->setTextAlignment(Qt::AlignCenter);
        m_tableResults->setItem(i, 3, regionItem);
    }
}

void OfflineDetectForm::drawRegionOverlay(cv::Mat &frame, int imgW, int imgH)
{
    if (!m_regionManager) return;

    for (const auto &region : m_regionManager->regions()) {
        if (region.polygon.size() < 3) continue;

        std::vector<cv::Point> polyPoints;
        for (const QPointF &pt : region.polygon) {
            int px = static_cast<int>(pt.x() * imgW);
            int py = static_cast<int>(pt.y() * imgH);
            polyPoints.push_back(cv::Point(px, py));
        }

        // 半透明填充
        cv::Mat overlay = frame.clone();
        cv::fillPoly(overlay, std::vector<std::vector<cv::Point>>{polyPoints},
                     cv::Scalar(255, 100, 0));
        cv::addWeighted(overlay, 0.2, frame, 0.8, 0, frame);

        // 边框
        cv::polylines(frame, std::vector<std::vector<cv::Point>>{polyPoints},
                      true, cv::Scalar(0, 255, 0), 2);

        // 名称 + 方向标注（A=顶/上车方向，B=底）
        if (!region.name.isEmpty()) {
            cv::Point namePos(polyPoints[0].x + 5, polyPoints[0].y + 15);
            cv::putText(frame, region.name.toStdString(), namePos,
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }
    }
}

// ============================================================================
// 上/下车图像相似度比对（判定是否同一乘客）
// ============================================================================

cv::Mat OfflineDetectForm::imageFeature(const cv::Mat &img)
{
    if (img.empty()) return cv::Mat();

    // 优先用检测到的人体框（取面积最大者）；检测不到则退回整图
    cv::Rect box;
    if (m_detector && m_detector->isLoaded()) {
        const QVector<cv::Rect> dets = m_detector->detect(img);
        int best = -1;
        double bestArea = 0.0;
        for (int i = 0; i < dets.size(); ++i) {
            const double a = static_cast<double>(dets[i].area());
            if (a > bestArea) { bestArea = a; best = i; }
        }
        if (best >= 0) box = dets[best];
    }
    if (box.empty()) box = cv::Rect(0, 0, img.cols, img.rows);

    // 深度 ReID（OSNet）：输出 512 维嵌入
    if (m_useDeepReid && m_deepReid.ready()) {
        return m_deepReid.embed(img, box);
    }

    if (!m_reidExtractor) return cv::Mat();
    return m_reidExtractor->extract(img, box);
}

cv::Mat OfflineDetectForm::meanFeature(const QVector<cv::Mat> &feats)
{
    cv::Mat sum;
    int n = 0;
    for (const cv::Mat &f : feats) {
        if (f.empty()) continue;
        const cv::Mat v = f.reshape(1, 1);
        if (sum.empty()) sum = cv::Mat::zeros(1, v.cols, CV_32F);
        if (v.cols != sum.cols) continue;   // 尺寸不一致跳过
        sum += v;
        n++;
    }
    if (n == 0) return cv::Mat();
    sum /= static_cast<float>(n);
    const double norm = cv::norm(sum, cv::NORM_L2);
    if (norm > 1e-12) sum /= static_cast<float>(norm);
    return sum;
}

float OfflineDetectForm::groupSimilarity(const QVector<cv::Mat> &a, const QVector<cv::Mat> &b) const
{
    if (a.isEmpty() || b.isEmpty()) return 0.0f;

    // ===== 深度 ReID 模式：组质心 + 余弦相似度 =====
    // 深度嵌入为语义向量，「质心」比「逐帧取最优再平均」更稳健、更少被离群帧抬高。
    if (m_useDeepReid && m_deepReid.ready()) {
        cv::Mat ca = meanFeature(a);
        cv::Mat cb = meanFeature(b);
        return m_deepReid.compare(ca, cb);
    }

    // ===== 手工特征模式（保持原逻辑）=====
    if (!m_reidExtractor) return 0.0f;
    float sum = 0.0f;
    int counted = 0;
    for (const cv::Mat &fa : a) {
        if (fa.empty()) continue;
        float best = 0.0f;
        for (const cv::Mat &fb : b) {
            if (fb.empty()) continue;
            const float s = m_reidExtractor->compare(fa, fb); // 内部已含 180° 旋转不变
            if (s > best) best = s;
        }
        sum += best;
        counted++;
    }
    return (counted > 0) ? (sum / static_cast<float>(counted)) : 0.0f;
}

void OfflineDetectForm::onDeepReidToggled(bool on)
{
    m_useDeepReid = on && m_deepReid.ready();
    // 特征后端改变 → 已提取的特征失效，清空并提示重新加载
    m_boardFeats.clear();
    m_alightFeats.clear();
    if (m_labelBoardCount)  m_labelBoardCount->setText("上车: 0 张");
    if (m_labelAlightCount) m_labelAlightCount->setText("下车: 0 张");
    if (m_labelCompare) {
        m_labelCompare->setText(on
            ? "已切换到【深度ReID(OSNet)】后端，请重新加载上车/下车图像后再比对。"
            : "已切换到【手工特征】后端，请重新加载上车/下车图像后再比对。");
    }
    LOG_INFO(QString("[OfflineDetect] 特征后端切换为: %1")
                 .arg(m_useDeepReid ? "深度ReID(OSNet)" : "手工特征(颜色+纹理+身形)"));
}

void OfflineDetectForm::onLoadBoardingImages()
{
    const QStringList files = QFileDialog::getOpenFileNames(
        this, "选择“上车”图像（可多选）", QString(),
        "图像文件 (*.jpg *.jpeg *.png *.bmp *.tiff);;所有文件 (*)");
    if (files.isEmpty()) return;

    m_boardPaths = files;
    m_boardFeats.clear();
    for (const QString &p : m_boardPaths) {
        cv::Mat img = cv::imread(p.toStdString());
        if (img.empty()) continue;
        cv::Mat f = imageFeature(img);
        if (!f.empty()) m_boardFeats.append(f);
    }
    // 默认显示第一张
    m_boardPreviewIndex = 0;
    showBoardPreview();
    if (m_labelBoardCount)
        m_labelBoardCount->setText(QString("上车: %1 张").arg(m_boardFeats.size()));
    LOG_INFO(QString("[OfflineDetect] 加载上车图像 %1 张(有效特征 %2)")
                 .arg(m_boardPaths.size()).arg(m_boardFeats.size()));
}

void OfflineDetectForm::onLoadAlightingImages()
{
    const QStringList files = QFileDialog::getOpenFileNames(
        this, "选择“下车”图像（可多选）", QString(),
        "图像文件 (*.jpg *.jpeg *.png *.bmp *.tiff);;所有文件 (*)");
    if (files.isEmpty()) return;

    m_alightPaths = files;
    m_alightFeats.clear();
    for (const QString &p : m_alightPaths) {
        cv::Mat img = cv::imread(p.toStdString());
        if (img.empty()) continue;
        cv::Mat f = imageFeature(img);
        if (!f.empty()) m_alightFeats.append(f);
    }
    // 默认显示第一张
    m_alightPreviewIndex = 0;
    showAlightPreview();
    if (m_labelAlightCount)
        m_labelAlightCount->setText(QString("下车: %1 张").arg(m_alightFeats.size()));
    LOG_INFO(QString("[OfflineDetect] 加载下车图像 %1 张(有效特征 %2)")
                 .arg(m_alightPaths.size()).arg(m_alightFeats.size()));
}

void OfflineDetectForm::onCompareGroups()
{
    if (m_boardFeats.isEmpty() || m_alightFeats.isEmpty()) {
        QMessageBox::warning(this, "无法比较",
            "请先分别加载“上车图像”和“下车图像”（可各选 1 张或多张）。");
        return;
    }

    const float thr = matchThreshold();
    const float sim = groupSimilarity(m_boardFeats, m_alightFeats);
    const bool same = (sim >= thr);
    const QString verdict = same ? QStringLiteral("✅ 同一乘客") : QStringLiteral("❌ 不同乘客");
    const QString backend = (m_useDeepReid && m_deepReid.ready())
                                ? QStringLiteral("深度ReID(OSNet)+质心")
                                : QStringLiteral("手工特征(颜色/纹理/身形)");

    if (m_labelCompare) {
        m_labelCompare->setText(QString(
            "特征后端: %1\n"
            "上车组: %2 张   下车组: %3 张\n"
            "相似度: %4   阈值: %5\n"
            "判定: %6")
            .arg(backend)
            .arg(m_boardFeats.size()).arg(m_alightFeats.size())
            .arg(sim, 0, 'f', 3).arg(thr, 0, 'f', 2).arg(verdict));
        m_labelCompare->setStyleSheet(QString(
            "color:%1;font-family:'Consolas';font-size:13px;font-weight:bold;"
            "background:rgba(0,0,0,0.25);border:1px solid #1a3058;border-radius:4px;padding:6px;")
            .arg(same ? "#00e676" : "#ef5350"));
    }

    LOG_INFO(QString("[OfflineDetect] 相似度比对: 上车%1张 vs 下车%2张 -> sim=%3 -> %4")
                 .arg(m_boardFeats.size()).arg(m_alightFeats.size())
                 .arg(sim, 0, 'f', 3).arg(verdict));
}

// ============================================================================
// 上车 / 下车 图像预览
// ============================================================================

void OfflineDetectForm::showImageFile(ImageView *view, const QString &path)
{
    if (!view || path.isEmpty()) return;
    cv::Mat img = cv::imread(path.toStdString());
    if (img.empty()) return;
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    QImage qi(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888);
    view->setImage(qi.copy());
}

void OfflineDetectForm::showBoardPreview()
{
    if (m_boardPaths.isEmpty()) {
        if (m_labelBoardImage) m_labelBoardImage->setText(QStringLiteral("(未加载)"));
        return;
    }
    m_boardPreviewIndex = qBound(0, m_boardPreviewIndex, m_boardPaths.size() - 1);
    const QString &p = m_boardPaths.at(m_boardPreviewIndex);
    showImageFile(m_boardView, p);
    if (m_labelBoardImage)
        m_labelBoardImage->setText(QString("%1 (%2/%3)")
            .arg(QFileInfo(p).fileName())
            .arg(m_boardPreviewIndex + 1).arg(m_boardPaths.size()));
}

void OfflineDetectForm::showAlightPreview()
{
    if (m_alightPaths.isEmpty()) {
        if (m_labelAlightImage) m_labelAlightImage->setText(QStringLiteral("(未加载)"));
        return;
    }
    m_alightPreviewIndex = qBound(0, m_alightPreviewIndex, m_alightPaths.size() - 1);
    const QString &p = m_alightPaths.at(m_alightPreviewIndex);
    showImageFile(m_alightView, p);
    if (m_labelAlightImage)
        m_labelAlightImage->setText(QString("%1 (%2/%3)")
            .arg(QFileInfo(p).fileName())
            .arg(m_alightPreviewIndex + 1).arg(m_alightPaths.size()));
}

void OfflineDetectForm::onBoardPrev()
{
    if (m_boardPaths.isEmpty()) return;
    m_boardPreviewIndex = (m_boardPreviewIndex - 1 + m_boardPaths.size()) % m_boardPaths.size();
    showBoardPreview();
}

void OfflineDetectForm::onBoardNext()
{
    if (m_boardPaths.isEmpty()) return;
    m_boardPreviewIndex = (m_boardPreviewIndex + 1) % m_boardPaths.size();
    showBoardPreview();
}

void OfflineDetectForm::onAlightPrev()
{
    if (m_alightPaths.isEmpty()) return;
    m_alightPreviewIndex = (m_alightPreviewIndex - 1 + m_alightPaths.size()) % m_alightPaths.size();
    showAlightPreview();
}

void OfflineDetectForm::onAlightNext()
{
    if (m_alightPaths.isEmpty()) return;
    m_alightPreviewIndex = (m_alightPreviewIndex + 1) % m_alightPaths.size();
    showAlightPreview();
}
