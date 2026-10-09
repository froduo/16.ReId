#include "offlinedetectform.h"
#include "logger.h"
#include <QMessageBox>
#include <QHeaderView>
#include <QFileInfo>
#include <QMap>
#include <QStringList>
#include <algorithm>

OfflineDetectForm::OfflineDetectForm(PersonDetector *detector,
                                     RegionManager *regionManager,
                                     QWidget *parent)
    : QDialog(parent)
    , m_detector(detector)
    , m_regionManager(regionManager)
{
    setupUI();
    setWindowTitle("离线图像检测");
    resize(1200, 700);

    // 创建独立的 tracker 和 reid extractor 用于离线检测
    m_tracker = new PersonTracker(0.3f, 30);
    m_reidExtractor = new ReIDExtractor();
}

OfflineDetectForm::~OfflineDetectForm()
{
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
    m_btnLoad = new QPushButton("📂 加载图像");
    m_btnLoad->setMinimumSize(120, 32);
    m_btnDetect = new QPushButton("🔍 开始检测");
    m_btnDetect->setMinimumSize(120, 32);
    m_btnDetect->setEnabled(false);
    m_labelInfo = new QLabel("请加载一张图像");
    m_labelInfo->setStyleSheet("color: #78ceff; font-family: 'Consolas'; font-size: 12px;");

    btnLayout->addWidget(m_btnLoad);
    btnLayout->addWidget(m_btnDetect);
    btnLayout->addWidget(m_labelInfo);
    btnLayout->addStretch();

    leftLayout->addLayout(btnLayout);

    // ImageView
    m_imageView = new ImageView(this);
    m_imageView->setMinimumSize(640, 480);
    leftLayout->addWidget(m_imageView, 1);

    mainLayout->addLayout(leftLayout, 3);

    // 右侧：检测结果面板
    auto *rightPanel = new QFrame();
    rightPanel->setMinimumWidth(260);
    rightPanel->setMaximumWidth(320);
    rightPanel->setStyleSheet("QFrame { background-color: rgba(6, 14, 26, 0.96); border: 1px solid #1a3860; border-radius: 8px; }");

    auto *rightLayout = new QVBoxLayout(rightPanel);
    rightLayout->setSpacing(6);
    rightLayout->setContentsMargins(6, 6, 6, 6);

    auto *titleLabel = new QLabel("◆ 检测结果");
    titleLabel->setStyleSheet("color: #00e5ff; font-family: 'Consolas'; font-size: 13px; font-weight: bold;");
    rightLayout->addWidget(titleLabel);

    m_tableResults = new QTableWidget();
    m_tableResults->setColumnCount(3);
    m_tableResults->setHorizontalHeaderLabels({"ID", "位置", "区域内"});
    m_tableResults->horizontalHeader()->setStretchLastSection(true);
    m_tableResults->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_tableResults->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_tableResults->verticalHeader()->setVisible(false);
    m_tableResults->verticalHeader()->setDefaultSectionSize(22);
    m_tableResults->setColumnWidth(0, 40);
    m_tableResults->setColumnWidth(1, 100);
    m_tableResults->setStyleSheet(
        "QTableWidget { background-color: #080f1e; border: 1px solid #1a3058; border-radius: 4px; "
        "color: #b8d4f0; font-family: 'Consolas'; font-size: 11px; gridline-color: #122040; }"
        "QTableWidget::item { padding: 2px 4px; }"
        "QHeaderView::section { background-color: #0a1a30; color: #78ceff; font-weight: bold; "
        "border: 1px solid #1a3058; padding: 3px 4px; }");

    rightLayout->addWidget(m_tableResults, 1);

    mainLayout->addWidget(rightPanel, 1);

    // 连接信号
    connect(m_btnLoad, &QPushButton::clicked, this, &OfflineDetectForm::onLoadImage);
    connect(m_btnDetect, &QPushButton::clicked, this, &OfflineDetectForm::onRunDetect);
}

void OfflineDetectForm::onLoadImage()
{
    QString filePath = QFileDialog::getOpenFileName(
        this, "选择图像文件", QString(),
        "图像文件 (*.jpg *.jpeg *.png *.bmp *.tiff);;所有文件 (*)");

    if (filePath.isEmpty()) return;

    m_currentImage = cv::imread(filePath.toStdString());
    if (m_currentImage.empty()) {
        QMessageBox::warning(this, "错误", "无法加载图像: " + filePath);
        return;
    }

    m_currentImagePath = filePath;

    // 显示图像
    cv::Mat rgb;
    cv::cvtColor(m_currentImage, rgb, cv::COLOR_BGR2RGB);
    QImage img(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888);
    m_imageView->setImage(img.copy());

    // 设置区域多边形到 ImageView
    if (m_regionManager) {
        QVector<QVector<QPointF>> polys;
        for (const auto &r : m_regionManager->regions()) {
            polys.append(r.polygon);
        }
        m_imageView->setRegionPolygons(polys);
    }

    m_btnDetect->setEnabled(true);
    m_labelInfo->setText(QString("已加载: %1 (%2x%3)")
                             .arg(QFileInfo(filePath).fileName())
                             .arg(m_currentImage.cols).arg(m_currentImage.rows));

    // 清空结果表格
    m_tableResults->setRowCount(0);

    LOG_INFO("[OfflineDetect] Image loaded: " + filePath);
}

void OfflineDetectForm::onRunDetect()
{
    if (m_currentImage.empty() || !m_detector || !m_detector->isLoaded()) {
        QMessageBox::warning(this, "错误", "未加载图像或检测模型未就绪");
        return;
    }

    // 运行检测（仅 person 类别）
    QVector<cv::Rect> detections = m_detector->detect(m_currentImage);

    // 运行跟踪（分配 ID）
    QVector<TrackedPerson> persons = m_tracker->update(detections);

    // 提取 ReID 特征
    if (m_reidExtractor) {
        for (auto &p : persons) {
            p.reidFeature = m_reidExtractor->extract(m_currentImage, p.bbox);
        }
    }

    // 检查区域内状态
    int imgW = m_currentImage.cols;
    int imgH = m_currentImage.rows;
    int inRegionCount = 0;

    for (auto &p : persons) {
        float normCx = p.center.x / imgW;
        float normCy = p.center.y / imgH;
        QPointF normCenter(normCx, normCy);

        bool inRegion = m_regionManager && m_regionManager->isPointInAnyRegion(normCenter);
        p.isInsideRegion = inRegion;
        if (inRegion) inRegionCount++;
    }

    // 显示结果
    displayResult(m_currentImage, detections, persons);

    // ===== 当检测到 0 人时，运行低阈值诊断 =====
    if (persons.isEmpty()) {
        QVector<Detection> allDets = m_detector->detectAll(m_currentImage);

        LOG_INFO(QString("[OfflineDetect] DIAGNOSTIC: person=0, all-class detections (threshold=0.05): %1")
                     .arg(allDets.size()));

        // 按置信度排序
        std::sort(allDets.begin(), allDets.end(), [](const Detection &a, const Detection &b) {
            return a.score > b.score;
        });

        // 统计各类别数量
        QMap<int, int> classCounts;
        QMap<int, float> classBestScore;
        for (const auto &d : allDets) {
            classCounts[d.class_id] = classCounts.value(d.class_id, 0) + 1;
            float best = classBestScore.value(d.class_id, 0.0f);
            if (d.score > best) classBestScore[d.class_id] = d.score;
        }

        // 打印前 20 个检测结果
        for (int i = 0; i < allDets.size() && i < 20; i++) {
            const auto &d = allDets[i];
            LOG_INFO(QString("  allDet[%1]: class=%2 score=%3 box=(%4,%5,%6x%7)")
                         .arg(i).arg(d.class_id).arg(d.score, 0, 'f', 3)
                         .arg(d.x).arg(d.y).arg(d.w).arg(d.h));
        }

        // 构建诊断信息
        QString diagMsg = QString("⚠️ 未检测到人 (person class=0)\n\n"
                                  "图像尺寸: %1x%2\n"
                                  "低阈值(0.05)全部检测数: %3\n\n")
                              .arg(imgW).arg(imgH).arg(allDets.size());

        if (!classCounts.isEmpty()) {
            diagMsg += "各类别检测统计:\n";
            for (auto it = classCounts.constBegin(); it != classCounts.constEnd(); ++it) {
                int clsId = it.key();
                QString clsName;
                // COCO 类别名称（常用）
                static const QStringList cocoNames = {
                    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
                    "traffic light","fire hydrant","","stop sign","parking meter","bench","bird",
                    "cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","",
                    "backpack","umbrella","","","handbag","tie","suitcase","frisbee","skis",
                    "snowboard","sports ball","kite","baseball bat","baseball glove","skateboard",
                    "surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon",
                    "bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
                    "donut","cake","chair","couch","potted plant","bed","dining table","",
                    "toilet","","tv","laptop","mouse","remote","keyboard","cell phone","microwave",
                    "oven","toaster","sink","refrigerator","","book","clock","vase","scissors",
                    "teddy bear","hair drier","toothbrush"
                };
                if (clsId >= 0 && clsId < cocoNames.size() && !cocoNames[clsId].isEmpty()) {
                    clsName = cocoNames[clsId];
                } else {
                    clsName = QString("class_%1").arg(clsId);
                }
                diagMsg += QString("  %1 (id=%2): %3 个, 最高分=%4\n")
                               .arg(clsName).arg(clsId).arg(it.value())
                               .arg(classBestScore.value(clsId, 0), 0, 'f', 3);
            }
        } else {
            diagMsg += "模型未产生任何检测结果。\n\n可能原因:\n"
                       "• 置信度阈值过高\n"
                       "• 图像格式/通道数不匹配\n"
                       "• 俯视角度与训练数据差异大";
        }

        diagMsg += "\n\n💡 建议: 垂直俯视摄像头下，标准COCO模型"
                   "可能无法识别头顶视角的人体。\n"
                   "可尝试通过「系统配置」降低置信度阈值。";

        QMessageBox::information(this, "检测诊断信息", diagMsg);
    }

    m_labelInfo->setText(QString("检测完成: %1 人, 区域内: %2 人")
                             .arg(persons.size()).arg(inRegionCount));

    LOG_INFO(QString("[OfflineDetect] Detection done: %1 persons, %2 in region")
                 .arg(persons.size()).arg(inRegionCount));
}

void OfflineDetectForm::displayResult(const cv::Mat &frame,
                                       const QVector<cv::Rect> &detections,
                                       const QVector<TrackedPerson> &trackedPersons)
{
    cv::Mat display = frame.clone();
    int imgW = display.cols;
    int imgH = display.rows;

    // 绘制区域叠加
    drawRegionOverlay(display, imgW, imgH);

    // 绘制检测框和 ID
    for (const auto &p : trackedPersons) {
        cv::Scalar boxColor = p.isInsideRegion ? cv::Scalar(0, 255, 0) : cv::Scalar(180, 180, 180);
        cv::rectangle(display, p.bbox, boxColor, 2);

        // ID 标签
        std::string label = "ID:" + std::to_string(p.trackId);
        int x = p.bbox.x;
        int y = p.bbox.y - 10;
        if (y < 20) y = p.bbox.y + 20;

        // 背景
        int baseline = 0;
        cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
        cv::Rect bg(x - 2, y - ts.height - 4, ts.width + 8, ts.height + 8);
        bg &= cv::Rect(0, 0, imgW, imgH);
        // 防御：越界/退化框会产生空 ROI，需跳过背景混合
        if (bg.width > 0 && bg.height > 0) {
            cv::Mat roi = display(bg);
            if (!roi.empty()) {
                cv::Mat overlay = roi.clone();
                cv::rectangle(overlay, cv::Point(0, 0), cv::Point(roi.cols, roi.rows), cv::Scalar(0, 0, 0), -1);
                cv::addWeighted(overlay, 0.6, roi, 0.4, 0, roi);
            }
        }

        cv::Scalar textColor = p.isInsideRegion ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 255, 255);
        cv::putText(display, label, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.7, textColor, 2);

        // 区域内标记
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

    // 更新结果表格
    m_tableResults->setRowCount(trackedPersons.size());
    for (int i = 0; i < trackedPersons.size(); ++i) {
        const auto &p = trackedPersons[i];

        auto *idItem = new QTableWidgetItem(QString("P%1").arg(p.trackId));
        idItem->setTextAlignment(Qt::AlignCenter);
        m_tableResults->setItem(i, 0, idItem);

        auto *posItem = new QTableWidgetItem(QString("(%1,%2)").arg(p.bbox.x).arg(p.bbox.y));
        posItem->setTextAlignment(Qt::AlignCenter);
        m_tableResults->setItem(i, 1, posItem);

        auto *regionItem = new QTableWidgetItem();
        if (p.isInsideRegion) {
            regionItem->setText("✓ 是");
            regionItem->setForeground(QBrush(QColor(0, 230, 118)));
        } else {
            regionItem->setText("✗ 否");
            regionItem->setForeground(QBrush(QColor(176, 190, 197)));
        }
        regionItem->setTextAlignment(Qt::AlignCenter);
        m_tableResults->setItem(i, 2, regionItem);
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

        // 名称
        if (!region.name.isEmpty()) {
            cv::Point namePos(polyPoints[0].x + 5, polyPoints[0].y + 15);
            cv::putText(frame, region.name.toStdString(), namePos,
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }
    }
}
