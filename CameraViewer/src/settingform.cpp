#include "settingform.h"
#include "ui_settingform.h"
#include "logger.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QFileInfo>
#include <QDir>
#include <QCoreApplication>
#include <QApplication>
#include <QHeaderView>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QRegularExpression>
#include <opencv2/opencv.hpp>

SettingForm::SettingForm(const AppConfig &config, QWidget *parent)
    : QDialog(parent)
    , ui(new Ui::SettingForm)
    , m_config(config)
{
    ui->setupUi(this);

    // 加载样式表
    QString stylePath = QCoreApplication::applicationDirPath() + "/style.qss";
    QFile styleFile(stylePath);
    if (!styleFile.exists()) {
        stylePath = QCoreApplication::applicationDirPath() + "/../style.qss";
        styleFile.setFileName(stylePath);
    }
    if (styleFile.open(QFile::ReadOnly)) {
        setStyleSheet(QLatin1String(styleFile.readAll()));
        styleFile.close();
    }

    // ===== 相机设置 =====
    ui->edit_cameraIP->setText(m_config.cameraIP);
    ui->edit_cameraUser->setText(m_config.cameraUser);
    ui->edit_cameraPass->setText(m_config.cameraPass);
    ui->spin_cameraPort->setValue(m_config.cameraPort);
    ui->edit_rtspUrl->setText(m_config.rtspUrl);

    // ===== 推理设置 =====
    ui->check_enableInference->setChecked(m_config.enableInference);
    ui->edit_modelPath->setText(m_config.modelPath);
    ui->spin_confThreshold->setValue(m_config.detectorConfThreshold);
    ui->spin_nmsThreshold->setValue(m_config.detectorNmsThreshold);

    // 跟踪参数
    ui->spin_iouThreshold->setValue(m_config.trackerIouThreshold);
    ui->spin_maxMissed->setValue(m_config.trackerMaxMissed);
    ui->spin_reidThreshold->setValue(m_config.reidMatchThreshold);
    ui->check_kalmanEnabled->setChecked(m_config.trackerKalmanEnabled);
    ui->check_trackerReidEnabled->setChecked(m_config.trackerReidEnabled);
    ui->spin_trackerReidThreshold->setValue(m_config.trackerReidThreshold);
    ui->spin_galleryMaxAge->setValue(m_config.trackerGalleryMaxAgeMs);

    // 相似度融合权重（颜色 / 纹理 / 身形）
    ui->spin_reidWeightColor->setValue(m_config.reidWeightColor);
    ui->spin_reidWeightTexture->setValue(m_config.reidWeightTexture);
    ui->spin_reidWeightShape->setValue(m_config.reidWeightShape);

    // 深度 ReID（OSNet）
    ui->check_reidUseDeep->setChecked(m_config.reidUseDeepReid);
    ui->edit_reidDeepModelPath->setText(m_config.reidDeepModelPath);
    ui->spin_reidDeepThreshold->setValue(m_config.reidDeepMatchThreshold);

    // ===== 保存设置 =====
    ui->edit_savePath->setText(m_config.savePath);
    ui->combo_saveFormat->setCurrentText(m_config.saveFormat);
    ui->spin_jpegQuality->setValue(m_config.jpegQuality);
    ui->edit_reidSavePath->setText(m_config.reidSavePath);
    ui->edit_regionSavePath->setText(m_config.regionSavePath);

    // 信号槽连接
    connect(ui->buttonBox, &QDialogButtonBox::accepted, this, &SettingForm::onAccept);
    connect(ui->btn_browseModel, &QPushButton::clicked, this, &SettingForm::onBrowseModel);
    connect(ui->btn_browseSavePath, &QPushButton::clicked, this, &SettingForm::onBrowseSavePath);
    connect(ui->btn_browseReidSavePath, &QPushButton::clicked, this, &SettingForm::onBrowseReidSavePath);
    connect(ui->btn_rebuildUrl, &QPushButton::clicked, this, &SettingForm::onRebuildRtspUrl);

    // ===== 相机搜索 =====
    m_searcher = new CameraSearcher(this);
    connect(ui->btn_searchCameras, &QPushButton::clicked, this, &SettingForm::onSearchCameras);
    connect(ui->btn_applyDevice, &QPushButton::clicked, this, &SettingForm::onApplyDevice);
    connect(ui->btn_verifyDevice, &QPushButton::clicked, this, &SettingForm::onVerifyDevice);
    connect(ui->table_devices, &QTableWidget::cellClicked, this, &SettingForm::onDeviceRowSelected);
    connect(ui->table_devices, &QTableWidget::cellDoubleClicked, this, &SettingForm::onApplyDevice);
    connect(m_searcher, &CameraSearcher::deviceFound, this, &SettingForm::onDeviceFound);
    connect(m_searcher, &CameraSearcher::searchProgress, this, &SettingForm::onSearchProgress);
    connect(m_searcher, &CameraSearcher::searchFinished, this, &SettingForm::onSearchFinished);

    // 初始化设备表格列宽
    QHeaderView *header = ui->table_devices->horizontalHeader();
    header->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    header->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    header->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    header->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    header->setSectionResizeMode(4, QHeaderView::ResizeToContents);
    header->setSectionResizeMode(5, QHeaderView::ResizeToContents);
    header->setSectionResizeMode(6, QHeaderView::ResizeToContents);
    header->setSectionResizeMode(7, QHeaderView::Stretch);

    LOG_INFO("[SettingForm] Settings dialog opened");
}

SettingForm::~SettingForm()
{
    if (m_searcher) {
        m_searcher->cancel();
    }
    delete ui;
}

AppConfig SettingForm::getConfig() const
{
    return m_config;
}

void SettingForm::onAccept()
{
    // 相机设置
    m_config.cameraIP = ui->edit_cameraIP->text().trimmed();
    m_config.cameraUser = ui->edit_cameraUser->text().trimmed();
    m_config.cameraPass = ui->edit_cameraPass->text().trimmed();
    m_config.cameraPort = ui->spin_cameraPort->value();
    m_config.rtspUrl = ui->edit_rtspUrl->text().trimmed();

    // 如果 RTSP URL 为空，自动生成
    if (m_config.rtspUrl.isEmpty()) {
        m_config.rebuildRtspUrl();
    }

    // 推理设置
    m_config.enableInference = ui->check_enableInference->isChecked();
    m_config.modelPath = ui->edit_modelPath->text().trimmed();
    m_config.detectorConfThreshold = static_cast<float>(ui->spin_confThreshold->value());
    m_config.detectorNmsThreshold = static_cast<float>(ui->spin_nmsThreshold->value());

    // 跟踪参数
    m_config.trackerIouThreshold = static_cast<float>(ui->spin_iouThreshold->value());
    m_config.trackerMaxMissed = ui->spin_maxMissed->value();
    m_config.reidMatchThreshold = static_cast<float>(ui->spin_reidThreshold->value());
    m_config.trackerKalmanEnabled = ui->check_kalmanEnabled->isChecked();
    m_config.trackerReidEnabled = ui->check_trackerReidEnabled->isChecked();
    m_config.trackerReidThreshold = static_cast<float>(ui->spin_trackerReidThreshold->value());
    m_config.trackerGalleryMaxAgeMs = ui->spin_galleryMaxAge->value();

    // 相似度融合权重（颜色 / 纹理 / 身形）—— 运行时应用到 ReID 提取器
    m_config.reidWeightColor = static_cast<float>(ui->spin_reidWeightColor->value());
    m_config.reidWeightTexture = static_cast<float>(ui->spin_reidWeightTexture->value());
    m_config.reidWeightShape = static_cast<float>(ui->spin_reidWeightShape->value());

    // 深度 ReID（OSNet）
    m_config.reidUseDeepReid = ui->check_reidUseDeep->isChecked();
    m_config.reidDeepModelPath = ui->edit_reidDeepModelPath->text().trimmed();
    m_config.reidDeepMatchThreshold = static_cast<float>(ui->spin_reidDeepThreshold->value());

    // 保存设置
    m_config.savePath = ui->edit_savePath->text().trimmed();
    m_config.saveFormat = ui->combo_saveFormat->currentText();
    m_config.jpegQuality = ui->spin_jpegQuality->value();
    m_config.reidSavePath = ui->edit_reidSavePath->text().trimmed();
    m_config.regionSavePath = ui->edit_regionSavePath->text().trimmed();

    // 验证模型路径
    if (m_config.enableInference) {
        QString modelPath = m_config.modelPath;
        if (!QFileInfo(modelPath).isAbsolute()) {
            modelPath = QCoreApplication::applicationDirPath() + "/" + modelPath;
        }
        if (!QFileInfo::exists(modelPath)) {
            QMessageBox::warning(this, "警告",
                QString("模型文件不存在:\n%1\n\n请检查模型路径，或取消勾选「启用推理」。")
                .arg(modelPath));
            return; // 不关闭对话框，让用户修正
        }
    }

    LOG_INFO("[SettingForm] Settings accepted");
    accept();
}

void SettingForm::onBrowseModel()
{
    QString dir = ui->edit_modelPath->text().isEmpty()
        ? QCoreApplication::applicationDirPath() + "/model"
        : QFileInfo(ui->edit_modelPath->text()).absolutePath();

    QString filePath = QFileDialog::getOpenFileName(this, "选择推理模型文件", dir,
        "TensorRT Engine (*.engine);;ONNX Model (*.onnx);;All Files (*)");
    if (!filePath.isEmpty()) {
        ui->edit_modelPath->setText(filePath);
    }
}

void SettingForm::onBrowseSavePath()
{
    QString dir = ui->edit_savePath->text().isEmpty()
        ? QCoreApplication::applicationDirPath()
        : ui->edit_savePath->text();
    dir = QFileDialog::getExistingDirectory(this, "选择图像保存目录", dir);
    if (!dir.isEmpty()) {
        ui->edit_savePath->setText(dir);
    }
}

void SettingForm::onBrowseReidSavePath()
{
    QString dir = ui->edit_reidSavePath->text().isEmpty()
        ? QCoreApplication::applicationDirPath() + "/save_images/reid"
        : ui->edit_reidSavePath->text();
    dir = QFileDialog::getExistingDirectory(this, "选择 ReID 图像保存目录", dir);
    if (!dir.isEmpty()) {
        ui->edit_reidSavePath->setText(dir);
    }
}

void SettingForm::onRebuildRtspUrl()
{
    m_config.cameraIP = ui->edit_cameraIP->text().trimmed();
    m_config.cameraUser = ui->edit_cameraUser->text().trimmed();
    m_config.cameraPass = ui->edit_cameraPass->text().trimmed();
    m_config.cameraPort = ui->spin_cameraPort->value();
    m_config.rebuildRtspUrl();
    ui->edit_rtspUrl->setText(m_config.rtspUrl);

    LOG_INFO("[SettingForm] RTSP URL rebuilt: " + m_config.rtspUrl);
}

// ===== 相机搜索 =====
void SettingForm::onSearchCameras()
{
    if (!m_searcher || m_searcher->isSearching()) {
        return;
    }

    m_devices.clear();
    ui->table_devices->setRowCount(0);

    ui->btn_searchCameras->setEnabled(false);
    ui->btn_searchCameras->setText("搜索中…");
    ui->label_searchStatus->setText("正在扫描局域网设备…");

    // 用当前「连接参数」的账号密码作为首选验证凭据
    const QString user = ui->edit_cameraUser->text().trimmed();
    const QString pass = ui->edit_cameraPass->text().trimmed();

    // 需要探测的视频端口（默认 554/8554，并包含当前 RTSP 端口）
    QVector<int> videoPorts;
    const int currentPort = ui->spin_cameraPort->value();
    for (int p : {554, 8554, currentPort}) {
        if (!videoPorts.contains(p)) {
            videoPorts.append(p);
        }
    }

    // 常见 RTSP 路径（按优先级尝试）
    const QStringList rtspPaths = {
        QStringLiteral("/live/camera"),
        QStringLiteral("/live/main"),
        QStringLiteral("/live/sub"),
        QStringLiteral("/stream1"),
        QStringLiteral("/h264"),
        QStringLiteral("/cam/realmonitor?channel=1&subtype=0"),
        QStringLiteral("/11"),
        QStringLiteral("/12"),
        QStringLiteral("/main"),
        QStringLiteral("/sub"),
    };

    // 网段留空 => 自动探测本机所有网段；ACP 控制端口 8091
    m_searcher->startSearch(QString(), videoPorts, user, pass, rtspPaths, 8091, 3000);
}

void SettingForm::onDeviceFound(const CameraSearcher::Device &device)
{
    const int row = m_devices.size();
    m_devices.append(device);

    ui->table_devices->insertRow(row);

    auto setItem = [row](QTableWidget *table, int col, const QString &text) {
        QTableWidgetItem *item = new QTableWidgetItem(text);
        item->setFlags(item->flags() & ~Qt::ItemIsEditable);
        table->setItem(row, col, item);
    };

    setItem(ui->table_devices, 0, device.displayName());
    setItem(ui->table_devices, 1, device.ip);
    setItem(ui->table_devices, 2, device.deviceType);
    setItem(ui->table_devices, 3, device.protocol);
    setItem(ui->table_devices, 4, QString::number(device.videoPort));
    setItem(ui->table_devices, 5, QString::number(device.controlPort));
    setItem(ui->table_devices, 6, device.user);

    QString status;
    if (device.verified) {
        status = QString("已验证 %1").arg(device.resolution);
    } else {
        status = "未验证";
    }
    setItem(ui->table_devices, 7, status);

    // 自动选中第一行，方便直接「应用所选设备」
    if (row == 0) {
        ui->table_devices->selectRow(0);
    }
}

void SettingForm::onSearchProgress(int current, int total, const QString &ip)
{
    QString msg;
    if (ip.isEmpty()) {
        msg = (total > 0)
            ? QString("正在扫描网段… 共 %1 个地址").arg(total)
            : QString("正在扫描局域网…");
    } else {
        msg = QString("正在验证 %1 (%2/%3)").arg(ip).arg(current).arg(total);
    }
    ui->label_searchStatus->setText(msg);
}

void SettingForm::onSearchFinished(int totalFound)
{
    ui->btn_searchCameras->setEnabled(true);
    ui->btn_searchCameras->setText("🔍 搜索相机");

    if (totalFound > 0) {
        ui->label_searchStatus->setText(
            QString("搜索完成，发现 %1 台设备").arg(totalFound));
    } else {
        ui->label_searchStatus->setText("搜索完成，未发现设备");
    }
}

void SettingForm::onDeviceRowSelected(int row, int column)
{
    Q_UNUSED(column);
    if (row >= 0 && row < m_devices.size()) {
        ui->table_devices->selectRow(row);
    }
}

void SettingForm::onApplyDevice()
{
    const int row = ui->table_devices->currentRow();
    if (row < 0 || row >= m_devices.size()) {
        QMessageBox::information(this, "提示", "请先在列表中选择一台设备");
        return;
    }

    const CameraSearcher::Device &device = m_devices.at(row);
    ui->edit_cameraIP->setText(device.ip);
    ui->edit_cameraUser->setText(device.user);
    ui->edit_cameraPass->setText(device.pass);
    ui->spin_cameraPort->setValue(device.videoPort);

    // 用设备信息重建 RTSP URL（优先使用搜索到的验证路径）
    if (!device.rtspUrl.isEmpty()) {
        ui->edit_rtspUrl->setText(device.rtspUrl);
    } else {
        onRebuildRtspUrl();
    }

    ui->label_searchStatus->setText(
        QString("已应用设备 %1 到连接参数").arg(device.displayName()));
}

void SettingForm::onVerifyDevice()
{
    const QString ip   = ui->edit_cameraIP->text().trimmed();
    const QString user = ui->edit_cameraUser->text().trimmed();
    const QString pass = ui->edit_cameraPass->text().trimmed();
    const int port     = ui->spin_cameraPort->value();

    if (ip.isEmpty()) {
        QMessageBox::warning(this, "警告", "请先填写相机 IP");
        return;
    }

    QString path = QStringLiteral("/live/camera");
    // 尽量从现有 RTSP URL 中提取路径，避免覆盖用户自定义路径
    const QString existingUrl = ui->edit_rtspUrl->text().trimmed();
    const QRegularExpression re(QStringLiteral("rtsp://[^/]+(?<path>/.*)"));
    const QRegularExpressionMatch match = re.match(existingUrl);
    if (match.hasMatch() && !match.captured("path").isEmpty()) {
        path = match.captured("path");
    }

    const QString url = QString("rtsp://%1:%2@%3:%4%5")
                            .arg(user, pass, ip).arg(port).arg(path);

    ui->btn_verifyDevice->setEnabled(false);
    ui->btn_verifyDevice->setText("验证中…");
    QApplication::setOverrideCursor(Qt::WaitCursor);
    QApplication::processEvents();

    cv::VideoCapture cap;
    cap.set(cv::CAP_PROP_OPEN_TIMEOUT_MSEC, 3000);
    cap.set(cv::CAP_PROP_READ_TIMEOUT_MSEC, 3000);
    const bool opened = cap.open(url.toStdString(), cv::CAP_FFMPEG);
    cv::Mat frame;
    const bool gotFrame = opened && cap.read(frame) && !frame.empty();
    if (opened) {
        cap.release();
    }

    QApplication::restoreOverrideCursor();
    ui->btn_verifyDevice->setEnabled(true);
    ui->btn_verifyDevice->setText("验证连接");

    if (gotFrame) {
        const int w = static_cast<int>(frame.cols);
        const int h = static_cast<int>(frame.rows);
        QMessageBox::information(this, "验证成功",
            QString("RTSP 连接成功\n地址: %1\n分辨率: %2x%3").arg(url).arg(w).arg(h));
        ui->edit_rtspUrl->setText(url);
    } else {
        QMessageBox::warning(this, "验证失败",
            QString("无法打开 RTSP 流\n%1\n\n请检查 IP、端口、账号、密码及 RTSP 路径。").arg(url));
    }
}
