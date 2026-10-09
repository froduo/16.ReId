#include "settingform.h"
#include "ui_settingform.h"
#include "logger.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QFileInfo>
#include <QDir>
#include <QCoreApplication>

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

    LOG_INFO("[SettingForm] Settings dialog opened");
}

SettingForm::~SettingForm()
{
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
