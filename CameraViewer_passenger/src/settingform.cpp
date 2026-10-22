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
    // 配置为手动指定 URL 时，视作“已手动编辑”，避免 IP 联动覆盖自定义地址
    m_urlUserEdited = m_config.cameraRtspManual;

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
    // 相机 IP/端口/用户名/密码 任一变更 → 联动刷新 RTSP URL（保证「改 IP 即生效」）
    connect(ui->edit_cameraIP, &QLineEdit::textChanged, this, &SettingForm::syncRtspUrlFromFields);
    connect(ui->edit_cameraUser, &QLineEdit::textChanged, this, &SettingForm::syncRtspUrlFromFields);
    connect(ui->edit_cameraPass, &QLineEdit::textChanged, this, &SettingForm::syncRtspUrlFromFields);
    connect(ui->spin_cameraPort, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &SettingForm::syncRtspUrlFromFields);
    // 用户直接编辑 RTSP URL 字段 → 记为手动指定（尊重自定义地址）
    connect(ui->edit_rtspUrl, &QLineEdit::textEdited, this, &SettingForm::onRtspUrlEdited);

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

    // 关键修复：默认以「相机 IP + 端口 + 用户名 + 密码」为唯一事实来源，收敛 RTSP URL，
    // 确保用户仅修改 IP 时相机也会连接新地址（旧逻辑会沿用对话框里预填的旧 URL）。
    // 仅当用户在本次会话「手动编辑」了 URL 字段且非空时，才按自定义地址处理。
    if (m_urlUserEdited && !m_config.rtspUrl.isEmpty()) {
        m_config.cameraRtspManual = true;
    } else {
        m_config.cameraRtspManual = false;
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
    // 恢复为“由 IP/端口等自动生成”模式
    m_urlUserEdited = false;
    m_config.cameraRtspManual = false;
    m_config.cameraIP = ui->edit_cameraIP->text().trimmed();
    m_config.cameraUser = ui->edit_cameraUser->text().trimmed();
    m_config.cameraPass = ui->edit_cameraPass->text().trimmed();
    m_config.cameraPort = ui->spin_cameraPort->value();
    m_config.rebuildRtspUrl();
    ui->edit_rtspUrl->setText(m_config.rtspUrl);

    LOG_INFO("[SettingForm] RTSP URL rebuilt: " + m_config.rtspUrl);
}

void SettingForm::syncRtspUrlFromFields()
{
    // IP/端口/用户名/密码 变更 → 联动刷新 URL 字段。
    // 若用户已手动编辑 URL，则尊重其自定义地址，不覆盖。
    if (m_urlUserEdited) return;
    m_config.cameraIP = ui->edit_cameraIP->text().trimmed();
    m_config.cameraUser = ui->edit_cameraUser->text().trimmed();
    m_config.cameraPass = ui->edit_cameraPass->text().trimmed();
    m_config.cameraPort = ui->spin_cameraPort->value();
    ui->edit_rtspUrl->setText(m_config.derivedRtspUrl());
}

void SettingForm::onRtspUrlEdited()
{
    m_urlUserEdited = true;
}
