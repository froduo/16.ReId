#ifndef SETTINGFORM_H
#define SETTINGFORM_H

#include <QDialog>
#include <QVector>
#include "app_config.h"
#include "camerasearcher.h"

namespace Ui {
class SettingForm;
}

class SettingForm : public QDialog
{
    Q_OBJECT

public:
    explicit SettingForm(const AppConfig &config, QWidget *parent = nullptr);
    ~SettingForm() override;

    AppConfig getConfig() const;

private slots:
    void onAccept();
    void onBrowseModel();
    void onBrowseSavePath();
    void onBrowseReidSavePath();
    void onRebuildRtspUrl();

    // 相机搜索
    void onSearchCameras();
    void onApplyDevice();
    void onVerifyDevice();
    void onDeviceRowSelected(int row, int column);
    void onDeviceFound(const CameraSearcher::Device &device);
    void onSearchProgress(int current, int total, const QString &ip);
    void onSearchFinished(int totalFound);

private:
    Ui::SettingForm *ui;
    AppConfig m_config;
    CameraSearcher *m_searcher = nullptr;
    QVector<CameraSearcher::Device> m_devices;
};

#endif // SETTINGFORM_H
