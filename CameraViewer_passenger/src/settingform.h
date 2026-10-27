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
    void syncRtspUrlFromFields();   // IP/端口/用户名/密码 变更时联动刷新 RTSP URL
    void onRtspUrlEdited();         // 用户手动编辑 RTSP URL 字段

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
    bool m_urlUserEdited = false;   // 本次会话中用户是否手动编辑过 RTSP URL
    CameraSearcher *m_searcher = nullptr;
    QVector<CameraSearcher::Device> m_devices;
};

#endif // SETTINGFORM_H
