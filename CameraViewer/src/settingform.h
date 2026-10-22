#ifndef SETTINGFORM_H
#define SETTINGFORM_H

#include <QDialog>
#include "app_config.h"

namespace Ui {
class SettingForm;
}

class SettingForm : public QDialog
{
    Q_OBJECT

public:
    explicit SettingForm(const AppConfig &config, QWidget *parent = nullptr);
    ~SettingForm();

    AppConfig getConfig() const;

private slots:
    void onAccept();
    void onBrowseModel();
    void onBrowseSavePath();
    void onBrowseReidSavePath();
    void onRebuildRtspUrl();
    void syncRtspUrlFromFields();   // IP/端口/用户名/密码 变更时联动刷新 RTSP URL
    void onRtspUrlEdited();         // 用户手动编辑 RTSP URL 字段

private:
    Ui::SettingForm *ui;
    AppConfig m_config;
    bool m_urlUserEdited = false;   // 本次会话中用户是否手动编辑过 RTSP URL
};

#endif // SETTINGFORM_H
