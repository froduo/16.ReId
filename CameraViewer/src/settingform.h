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

private:
    Ui::SettingForm *ui;
    AppConfig m_config;
};

#endif // SETTINGFORM_H
