#ifndef OFFLINEDETECTFORM_H
#define OFFLINEDETECTFORM_H

#include <QDialog>
#include <QLabel>
#include <QPushButton>
#include <QTableWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFileDialog>
#include <opencv2/opencv.hpp>
#include "imageview.h"
#include "persondetector.h"
#include "persontracker.h"
#include "reidextractor.h"
#include "regionmanager.h"

class OfflineDetectForm : public QDialog
{
    Q_OBJECT
public:
    explicit OfflineDetectForm(PersonDetector *detector,
                               RegionManager *regionManager,
                               QWidget *parent = nullptr);
    ~OfflineDetectForm();

private slots:
    void onLoadImage();
    void onRunDetect();

private:
    void setupUI();
    void displayResult(const cv::Mat &frame,
                       const QVector<cv::Rect> &detections,
                       const QVector<TrackedPerson> &trackedPersons);
    void drawRegionOverlay(cv::Mat &frame, int imgW, int imgH);

    ImageView *m_imageView{nullptr};
    QPushButton *m_btnLoad{nullptr};
    QPushButton *m_btnDetect{nullptr};
    QLabel *m_labelInfo{nullptr};
    QTableWidget *m_tableResults{nullptr};

    PersonDetector *m_detector{nullptr};
    PersonTracker *m_tracker{nullptr};
    ReIDExtractor *m_reidExtractor{nullptr};
    RegionManager *m_regionManager{nullptr};

    cv::Mat m_currentImage;
    QString m_currentImagePath;
};

#endif // OFFLINEDETECTFORM_H
