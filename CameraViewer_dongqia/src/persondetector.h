#ifndef PERSONDETECTOR_H
#define PERSONDETECTOR_H

#include <QString>
#include <QVector>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "trt_yolo.h"

// 前置声明
class TrtYolo;
class TrtLogger;

class PersonDetector
{
public:
    PersonDetector();
    ~PersonDetector();

    bool loadModel(const QString &modelPath, float confThreshold = 0.45f, float nmsThreshold = 0.5f);
    QVector<cv::Rect> detect(const cv::Mat &frame);
    /// 返回所有类别检测结果（含 class_id, score 等，用于诊断）
    QVector<Detection> detectAll(const cv::Mat &frame);
    bool isLoaded() const;

    /// 是否使用 TensorRT 后端
    bool isTrtBackend() const { return m_useTrt; }

private:
    // OpenCV DNN 后端
    cv::dnn::Net m_net;
    float m_confThreshold;
    float m_nmsThreshold;
    bool m_loaded;

    // TensorRT 后端
    bool m_useTrt{false};
    TrtYolo *m_trtYolo{nullptr};
    TrtLogger *m_trtLogger{nullptr};
};

#endif // PERSONDETECTOR_H
