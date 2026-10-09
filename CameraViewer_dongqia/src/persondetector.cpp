#include "persondetector.h"
#include "trt_yolo.h"
#include "logger.h"
#include <QFileInfo>

PersonDetector::PersonDetector()
    : m_confThreshold(0.45f)
    , m_nmsThreshold(0.5f)
    , m_loaded(false)
{
}

PersonDetector::~PersonDetector()
{
    delete m_trtYolo;
    delete m_trtLogger;
}

bool PersonDetector::loadModel(const QString &modelPath, float confThreshold, float nmsThreshold)
{
    m_confThreshold = confThreshold;
    m_nmsThreshold = nmsThreshold;

    // 判断是 TensorRT 引擎还是 ONNX 模型
    if (modelPath.endsWith(".engine", Qt::CaseInsensitive)) {
        // TensorRT 引擎模式
        try {
            m_trtLogger = new TrtLogger();
            m_trtYolo = new TrtYolo(modelPath.toStdString(), *m_trtLogger, confThreshold);
            m_useTrt = true;
            m_loaded = true;
            LOG_INFO("[PersonDetector] TensorRT engine loaded: " + modelPath);
            return true;
        } catch (const std::exception &e) {
            LOG_ERR(QString("[PersonDetector] Failed to load TensorRT engine: %1 - %2")
                        .arg(modelPath).arg(e.what()));
            m_useTrt = false;
            m_loaded = false;
            return false;
        }
    } else {
        // OpenCV DNN 模式 (ONNX)
        try {
            m_net = cv::dnn::readNetFromONNX(modelPath.toStdString());
            m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            m_useTrt = false;
            m_loaded = true;
            LOG_INFO("[PersonDetector] OpenCV DNN model loaded: " + modelPath);
            return true;
        } catch (const cv::Exception &e) {
            LOG_WARN(QString("[PersonDetector] Failed to load ONNX model: %1 - %2")
                         .arg(modelPath).arg(e.what()));
            m_loaded = false;
            return false;
        }
    }
}

QVector<cv::Rect> PersonDetector::detect(const cv::Mat &frame)
{
    QVector<cv::Rect> results;

    if (!m_loaded || frame.empty())
        return results;

    if (m_useTrt && m_trtYolo) {
        // TensorRT 推理
        m_trtYolo->preprocess(frame);
        m_trtYolo->infer();

        std::vector<Detection> detections;
        m_trtYolo->postprocess(detections);

        // 诊断日志：打印所有检测结果（包含非 person 类别）
        if (!detections.empty()) {
            LOG_INFO(QString("[PersonDetector] TensorRT total detections: %1 (confThreshold=%2)")
                         .arg(detections.size()).arg(m_confThreshold, 0, 'f', 2));
            for (size_t i = 0; i < detections.size() && i < 20; i++) {
                const auto &d = detections[i];
                LOG_INFO(QString("  det[%1]: class=%2 score=%3 box=(%4,%5,%6x%7)")
                             .arg(i).arg(d.class_id).arg(d.score, 0, 'f', 3)
                             .arg(d.x).arg(d.y).arg(d.w).arg(d.h));
            }
        } else {
            LOG_WARN(QString("[PersonDetector] TensorRT: NO detections at all (confThreshold=%1, image=%2x%3)")
                         .arg(m_confThreshold, 0, 'f', 2).arg(frame.cols).arg(frame.rows));
        }

        for (const auto &det : detections) {
            if (det.class_id == 0) { // person 类别
                results.push_back(cv::Rect(det.x, det.y, det.w, det.h));
            }
        }

        if (results.isEmpty() && !detections.empty()) {
            LOG_WARN(QString("[PersonDetector] Found %1 detections but NONE are person (class 0). "
                             "Top-down view may not be recognized as 'person' by COCO model.")
                         .arg(detections.size()));
        }

        return results;
    }

    // OpenCV DNN 推理
    const int inputSize = 640;
    int imgW = frame.cols;
    int imgH = frame.rows;

    float scale = std::min(static_cast<float>(inputSize) / imgW,
                           static_cast<float>(inputSize) / imgH);
    int padW = static_cast<int>(imgW * scale);
    int padH = static_cast<int>(imgH * scale);
    int dw = (inputSize - padW) / 2;
    int dh = (inputSize - padH) / 2;

    cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0 / 255.0,
                                          cv::Size(inputSize, inputSize),
                                          cv::Scalar(0, 0, 0), true, false);

    m_net.setInput(blob);

    std::vector<cv::Mat> outputs;
    m_net.forward(outputs, m_net.getUnconnectedOutLayersNames());

    if (outputs.empty())
        return results;

    cv::Mat output = outputs[0];
    cv::Mat detectionMat;
    if (output.dims == 3) {
        cv::Mat reshaped(output.size[1], output.size[2], CV_32F, output.ptr<float>());
        reshaped = reshaped.t();
        detectionMat = reshaped;
    } else {
        detectionMat = output;
    }

    int numClasses = detectionMat.cols - 4;

    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < detectionMat.rows; i++) {
        float *row = detectionMat.ptr<float>(i);
        cv::Mat scores(1, numClasses, CV_32F, row + 4);
        cv::Point classIdPoint;
        double maxConf;
        cv::minMaxLoc(scores, nullptr, &maxConf, nullptr, &classIdPoint);

        if (classIdPoint.x == 0 && maxConf >= m_confThreshold) {
            float cx = (row[0] - dw) / scale;
            float cy = (row[1] - dh) / scale;
            float w = row[2] / scale;
            float h = row[3] / scale;

            int left = std::max(0, static_cast<int>(cx - w / 2));
            int top = std::max(0, static_cast<int>(cy - h / 2));
            int width = std::min(static_cast<int>(w), imgW - left);
            int height = std::min(static_cast<int>(h), imgH - top);

            if (width > 0 && height > 0) {
                boxes.push_back(cv::Rect(left, top, width, height));
                confidences.push_back(static_cast<float>(maxConf));
            }
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, m_confThreshold, m_nmsThreshold, indices);

    for (int idx : indices) {
        results.push_back(boxes[idx]);
    }

    return results;
}

bool PersonDetector::isLoaded() const
{
    return m_loaded;
}

QVector<Detection> PersonDetector::detectAll(const cv::Mat &frame)
{
    QVector<Detection> results;
    if (!m_loaded || frame.empty())
        return results;

    if (m_useTrt && m_trtYolo) {
        m_trtYolo->preprocess(frame);
        m_trtYolo->infer();
        std::vector<Detection> dets;
        m_trtYolo->postprocessAll(dets, 0.05f);
        for (const auto &d : dets) results.push_back(d);
        return results;
    }

    // OpenCV DNN: 用低阈值检测所有类别
    const int inputSize = 640;
    int imgW = frame.cols;
    int imgH = frame.rows;

    float scale = std::min(static_cast<float>(inputSize) / imgW,
                           static_cast<float>(inputSize) / imgH);
    int padW = static_cast<int>(imgW * scale);
    int padH = static_cast<int>(imgH * scale);
    int dw = (inputSize - padW) / 2;
    int dh = (inputSize - padH) / 2;

    cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0 / 255.0,
                                          cv::Size(inputSize, inputSize),
                                          cv::Scalar(0, 0, 0), true, false);
    m_net.setInput(blob);
    std::vector<cv::Mat> outputs;
    m_net.forward(outputs, m_net.getUnconnectedOutLayersNames());
    if (outputs.empty()) return results;

    cv::Mat output = outputs[0];
    cv::Mat detectionMat;
    if (output.dims == 3) {
        cv::Mat reshaped(output.size[1], output.size[2], CV_32F, output.ptr<float>());
        reshaped = reshaped.t();
        detectionMat = reshaped;
    } else {
        detectionMat = output;
    }

    int numClasses = detectionMat.cols - 4;
    float lowThresh = 0.05f;

    for (int i = 0; i < detectionMat.rows; i++) {
        float *row = detectionMat.ptr<float>(i);
        cv::Mat scores(1, numClasses, CV_32F, row + 4);
        cv::Point classIdPoint;
        double maxConf;
        cv::minMaxLoc(scores, nullptr, &maxConf, nullptr, &classIdPoint);

        if (maxConf >= lowThresh) {
            float cx = (row[0] - dw) / scale;
            float cy = (row[1] - dh) / scale;
            float w = row[2] / scale;
            float h = row[3] / scale;

            Detection det;
            det.x = std::max(0, static_cast<int>(cx - w / 2));
            det.y = std::max(0, static_cast<int>(cy - h / 2));
            det.w = std::min(static_cast<int>(w), imgW - det.x);
            det.h = std::min(static_cast<int>(h), imgH - det.y);
            det.score = static_cast<float>(maxConf);
            det.class_id = classIdPoint.x;

            if (det.w > 0 && det.h > 0) {
                results.push_back(det);
            }
        }
    }

    return results;
}
