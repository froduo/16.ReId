#include "persondetector.h"
#include "trt_yolo.h"
#include "logger.h"
#include <fstream>
#include <QElapsedTimer>
#include <QElapsedTimer>
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

void PersonDetector::setMinSizeFilter(float minHeightRatio, float minAreaRatio)
{
    m_minHeightRatio = qBound(0.0f, minHeightRatio, 1.0f);
    m_minAreaRatio   = qBound(0.0f, minAreaRatio, 1.0f);
    LOG_INFO(QString("[PersonDetector] Min-size filter set: heightRatio>=%1, areaRatio>=%2 (enabled=%3)")
                 .arg(m_minHeightRatio, 0, 'f', 4)
                 .arg(m_minAreaRatio, 0, 'f', 4)
                 .arg(m_sizeFilterEnabled ? "true" : "false"));
}

void PersonDetector::setMinBoxSizePx(int minWidth, int minHeight)
{
    m_minBoxWidthPx = qMax(1, minWidth);
    m_minBoxHeightPx = qMax(1, minHeight);
    LOG_INFO(QString("[PersonDetector] Min box size set: width>=%1px, height>=%2px, "
                     "aspect(宽/高)∈[%3, %4]")
                 .arg(m_minBoxWidthPx).arg(m_minBoxHeightPx)
                 .arg(m_minAspect, 0, 'f', 3).arg(m_maxAspect, 0, 'f', 3));
}

void PersonDetector::setBoxAspectRange(float minAspect, float maxAspect)
{
    m_minAspect = qBound(0.01f, minAspect, 5.0f);
    m_maxAspect = qBound(m_minAspect, maxAspect, 10.0f);
    LOG_INFO(QString("[PersonDetector] Box aspect range set: [%1, %2] (宽/高)")
                 .arg(m_minAspect, 0, 'f', 3).arg(m_maxAspect, 0, 'f', 3));
}

bool PersonDetector::passSizeFilter(const cv::Rect &box, int imgW, int imgH) const
{
    if (!m_sizeFilterEnabled) return true;
    if (imgW <= 0 || imgH <= 0) return true;
    if (box.width <= 0 || box.height <= 0) return false;

    // ===== 绝对像素尺寸过滤（优先判断，专治区域内的小框误检）=====
    if (box.width < m_minBoxWidthPx || box.height < m_minBoxHeightPx) return false;

    // ===== 宽高比合理性：过滤竖条/横条状误检 =====
    const float aspect = static_cast<float>(box.width) / static_cast<float>(box.height);
    if (aspect < m_minAspect || aspect > m_maxAspect) return false;

    // 高度比例过滤：远处/误识别的小框直接丢弃
    const float heightRatio = static_cast<float>(box.height) / static_cast<float>(imgH);
    if (heightRatio < m_minHeightRatio) return false;

    // 面积比例过滤：极扁/极小的框（如地面反光、零件）丢弃
    const float areaRatio = static_cast<float>(box.width) * static_cast<float>(box.height)
                            / (static_cast<float>(imgW) * static_cast<float>(imgH));
    if (areaRatio < m_minAreaRatio) return false;

    return true;
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
            m_trtYolo->setLetterbox(m_trtLetterbox);   // 预处理模式（head 模型建议 squash）
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
            m_useTrt = false;
            m_dnnCudaActive = false;

            // ===== 后端选择：强制 CPU =====
            // 重要：OpenCV 的 CUDA 后端(DNN_BACKEND_CUDA) 对该 YOLOv8 ONNX 模型会输出
            // 错误张量（框坐标恒为 0、person 类分数虚高），导致“未识别人/误检”。
            // 已通过与 onnxruntime 及 cv2.dnn(CPU) 逐元素比对确认：CPU 后端结果正确。
            // CPU 后端单帧约 10ms，满足实时性，故此处强制使用 CPU。
            m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            m_dnnCudaActive = false;

            m_loaded = true;
            LOG_INFO(QString("[PersonDetector] OpenCV DNN(ONNX) loaded: %1 | backend=%2 | input=%3x%3")
                         .arg(modelPath)
                         .arg(m_dnnCudaActive ? "CUDA(GPU)" : "CPU")
                         .arg(m_dnnInputSize));
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

        // 诊断日志（限频：最多每秒 1 条，避免逐帧写盘拖慢采集/检测线程）
        static QElapsedTimer s_diagTimer;
        if (!s_diagTimer.isValid()) s_diagTimer.start();
        const bool diagLog = (s_diagTimer.elapsed() >= 1000);
        if (diagLog) s_diagTimer.restart();

        if (!detections.empty()) {
            if (diagLog) {
                LOG_INFO(QString("[PersonDetector] TensorRT total detections: %1 (confThreshold=%2)")
                             .arg(detections.size()).arg(m_confThreshold, 0, 'f', 2));
                for (size_t i = 0; i < detections.size() && i < 20; i++) {
                    const auto &d = detections[i];
                    LOG_INFO(QString("  det[%1]: class=%2 score=%3 box=(%4,%5,%6x%7)")
                                 .arg(i).arg(d.class_id).arg(d.score, 0, 'f', 3)
                                 .arg(d.x).arg(d.y).arg(d.w).arg(d.h));
                }
            }
        } else if (diagLog) {
            LOG_WARN(QString("[PersonDetector] TensorRT: NO detections at all (confThreshold=%1, image=%2x%3)")
                         .arg(m_confThreshold, 0, 'f', 2).arg(frame.cols).arg(frame.rows));
        }

        int filteredSmall = 0;
        for (const auto &det : detections) {
            if (det.class_id != 0) continue; // 仅保留 person 类别
            const cv::Rect box(det.x, det.y, det.w, det.h);
            if (!passSizeFilter(box, frame.cols, frame.rows)) {
                filteredSmall++;
                continue;
            }
            results.push_back(box);
        }

        if (filteredSmall > 0) {
            LOG_INFO(QString("[PersonDetector] Size filter dropped %1 small/implausible boxes "
                             "(image=%2x%3, hRatio>=%4, aRatio>=%5)")
                         .arg(filteredSmall).arg(frame.cols).arg(frame.rows)
                         .arg(m_minHeightRatio, 0, 'f', 3).arg(m_minAreaRatio, 0, 'f', 4));
        }

        if (results.isEmpty() && !detections.empty()) {
            LOG_WARN(QString("[PersonDetector] Found %1 detections but NONE are valid person (class 0) "
                             "after size filtering (confThreshold=%2, image=%3x%4).")
                         .arg(detections.size()).arg(m_confThreshold, 0, 'f', 2)
                         .arg(frame.cols).arg(frame.rows));
        }

        return results;
    }

    // OpenCV DNN 推理（输入尺寸须与导出模型一致）
    const int inputSize = m_dnnInputSize;
    int imgW = frame.cols;
    int imgH = frame.rows;

    // ===== letterbox 预处理（保持长宽比 + 114 灰边），与下方坐标映射一致 =====
    // 旧实现用 blobFromImage 直接拉伸（等价 squash）却按 letterbox 映射坐标，
    // 会使框位置/尺寸错乱进而被尺寸过滤误杀；这里修正为真正的 letterbox。
    const float scale = std::min(static_cast<float>(inputSize) / imgW,
                                 static_cast<float>(inputSize) / imgH);
    const int newW = static_cast<int>(std::round(imgW * scale));
    const int newH = static_cast<int>(std::round(imgH * scale));
    const int dw = (inputSize - newW) / 2;
    const int dh = (inputSize - newH) / 2;

    cv::Mat canvas(inputSize, inputSize, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(newW, newH));
    resized.copyTo(canvas(cv::Rect(dw, dh, newW, newH)));

    // 【一次性】把 letterbox 后的 640×640 输入存盘，便于用 Python 复算对比张量读取是否正确
    static bool s_canvasDumped = false;
    if (!s_canvasDumped) { s_canvasDumped = true; cv::imwrite("/tmp/orbbec_canvas.png", canvas); }

    cv::Mat blob = cv::dnn::blobFromImage(canvas, 1.0 / 255.0,
                                          cv::Size(inputSize, inputSize),
                                          cv::Scalar(0, 0, 0), true, false);

    m_net.setInput(blob);

    std::vector<cv::Mat> outputs;
    m_net.forward(outputs, m_net.getUnconnectedOutLayersNames());

    if (outputs.empty())
        return results;

    cv::Mat output = outputs[0];
    const float *data = output.ptr<float>();
    // 解析张量几何：YOLOv8 官方导出为 1 x attrs(4+nc) x anchors（如 1x84x8400）。
    // attrs(≈84) 远小于 anchors(8400)，据此判定“属性在前(通道主序)”还是“锚点在前”。
    int attrs = 0, anchors = 0;
    bool channelMajor = true;   // true: data[c*anchors + a]（属性 c、锚点 a）
    if (output.dims >= 3 && output.size[1] > 0 && output.size[2] > 0) {
        const int d1 = output.size[1], d2 = output.size[2];
        attrs   = std::min(d1, d2);
        anchors = std::max(d1, d2);
        channelMajor = (d1 == attrs);
    } else if (output.dims == 2) {
        const int r = output.rows, c = output.cols;
        attrs   = std::min(r, c);
        anchors = std::max(r, c);
        channelMajor = (r == attrs);
    } else {
        anchors = output.rows; attrs = output.cols; channelMajor = false;
    }

    // 直接按原始缓冲索引读取，避免 Mat 视图/转置与 numpy 语义不一致导致的错位。
    auto at = [&](int a, int c) -> float {
        return channelMajor ? data[static_cast<size_t>(c) * anchors + a]
                            : data[static_cast<size_t>(a) * attrs + c];
    };

    const int numClasses = std::max(0, attrs - 4);

    // 【一次性】把 C++ 侧输出张量原始缓冲导出，供 Python 与 onnxruntime 逐元素对比
    static bool s_tensorDumped = false;
    if (!s_tensorDumped) {
        s_tensorDumped = true;
        std::ofstream ofs("/tmp/orbbec_out.f32", std::ios::binary);
        ofs.write(reinterpret_cast<const char *>(data),
                  static_cast<std::streamsize>(sizeof(float) * attrs * anchors));
    }

    // 一次性布局诊断（确认维度与主轴方向）
    static bool s_layoutLogged = false;
    if (!s_layoutLogged) {
        s_layoutLogged = true;
        LOG_INFO(QString("[PersonDetector][ONNX] layout: dims=%1 size=(%2,%3,%4) attrs=%5 anchors=%6 channelMajor=%7")
                     .arg(output.dims)
                     .arg(output.dims > 0 ? output.size[0] : 0)
                     .arg(output.dims > 1 ? output.size[1] : 0)
                     .arg(output.dims > 2 ? output.size[2] : 0)
                     .arg(attrs).arg(anchors).arg(channelMajor ? 1 : 0));
    }

    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    float bestPersonConf = 0.0f;   // 诊断：person 类最高分（含低于阈值者）
    int   scorePass = 0;           // 诊断：person 类过阈值(但未必通过几何)的锚点数
    int   bestRow = -1;            // 诊断：最高分 person 所在锚点
    float bestRaw[8] = {0,0,0,0,0,0,0,0};  // 诊断：该锚点前 8 个原始值

    for (int a = 0; a < anchors; ++a) {
        int   clsMax = -1;
        float scMax = -1.0f;
        for (int c = 0; c < numClasses; ++c) {
            const float s = at(a, 4 + c);
            if (s > scMax) { scMax = s; clsMax = c; }
        }

        if (clsMax == 0 && scMax > bestPersonConf) {
            bestPersonConf = scMax;
            bestRow = a;
            for (int k = 0; k < 8 && k < attrs; ++k) bestRaw[k] = at(a, k);
        }

        if (clsMax == 0 && scMax >= m_confThreshold) {
            scorePass++;
            const float cx = (at(a,0) - dw) / scale;
            const float cy = (at(a,1) - dh) / scale;
            const float w  = at(a,2) / scale;
            const float h  = at(a,3) / scale;

            int left = std::max(0, static_cast<int>(cx - w / 2));
            int top  = std::max(0, static_cast<int>(cy - h / 2));
            int width  = std::min(static_cast<int>(w), imgW - left);
            int height = std::min(static_cast<int>(h), imgH - top);

            if (width > 0 && height > 0) {
                boxes.push_back(cv::Rect(left, top, width, height));
                confidences.push_back(scMax);
            }
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, m_confThreshold, m_nmsThreshold, indices);

    int filteredSmall = 0;
    for (int idx : indices) {
        const cv::Rect &box = boxes[idx];
        if (!passSizeFilter(box, imgW, imgH)) { filteredSmall++; continue; }
        results.push_back(box);
    }

    if (filteredSmall > 0) {
        LOG_INFO(QString("[PersonDetector] Size filter dropped %1 small/implausible boxes "
                         "(image=%2x%3, hRatio>=%4, aRatio>=%5)")
                     .arg(filteredSmall).arg(imgW).arg(imgH)
                     .arg(m_minHeightRatio, 0, 'f', 3).arg(m_minAreaRatio, 0, 'f', 4));
    }

    // 诊断日志（限频 1s）：确认 ONNX(YOLO) 路径是否正常出分
    static QElapsedTimer s_dnnDiagTimer;
    if (!s_dnnDiagTimer.isValid()) s_dnnDiagTimer.start();
    if (s_dnnDiagTimer.elapsed() >= 1000) {
        s_dnnDiagTimer.restart();
        LOG_INFO(QString("[PersonDetector][ONNX] image=%1x%2 候选=%3 通过过滤=%4 阈值=%5 "
                         "person最高分=%6 scorePass=%7 bestRow=%8 raw8=[%9,%10,%11,%12,%13,%14,%15,%16]")
                     .arg(imgW).arg(imgH)
                     .arg(static_cast<int>(boxes.size()))
                     .arg(results.size())
                     .arg(m_confThreshold, 0, 'f', 2)
                     .arg(bestPersonConf, 0, 'f', 3)
                     .arg(scorePass).arg(bestRow)
                     .arg(bestRaw[0], 0, 'f', 1).arg(bestRaw[1], 0, 'f', 1)
                     .arg(bestRaw[2], 0, 'f', 1).arg(bestRaw[3], 0, 'f', 1)
                     .arg(bestRaw[4], 0, 'f', 3).arg(bestRaw[5], 0, 'f', 3)
                     .arg(bestRaw[6], 0, 'f', 3).arg(bestRaw[7], 0, 'f', 3));
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
