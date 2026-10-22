#include "persondetector.h"
#include "trt_yolo.h"
#include "logger.h"
#include <QElapsedTimer>
#include <QFileInfo>
#include <algorithm>
#include <cmath>

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

void PersonDetector::setMaxSizeFilter(float maxWidthRatio, float maxHeightRatio, float maxAreaRatio)
{
    // <=0 或 >=1 视为不限制
    m_maxWidthRatio  = (maxWidthRatio  > 0.0f && maxWidthRatio  < 1.0f) ? maxWidthRatio  : 1.0f;
    m_maxHeightRatio = (maxHeightRatio > 0.0f && maxHeightRatio < 1.0f) ? maxHeightRatio : 1.0f;
    m_maxAreaRatio   = (maxAreaRatio   > 0.0f && maxAreaRatio   < 1.0f) ? maxAreaRatio   : 1.0f;
    LOG_INFO(QString("[PersonDetector] Max-size filter set: widthRatio<=%1, heightRatio<=%2, "
                     "areaRatio<=%3 (1.0=不限)")
                 .arg(m_maxWidthRatio, 0, 'f', 3)
                 .arg(m_maxHeightRatio, 0, 'f', 3)
                 .arg(m_maxAreaRatio, 0, 'f', 3));
}

void PersonDetector::setStaticSuppression(bool enabled, int confirmFrames,
                                          float maxMoveRatio, float iouThreshold, int missTolerance)
{
    m_staticEnabled       = enabled;
    m_staticConfirmFrames = qMax(1, confirmFrames);
    m_staticMaxMoveRatio  = qBound(0.0005f, maxMoveRatio, 0.20f);
    m_staticIouThresh     = qBound(0.10f, iouThreshold, 0.99f);
    m_staticMissTolerance = qMax(1, missTolerance);
    LOG_INFO(QString("[PersonDetector] Static suppression set: enabled=%1, confirmFrames=%2, "
                     "maxMoveRatio=%3, iou>%4, missTolerance=%5")
                 .arg(m_staticEnabled ? "true" : "false")
                 .arg(m_staticConfirmFrames)
                 .arg(m_staticMaxMoveRatio, 0, 'f', 4)
                 .arg(m_staticIouThresh, 0, 'f', 2)
                 .arg(m_staticMissTolerance));
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

    // ===== 最大尺寸上限：抑制超大静态误检（空场景固定物体被误判）=====
    if (m_maxWidthRatio < 0.999f || m_maxHeightRatio < 0.999f) {
        const float wRatio = static_cast<float>(box.width) / static_cast<float>(imgW);
        const float hRatio = static_cast<float>(box.height) / static_cast<float>(imgH);
        if (wRatio > m_maxWidthRatio || hRatio > m_maxHeightRatio) return false;
    }
    if (m_maxAreaRatio > 0.0f && m_maxAreaRatio < 0.999f && areaRatio > m_maxAreaRatio) {
        return false;
    }

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

            // ===== 后端选择：优先 CUDA（GPU），不可用自动回退 CPU =====
            if (m_dnnUseCuda) {
                try {
                    m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                    m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
                    m_dnnCudaActive = true;
                } catch (const cv::Exception &e) {
                    m_dnnCudaActive = false;
                    LOG_WARN(QString("[PersonDetector] DNN CUDA 后端不可用: %1")
                                 .arg(QString::fromStdString(e.what())));
                }
            }

            // 自检：用一次前向验证后端真正可用（部分环境设置后端不报错，
            // 直到 forward 才抛异常）；失败则回退 CPU 并重试。
            if (m_dnnCudaActive) {
                try {
                    cv::Mat dummy(m_dnnInputSize, m_dnnInputSize, CV_8UC3, cv::Scalar(0, 0, 0));
                    cv::Mat blob = cv::dnn::blobFromImage(
                        dummy, 1.0 / 255.0, cv::Size(m_dnnInputSize, m_dnnInputSize),
                        cv::Scalar(0, 0, 0), true, false);
                    m_net.setInput(blob);
                    m_net.forward();
                } catch (const cv::Exception &e) {
                    LOG_WARN(QString("[PersonDetector] DNN CUDA 前向自检失败，回退 CPU: %1")
                                 .arg(QString::fromStdString(e.what())));
                    m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                    m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
                    m_dnnCudaActive = false;
                }
            }

            if (!m_dnnCudaActive) {
                m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            }

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

        // 「无人检测」日志单独降频（10 秒 1 条），空场景下不刷屏
        static QElapsedTimer s_noDetTimer;
        if (!s_noDetTimer.isValid()) s_noDetTimer.start();
        const bool noDetLog = (s_noDetTimer.elapsed() >= 10000);
        if (noDetLog) s_noDetTimer.restart();

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
        } else if (noDetLog) {
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

        // 静态误检抑制：同位置、近零位移的持续检测判为背景误检并忽略
        results = applyStaticSuppression(results, frame.cols, frame.rows);

        return results;
    }

    // OpenCV DNN 推理（输入尺寸须与导出模型一致）
    const int inputSize = m_dnnInputSize;
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

    // 静态误检抑制：同位置、近零位移的持续检测判为背景误检并忽略
    results = applyStaticSuppression(results, imgW, imgH);

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

// ===== 静态误检抑制实现 =====
namespace {
inline float rectIoU(const cv::Rect &a, const cv::Rect &b)
{
    const int x1 = std::max(a.x, b.x);
    const int y1 = std::max(a.y, b.y);
    const int x2 = std::min(a.x + a.width,  b.x + b.width);
    const int y2 = std::min(a.y + a.height, b.y + b.height);
    const int iw = x2 - x1;
    const int ih = y2 - y1;
    if (iw <= 0 || ih <= 0) return 0.0f;
    const float inter = static_cast<float>(iw) * static_cast<float>(ih);
    const float uni = static_cast<float>(a.area()) + static_cast<float>(b.area()) - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}
} // namespace

QVector<cv::Rect> PersonDetector::applyStaticSuppression(const QVector<cv::Rect> &boxes,
                                                         int imgW, int imgH)
{
    if (!m_staticEnabled || imgW <= 0 || imgH <= 0)
        return boxes;

    const float maxMove = m_staticMaxMoveRatio * static_cast<float>(std::max(imgW, imgH));

    // 1) 已有区域先记为「本帧未命中」
    for (auto &r : m_staticRegions) r.miss++;

    std::vector<bool> matched(m_staticRegions.size(), false);
    QVector<cv::Rect> kept;
    kept.reserve(boxes.size());

    for (const cv::Rect &box : boxes) {
        const float cx = box.x + box.width * 0.5f;
        const float cy = box.y + box.height * 0.5f;

        // 2) 找与已有区域匹配（IoU 足够且相对起点位移极小）
        int best = -1;
        float bestDist = 1e9f;
        for (size_t k = 0; k < m_staticRegions.size(); ++k) {
            if (matched[k]) continue;
            if (rectIoU(box, m_staticRegions[k].box) < m_staticIouThresh) continue;
            const float d = std::hypot(cx - m_staticRegions[k].refCenter.x,
                                       cy - m_staticRegions[k].refCenter.y);
            if (d > maxMove) continue;   // 相对起点移动过大 → 不是同一静止目标
            if (d < bestDist) { bestDist = d; best = static_cast<int>(k); }
        }

        if (best >= 0) {
            StaticRegion &r = m_staticRegions[best];
            matched[best] = true;
            r.box = box;
            r.miss = 0;
            // 仅「连续出现」才累计；明显中断后重新计数，避免间断命中被误判为静止
            if (r.hits > 0 && r.miss > 3) r.hits = 0;
            r.hits++;

            if (!r.suppressed && r.hits >= m_staticConfirmFrames) {
                r.suppressed = true;
                LOG_WARN(QString("[PersonDetector] 静态误检抑制：框(%1,%2,%3x%4) 连续 %5 帧位置基本不变，"
                                 "判定为静态背景误检，已忽略")
                             .arg(box.x).arg(box.y).arg(box.width).arg(box.height).arg(r.hits));
            }
            if (r.suppressed) continue;   // 命中已确认的静态误检 → 丢弃
        } else {
            // 新建区域（限制总数，避免无界增长）
            if (m_staticRegions.size() < 128) {
                StaticRegion nr;
                nr.box = box;
                nr.refCenter = cv::Point2f(cx, cy);
                nr.hits = 1;
                nr.miss = 0;
                m_staticRegions.push_back(nr);
                matched.push_back(true);
            }
        }

        kept.push_back(box);
    }

    // 3) 长时间未出现的区域移除（目标离开后自动「解封」）
    for (int k = static_cast<int>(m_staticRegions.size()) - 1; k >= 0; --k) {
        if (!matched[k] && m_staticRegions[k].miss > m_staticMissTolerance) {
            if (m_staticRegions[k].suppressed) {
                LOG_INFO(QString("[PersonDetector] 静态误检区域解封（连续 %1 帧未出现）")
                             .arg(m_staticRegions[k].miss));
            }
            m_staticRegions.erase(m_staticRegions.begin() + k);
        }
    }

    return kept;
}
