#ifndef PERSONDETECTOR_H
#define PERSONDETECTOR_H

#include <QString>
#include <QVector>
#include <vector>
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

    /// ONNX(DNN) 推理后端：true=优先 CUDA（不可用自动回退 CPU）。需在 loadModel 前设置
    void setDnnUseCuda(bool useCuda) { m_dnnUseCuda = useCuda; }
    /// ONNX(DNN) 推理输入尺寸（须与导出模型一致，默认 640）。需在 loadModel 前设置
    void setDnnInputSize(int size) { m_dnnInputSize = (size >= 160 ? size : 640); }
    /// 当前 ONNX 是否实际运行在 CUDA 上（loadModel 后有效）
    bool isDnnCudaActive() const { return m_dnnCudaActive; }
    /// 当前 ONNX 输入尺寸
    int dnnInputSize() const { return m_dnnInputSize; }

    // ===== 小目标过滤（按整幅图像尺寸的归一化比例）=====
    /**
     * @brief 设置小目标过滤阈值
     * @param minHeightRatio 框高 / 图像高 的最小值，低于则丢弃（如 0.08）
     * @param minAreaRatio   框面积 / 图像面积 的最小值，低于则丢弃（如 0.005）
     *
     * 之所以用「相对整幅图像的比例」，是因为相机分辨率固定（1920x1080），
     * 用比例可自动适配不同分辨率，避免把远处小人误判为行人。
     */
    void setMinSizeFilter(float minHeightRatio, float minAreaRatio);

    /// 启用/关闭小目标过滤
    void setMinSizeFilterEnabled(bool enabled) { m_sizeFilterEnabled = enabled; }
    bool minSizeFilterEnabled() const { return m_sizeFilterEnabled; }

    /**
     * @brief 设置「绝对像素尺寸」门槛（叠加在比例过滤之上）
     *
     * 比例过滤对"细长/极小但高度勉强达标"的误检（区域内的反光、衣角、局部误检）
     * 效果有限，这里再加一道像素级下限。
     *
     * @param minWidth  最小框宽（像素，默认 30）
     * @param minHeight 最小框高（像素，默认 90）
     */
    void setMinBoxSizePx(int minWidth, int minHeight);

    /**
     * @brief 设置人体框宽高比（宽/高）合理区间，默认 0.12 ~ 1.00
     *
     * 站立/行走的人体通常 0.2~0.6；过细（<0.12，竖条状误检）或
     * 过扁（>1.0，横向误检/多目标粘连）都视为不合理并丢弃。
     */
    void setBoxAspectRange(float minAspect, float maxAspect);

    /**
     * @brief 设置框「最大尺寸」上限（比例，1.0 表示不限制）
     *
     * 用于抑制「超大静态误检」：空场景中固定物体（座椅/地面/反光）被误判为目标时，
     * 其框通常远大于真实目标。head 头部框很小，超过上限即可判为非目标——
     * 比单纯抬高置信度更可靠（实测误检置信度可达 0.81）。
     * @param maxWidthRatio  框宽 / 图像宽 的最大值（<=0 或 >=1 视为不限制）
     * @param maxHeightRatio 框高 / 图像高 的最大值（<=0 或 >=1 视为不限制）
     * @param maxAreaRatio   框面积 / 图像面积 的最大值（<=0 或 >=1 视为不限制）
     */
    void setMaxSizeFilter(float maxWidthRatio, float maxHeightRatio, float maxAreaRatio = 1.0f);

    /**
     * @brief 静态误检抑制：同位置、近零位移的检测持续 N 帧后判为静态背景误检并忽略
     *
     * 固定摄像头下背景静止，真实乘客会移动；连续多帧停留在同一像素且位移极小的大框，
     * 基本可判定为背景误检。一旦确认即忽略，直到该区域长时间不再出现才解封。
     * @param enabled       是否启用
     * @param confirmFrames 连续命中多少帧后拉黑（帧数，越大越保守）
     * @param maxMoveRatio  判定「静止」的中心点相对起点位移上限（相对图像长边）
     * @param iouThreshold  与已有静态区域的 IoU 匹配阈值
     * @param missTolerance 连续多少帧未命中则移除该静态区域（容忍偶发漏检）
     */
    void setStaticSuppression(bool enabled, int confirmFrames,
                              float maxMoveRatio, float iouThreshold, int missTolerance);

    /**
     * @brief TensorRT 预处理模式：true=letterbox（保持长宽比+灰边），false=等比拉伸(squash)
     *
     * 需在 loadModel 之前设置。实测 head 头部检测模型在本相机下 squash 的置信度显著高于
     * letterbox（0.72 vs 0.23），后者会低于阈值导致“有人却检测不到”。
     */
    void setLetterbox(bool v) { m_trtLetterbox = v; }

    /// 是否通过小目标过滤（框尺寸相对整幅图像）
    bool passSizeFilter(const cv::Rect &box, int imgW, int imgH) const;

private:
    // OpenCV DNN 后端
    cv::dnn::Net m_net;
    float m_confThreshold;
    float m_nmsThreshold;
    bool m_loaded;

    // 小目标过滤参数（相对整幅图像的比例）
    bool  m_sizeFilterEnabled{true};
    float m_minHeightRatio{0.08f};   // 1080p 下约 86px
    float m_minAreaRatio{0.005f};    // 1080p 下约 10368px²

    // 绝对像素尺寸与宽高比（叠加过滤，专治区域内小框误检）
    int   m_minBoxWidthPx{30};
    int   m_minBoxHeightPx{90};
    float m_minAspect{0.12f};
    float m_maxAspect{1.00f};

    // 框最大尺寸上限（比例；1.0=不限制）——抑制超大静态误检
    float m_maxWidthRatio{1.0f};
    float m_maxHeightRatio{1.0f};
    float m_maxAreaRatio{1.0f};

    // ONNX(DNN) 后端配置
    bool m_dnnUseCuda{true};
    int  m_dnnInputSize{640};
    bool m_dnnCudaActive{false};

    // ===== 静态误检抑制（固定摄像头：同位置、近零位移的持续检测判为背景误检）=====
    struct StaticRegion {
        cv::Rect    box;
        cv::Point2f refCenter;   // 起始中心（用于累计位移判断）
        int         hits{0};
        int         miss{0};
        bool        suppressed{false};
    };
    bool  m_staticEnabled{false};
    int   m_staticConfirmFrames{150};
    float m_staticMaxMoveRatio{0.008f};
    float m_staticIouThresh{0.70f};
    int   m_staticMissTolerance{30};
    std::vector<StaticRegion> m_staticRegions;

    QVector<cv::Rect> applyStaticSuppression(const QVector<cv::Rect> &boxes,
                                             int imgW, int imgH);

    // TensorRT 后端
    bool m_useTrt{false};
    bool m_trtLetterbox{true};   // TRT 预处理：letterbox / squash
    TrtYolo *m_trtYolo{nullptr};
    TrtLogger *m_trtLogger{nullptr};
};

#endif // PERSONDETECTOR_H
