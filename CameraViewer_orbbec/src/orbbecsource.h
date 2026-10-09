#ifndef ORBBECSOURCE_H
#define ORBBECSOURCE_H

#include <opencv2/opencv.hpp>
#include <cstdint>
#include <memory>
#include <string>

/**
 * @brief Orbbec Gemini 335 深度相机采集封装（OrbbecSDK v2）
 *
 * 同时输出彩色（BGR，CV_8UC3）与深度（毫米，CV_16U，已按 depthScale 换算）。
 *
 * D2C 对齐策略（实测经 ob::Align 验证）：
 *  - 优先硬件对齐 ALIGN_D2C_HW_MODE（无 CPU 开销）；
 *  - 硬件不支持时改用 SDK 的 **Align 滤镜**（输出到彩色分辨率，真正逐像素对应）。
 *    注意：config 的 ALIGN_D2C_SW_MODE 实测**不会**改变深度分辨率（深度仍为
 *    物理 848x480），因此不能用它来保证像素对应。
 */
class OrbbecSource
{
public:
    OrbbecSource();
    ~OrbbecSource();

    /// 打开相机；w/h/fps 传 0 表示使用默认 profile。返回是否成功。
    bool open(int colorWidth = 0, int colorHeight = 0, int fps = 0);
    bool isOpen() const { return m_open; }
    void close();

    /**
     * @brief 取一帧（阻塞至多 timeoutMs）
     * @param colorBGR 彩色图 CV_8UC3
     * @param depthMM  深度图 CV_16U（毫米；0 表示无效）
     * @param colorTsUs 彩色帧时间戳（微秒）
     * @param depthTsUs 深度帧时间戳（微秒）
     */
    bool getFrames(cv::Mat &colorBGR, cv::Mat &depthMM,
                   uint64_t &colorTsUs, uint64_t &depthTsUs, int timeoutMs = 200);

    int width() const { return m_w; }
    int height() const { return m_h; }
    bool hasDepth() const { return m_hasDepth; }
    std::string lastError() const { return m_err; }
    std::string deviceName() const { return m_devName; }

    // ===== 硬件参数（open 后有效，供文档/测量使用）=====
    int depthWidth() const { return m_depthW; }        ///< 深度图宽（对齐后 = 彩色宽）
    int depthHeight() const { return m_depthH; }       ///< 深度图高（对齐后 = 彩色高）
    float depthScale() const { return m_depthScale; }  ///< 深度单位：毫米/原始值（1.0 即原始值=毫米）
    bool alignedToColor() const { return m_aligned; }  ///< 彩色/深度是否已逐像素对齐（同分辨率）
    std::string alignMode() const { return m_alignName; } ///< 对齐方式：HW / SW(Align滤镜) / OFF

    /// 深度伪彩可视化（无效像素置黑）
    static cv::Mat depthToColor(const cv::Mat &depthMM, float minMM = 300.0f, float maxMM = 3000.0f);

    /// 针对 depthMM 中有效区域的 5%~95% 分位自动拉伸范围的伪彩
    static cv::Mat depthToColorAuto(const cv::Mat &depthMM);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
    bool m_open{false};
    bool m_hasDepth{false};
    int m_w{0};
    int m_h{0};
    int m_depthW{0};
    int m_depthH{0};
    float m_depthScale{1.0f};
    bool m_aligned{false};
    std::string m_alignName{"OFF"};
    std::string m_err;
    std::string m_devName;
};

#endif // ORBBECSOURCE_H
