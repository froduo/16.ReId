#ifndef CAMERACONTROLLER_H
#define CAMERACONTROLLER_H

#include <QObject>
#include <QString>
#include <opencv2/opencv.hpp>
#include "ffmpegcapture.h"

/**
 * @brief 相机控制器
 *
 * RTSP 实时流用 FFmpeg 直接拉流（FfmpegCapture，坏帧 conceal 容错，与 VLC 对齐）；
 * 本地视频文件用 OpenCV VideoCapture（离线检测/倍速播放）。
 */
class CameraController : public QObject
{
    Q_OBJECT
public:
    explicit CameraController(QObject *parent = nullptr);
    ~CameraController();

    /**
     * @brief 打开 RTSP 流
     * @param rtspUrl 完整的 RTSP URL
     * @return 成功返回 true
     */
    bool openCamera(const QString &rtspUrl);

    /**
     * @brief 打开本地视频文件（用于「选择视频」离线检测）
     * @param path 本地视频文件路径
     * @return 成功返回 true
     */
    bool openVideoFile(const QString &path);

    /// 当前是否打开的是本地视频文件（而非 RTSP 流）
    bool isVideoFile() const { return m_isVideoFile; }

    /**
     * @brief 获取一帧图像
     * @param frame 输出的 cv::Mat 帧（BGR 格式）
     * @return 成功返回 true
     */
    bool getFrame(cv::Mat &frame);

    /**
     * @brief 关闭 RTSP 流
     */
    void closeCamera();

    /**
     * @brief 检查相机是否已打开
     */
    bool isOpen() const;

    /**
     * @brief 获取当前 RTSP URL
     */
    QString rtspUrl() const { return m_rtspUrl; }

    /**
     * @brief 获取帧宽
     */
    int frameWidth() const { return m_frameWidth; }

    /**
     * @brief 获取帧高
     */
    int frameHeight() const { return m_frameHeight; }

    /**
     * @brief 获取当前 FPS
     */
    double fps() const { return m_fps; }

signals:
    void deviceDisconnected();

private:
    cv::VideoCapture m_cap;      // 本地视频文件（OpenCV）
    FfmpegCapture m_ffCap;       // RTSP 实时流（FFmpeg 直接拉流，坏帧容错）
    QString m_rtspUrl;
    bool m_isVideoFile{false};
    int m_frameWidth = 0;
    int m_frameHeight = 0;
    double m_fps = 0.0;
};

#endif // CAMERACONTROLLER_H
