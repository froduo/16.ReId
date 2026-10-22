#ifndef CAMERACONTROLLER_H
#define CAMERACONTROLLER_H

#include <QObject>
#include <QString>
#include <opencv2/opencv.hpp>

/**
 * @brief 相机控制器：RGB 主路（RTSP 或本地视频文件）+ 可选深度流
 *
 *  - openCamera(rgbUrl, depthUrl)：拉取两路 RTSP（深度流可选，失败自动降级为仅 RGB）
 *  - openVideoFile(path)：本地视频文件检测（无深度）
 */
class CameraController : public QObject
{
    Q_OBJECT
public:
    explicit CameraController(QObject *parent = nullptr);
    ~CameraController();

    bool openCamera(const QString &rtspUrl, const QString &depthRtspUrl = QString());
    bool openVideoFile(const QString &path);
    bool isVideoFile() const { return m_isVideoFile; }

    bool getFrame(cv::Mat &frame);
    bool getDepthFrame(cv::Mat &frame);
    /// 统一取帧：RGB 主路 + 深度（深度失败不影响 RGB 主路）
    bool grabFrames(cv::Mat &colorBGR, cv::Mat &depth);

    void closeCamera();
    bool isOpen() const;

    bool hasDepth() const { return m_hasDepth; }
    QString rtspUrl() const { return m_rtspUrl; }
    QString depthRtspUrl() const { return m_depthRtspUrl; }

    int frameWidth() const { return m_frameWidth; }
    int frameHeight() const { return m_frameHeight; }
    double fps() const { return m_fps; }

signals:
    void deviceDisconnected();

private:
    cv::VideoCapture m_cap;       // RGB 主路
    cv::VideoCapture m_depthCap;  // 深度流（可选）
    QString m_rtspUrl;
    QString m_depthRtspUrl;
    bool m_isVideoFile{false};
    bool m_hasDepth{false};
    int m_frameWidth = 0;
    int m_frameHeight = 0;
    double m_fps = 0.0;
};

#endif // CAMERACONTROLLER_H
