#ifndef CAMERACONTROLLER_H
#define CAMERACONTROLLER_H

#include <QObject>
#include <QString>
#include <opencv2/opencv.hpp>

/**
 * @brief RTSP 相机控制器，基于 OpenCV VideoCapture
 *
 * 用 OpenCV VideoCapture 连接 HX-CCD23 相机的 RTSP 流。
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
    cv::VideoCapture m_cap;
    QString m_rtspUrl;
    int m_frameWidth = 0;
    int m_frameHeight = 0;
    double m_fps = 0.0;
};

#endif // CAMERACONTROLLER_H
