#ifndef CAMERACONTROLLER_H
#define CAMERACONTROLLER_H

#include <QObject>
#include <QString>
#include <memory>
#include <opencv2/opencv.hpp>

class OrbbecSource;

/**
 * @brief 相机控制器：支持两种帧源
 *  1) RTSP 网络相机（OpenCV VideoCapture）—— 原 HX-CCD23 方案
 *  2) Orbbec Gemini 335 深度相机（OrbbecSDK v2）—— 车门正上方俯视，输出 RGB + Depth
 *
 * 统一对外提供 getFrame()/grabFrames()，GrabThread 无需区分来源。
 */
class CameraController : public QObject
{
    Q_OBJECT
public:
    explicit CameraController(QObject *parent = nullptr);
    ~CameraController();

    /** 打开 RTSP 流 */
    bool openCamera(const QString &rtspUrl);

    /**
     * @brief 打开 Orbbec 深度相机
     * @param colorWidth/colorHeight/fps 0 表示默认 profile
     */
    bool openOrbbec(int colorWidth = 0, int colorHeight = 0, int fps = 0);

    /** 获取一帧彩色图像（BGR） */
    bool getFrame(cv::Mat &frame);

    /** 统一取帧：彩色(BGR, CV_8UC3) + 深度(CV_16U, 毫米；非 Orbbec 时为空) */
    bool grabFrames(cv::Mat &colorBGR, cv::Mat &depthMM);

    void closeCamera();
    bool isOpen() const;

    bool isOrbbec() const { return m_isOrbbec; }
    bool hasDepth() const { return m_hasDepth; }

    QString rtspUrl() const { return m_rtspUrl; }
    QString deviceName() const { return m_devName; }
    QString lastError() const { return m_lastError; }

    int frameWidth() const { return m_frameWidth; }
    int frameHeight() const { return m_frameHeight; }
    double fps() const { return m_fps; }

signals:
    void deviceDisconnected();

private:
    // RTSP
    cv::VideoCapture m_cap;
    QString m_rtspUrl;
    int m_frameWidth = 0;
    int m_frameHeight = 0;
    double m_fps = 0.0;

    // Orbbec
    std::shared_ptr<OrbbecSource> m_orbbec;
    bool m_isOrbbec = false;
    bool m_hasDepth = false;
    QString m_devName;
    QString m_lastError;
};

#endif // CAMERACONTROLLER_H
