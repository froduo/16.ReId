#include "cameracontroller.h"
#include "logger.h"
#include <QDebug>

CameraController::CameraController(QObject *parent)
    : QObject(parent)
{
}

CameraController::~CameraController()
{
    closeCamera();
}

bool CameraController::openCamera(const QString &rtspUrl)
{
    closeCamera();

    m_rtspUrl = rtspUrl;

    LOG_INFO(QString("Opening RTSP stream: %1").arg(rtspUrl));

    // 打开 RTSP 流（默认使用 FFmpeg 后端）
    if (!m_cap.open(rtspUrl.toStdString(), cv::CAP_FFMPEG)) {
        LOG_ERR(QString("Failed to open RTSP stream: %1").arg(rtspUrl));
        return false;
    }

    // 关键：CAP_PROP_BUFFERSIZE 必须在 open() 之后设置才生效；
    // open() 之前设置是无效的，会导致 RTSP/FFmpeg 内部缓存大量帧，
    // 从而出现“画面与检测严重滞后（可达 1 分钟）”的问题。
    m_cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

    m_frameWidth  = static_cast<int>(m_cap.get(cv::CAP_PROP_FRAME_WIDTH));
    m_frameHeight = static_cast<int>(m_cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    m_fps         = m_cap.get(cv::CAP_PROP_FPS);

    if (m_fps <= 0) m_fps = 25.0; // 设置默认 FPS

    qDebug() << "RTSP stream opened:"
             << m_frameWidth << "x" << m_frameHeight
             << "@" << m_fps << "fps";

    LOG_INFO(QString("RTSP stream opened: %1x%2 @ %3 fps")
             .arg(m_frameWidth).arg(m_frameHeight).arg(m_fps));

    return true;
}

bool CameraController::getFrame(cv::Mat &frame)
{
    if (!m_cap.isOpened()) {
        return false;
    }

    if (!m_cap.read(frame)) {
        return false;
    }

    if (frame.empty()) {
        return false;
    }

    return true;
}

void CameraController::closeCamera()
{
    if (m_cap.isOpened()) {
        m_cap.release();
        LOG_INFO("RTSP stream closed");
    }
    m_frameWidth  = 0;
    m_frameHeight = 0;
    m_fps         = 0.0;
}

bool CameraController::isOpen() const
{
    return m_cap.isOpened();
}
