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

    // 设置 FFmpeg 缓冲区参数，减少延迟
    m_cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    // 设置传输协议为 TCP (更稳定)
    m_cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('H', '2', '6', '4'));

    if (!m_cap.open(rtspUrl.toStdString(), cv::CAP_FFMPEG)) {
        LOG_ERR(QString("Failed to open RTSP stream: %1").arg(rtspUrl));
        return false;
    }

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
