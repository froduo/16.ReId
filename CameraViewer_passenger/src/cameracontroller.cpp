#include "cameracontroller.h"
#include "logger.h"
#include <QDebug>
#include <QtGlobal>

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
    m_isVideoFile = false;

    LOG_INFO(QString("Opening RTSP stream (FFmpeg): %1").arg(rtspUrl));

    // 用 FFmpeg 直接拉流（FfmpegCapture），内部已设置：
    //   rtsp_transport=tcp / stimeout=6s / timeout=3s，并在解码器层 err_detect=ignore_err
    // 实现坏帧 conceal 容错（与 VLC 对齐），避免 OpenCV read 遇坏帧立即失败导致误判掉线。
    if (!m_ffCap.open(rtspUrl)) {
        LOG_ERR(QString("Failed to open RTSP stream: %1").arg(rtspUrl));
        return false;
    }

    m_frameWidth  = m_ffCap.width();
    m_frameHeight = m_ffCap.height();
    m_fps         = m_ffCap.fps();

    LOG_INFO(QString("RTSP stream opened: %1x%2 @ %3 fps")
             .arg(m_frameWidth).arg(m_frameHeight).arg(m_fps));

    return true;
}

bool CameraController::openVideoFile(const QString &path)
{
    closeCamera();

    m_rtspUrl = path;
    m_isVideoFile = true;

    LOG_INFO(QString("Opening video file: %1").arg(path));

    if (!m_cap.open(path.toStdString(), cv::CAP_FFMPEG)) {
        LOG_ERR(QString("Failed to open video file: %1").arg(path));
        return false;
    }

    m_frameWidth  = static_cast<int>(m_cap.get(cv::CAP_PROP_FRAME_WIDTH));
    m_frameHeight = static_cast<int>(m_cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    m_fps         = m_cap.get(cv::CAP_PROP_FPS);
    if (m_fps <= 0) m_fps = 25.0;

    LOG_INFO(QString("Video file opened: %1x%2 @ %3 fps")
                 .arg(m_frameWidth).arg(m_frameHeight).arg(m_fps));
    return true;
}

bool CameraController::getFrame(cv::Mat &frame)
{
    if (m_isVideoFile) {
        if (!m_cap.isOpened()) return false;
        if (!m_cap.read(frame)) return false;
        if (frame.empty()) return false;
        return true;
    }

    if (!m_ffCap.isOpened()) return false;
    if (!m_ffCap.read(frame)) return false;
    if (frame.empty()) return false;
    return true;
}

void CameraController::closeCamera()
{
    if (m_cap.isOpened()) {
        m_cap.release();
        LOG_INFO("Video file stream closed");
    }
    if (m_ffCap.isOpened()) {
        m_ffCap.close();
        LOG_INFO("RTSP stream closed");
    }
    m_frameWidth  = 0;
    m_frameHeight = 0;
    m_fps         = 0.0;
}

bool CameraController::isOpen() const
{
    return m_isVideoFile ? m_cap.isOpened() : m_ffCap.isOpened();
}
