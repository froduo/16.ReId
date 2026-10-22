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

bool CameraController::openCamera(const QString &rtspUrl, const QString &depthRtspUrl)
{
    closeCamera();

    m_rtspUrl = rtspUrl;
    m_depthRtspUrl = depthRtspUrl;
    m_isVideoFile = false;

    LOG_INFO(QString("Opening RTSP stream: %1").arg(rtspUrl));

    // 设置 FFmpeg 读取超时：RTSP 网络中断时 m_cap.read() 会长时间阻塞，
    // 导致采集线程卡死、画面与检测停摆（进而漏检乘客）。设置 stimeout 让
    // read 在超时后返回 false，配合 GrabThread 的连续失败计数快速触发重连。
    qputenv("OPENCV_FFMPEG_CAPTURE_OPTIONS",
            "rtsp_transport;tcp|stimeout;5000000");

    m_cap.set(cv::CAP_PROP_OPEN_TIMEOUT_MSEC, 3000);
    if (!m_cap.open(rtspUrl.toStdString(), cv::CAP_FFMPEG)) {
        LOG_ERR(QString("Failed to open RTSP stream: %1").arg(rtspUrl));
        return false;
    }

    m_cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

    m_frameWidth  = static_cast<int>(m_cap.get(cv::CAP_PROP_FRAME_WIDTH));
    m_frameHeight = static_cast<int>(m_cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    m_fps         = m_cap.get(cv::CAP_PROP_FPS);
    if (m_fps <= 0) m_fps = 25.0;

    LOG_INFO(QString("RTSP stream opened: %1x%2 @ %3 fps")
             .arg(m_frameWidth).arg(m_frameHeight).arg(m_fps));

    // ===== 深度流（可选）：拉流失败不影响 RGB 主路 =====
    if (!depthRtspUrl.isEmpty()) {
        LOG_INFO(QString("Opening depth stream: %1").arg(depthRtspUrl));
        m_depthCap.set(cv::CAP_PROP_OPEN_TIMEOUT_MSEC, 3000);
        if (m_depthCap.open(depthRtspUrl.toStdString(), cv::CAP_FFMPEG)) {
            m_depthCap.set(cv::CAP_PROP_BUFFERSIZE, 1);
            m_hasDepth = true;
            LOG_INFO(QString("Depth stream opened: %1x%2")
                     .arg(static_cast<int>(m_depthCap.get(cv::CAP_PROP_FRAME_WIDTH)))
                     .arg(static_cast<int>(m_depthCap.get(cv::CAP_PROP_FRAME_HEIGHT))));
        } else {
            m_depthCap.release();
            m_hasDepth = false;
            LOG_WARN(QString("Failed to open depth stream (will run RGB-only): %1").arg(depthRtspUrl));
        }
    }

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
    frame.release();
    if (!m_cap.isOpened()) return false;
    if (!m_cap.read(frame)) return false;
    return !frame.empty();
}

bool CameraController::getDepthFrame(cv::Mat &frame)
{
    frame.release();
    if (!m_hasDepth || !m_depthCap.isOpened()) return false;
    if (!m_depthCap.read(frame)) return false;
    return !frame.empty();
}

bool CameraController::grabFrames(cv::Mat &colorBGR, cv::Mat &depth)
{
    colorBGR.release();
    depth.release();
    if (!getFrame(colorBGR)) return false;
    if (m_hasDepth) getDepthFrame(depth);   // 深度失败不影响 RGB 主路
    return !colorBGR.empty();
}

void CameraController::closeCamera()
{
    if (m_cap.isOpened()) {
        m_cap.release();
        LOG_INFO("RTSP stream closed");
    }
    if (m_depthCap.isOpened()) {
        m_depthCap.release();
        LOG_INFO("Depth stream closed");
    }
    m_frameWidth  = 0;
    m_frameHeight = 0;
    m_fps         = 0.0;
    m_hasDepth    = false;
    m_isVideoFile = false;
}

bool CameraController::isOpen() const
{
    return m_cap.isOpened();
}
