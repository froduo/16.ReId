#include "cameracontroller.h"
#include "orbbecsource.h"
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

    if (!m_cap.open(rtspUrl.toStdString(), cv::CAP_FFMPEG)) {
        LOG_ERR(QString("Failed to open RTSP stream: %1").arg(rtspUrl));
        m_lastError = "RTSP open failed";
        return false;
    }

    // 关键：CAP_PROP_BUFFERSIZE 必须在 open() 之后设置才生效
    m_cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

    m_frameWidth  = static_cast<int>(m_cap.get(cv::CAP_PROP_FRAME_WIDTH));
    m_frameHeight = static_cast<int>(m_cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    m_fps         = m_cap.get(cv::CAP_PROP_FPS);
    if (m_fps <= 0) m_fps = 25.0;

    m_isOrbbec = false;
    m_hasDepth = false;
    m_devName = "RTSP";

    LOG_INFO(QString("RTSP stream opened: %1x%2 @ %3 fps")
             .arg(m_frameWidth).arg(m_frameHeight).arg(m_fps));
    return true;
}

bool CameraController::openOrbbec(int colorWidth, int colorHeight, int fps)
{
    closeCamera();

    m_orbbec = std::make_shared<OrbbecSource>();
    if (!m_orbbec->open(colorWidth, colorHeight, fps)) {
        m_lastError = QString::fromStdString(m_orbbec->lastError());
        LOG_ERR(QString("[Orbbec] 打开失败: %1").arg(m_lastError));
        m_orbbec.reset();
        return false;
    }

    m_isOrbbec  = true;
    m_hasDepth  = m_orbbec->hasDepth();
    m_frameWidth  = m_orbbec->width();
    m_frameHeight = m_orbbec->height();
    m_devName   = QString::fromStdString(m_orbbec->deviceName());
    m_fps       = 30.0;

    LOG_INFO(QString("[Orbbec] 已打开: %1 | 彩色 %2x%3 | 深度 %4x%5 | 对齐=%6 | "
                     "depthScale=%7mm | depth=%8")
             .arg(m_devName).arg(m_frameWidth).arg(m_frameHeight)
             .arg(m_orbbec->depthWidth()).arg(m_orbbec->depthHeight())
             .arg(QString::fromStdString(m_orbbec->alignMode()))
             .arg(m_orbbec->depthScale(), 0, 'f', 3)
             .arg(m_hasDepth ? "on" : "off"));
    return true;
}

bool CameraController::getFrame(cv::Mat &frame)
{
    if (m_isOrbbec) {
        cv::Mat depth;
        return grabFrames(frame, depth);
    }
    if (!m_cap.isOpened())
        return false;
    if (!m_cap.read(frame))
        return false;
    return !frame.empty();
}

bool CameraController::grabFrames(cv::Mat &colorBGR, cv::Mat &depthMM)
{
    depthMM.release();
    if (m_isOrbbec) {
        if (!m_orbbec)
            return false;
        uint64_t cts = 0, dts = 0;
        return m_orbbec->getFrames(colorBGR, depthMM, cts, dts, 500);
    }
    return getFrame(colorBGR);
}

void CameraController::closeCamera()
{
    if (m_orbbec) {
        m_orbbec->close();
        m_orbbec.reset();
        LOG_INFO("Orbbec camera closed");
    }
    if (m_cap.isOpened()) {
        m_cap.release();
        LOG_INFO("RTSP stream closed");
    }
    m_frameWidth  = 0;
    m_frameHeight = 0;
    m_fps         = 0.0;
    m_isOrbbec    = false;
    m_hasDepth    = false;
    m_devName.clear();
}

bool CameraController::isOpen() const
{
    if (m_isOrbbec)
        return m_orbbec && m_orbbec->isOpen();
    return m_cap.isOpened();
}
