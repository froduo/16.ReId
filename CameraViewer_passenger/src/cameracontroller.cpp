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

    LOG_INFO(QString("Opening RTSP stream: %1").arg(rtspUrl));

    // 设置 FFmpeg 超时：RTSP 网络中断时 m_cap.read() 会长时间阻塞，
    // 导致采集线程卡死、画面与检测停摆（进而漏检乘客）。
    //   stimeout = RTSP 会话读超时（微秒），2s 内读不到数据即返回失败
    //   timeout  = socket I/O 超时（微秒），限制 connect/read 阻塞，3s 快速失败
    // 两者配合让 openCamera 在相机掉线时 3 秒内快速返回失败，配合 GrabThread
    // 的连续失败计数快速触发重连（否则默认 30 秒超时，画面冻结半分钟）。
    // 传输方式用 TCP（本相机 RTSP 必须用 TCP 交错传输，UDP 会连接超时。
    // 实测：系统 ffmpeg -rtsp_transport tcp 能稳定拉流，udp 超时失败）。
    qputenv("OPENCV_FFMPEG_CAPTURE_OPTIONS",
            "rtsp_transport;tcp|stimeout;2000000|timeout;3000000");

    // 打开 RTSP 流（默认使用 FFmpeg 后端）。
    // 设置打开超时：相机掉线/不可达时，ffmpeg 连接会长时间阻塞；
    // 而自动重连在 UI 线程执行，若不加超时会导致整个界面“点了没反应”。
    m_cap.set(cv::CAP_PROP_OPEN_TIMEOUT_MSEC, 3000);
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
