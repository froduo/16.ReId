#include "grabthread.h"
#include "logger.h"
#include <QImage>
#include <QElapsedTimer>

GrabThread::GrabThread(CameraController *cam, LatestFrameSlot *slot,
                       const QString &channelName, QObject *parent)
    : QThread(parent)
    , m_camera(cam)
    , m_slot(slot)
    , m_channelName(channelName)
{
    m_consecutiveFailCount = 0;
}

void GrabThread::run()
{
    cv::Mat frame;
    int frameIndex = 0;
    QElapsedTimer frameTimer;
    frameTimer.start();

    LOG_INFO(QString("[GrabThread-%1] Started grabbing RGB frames").arg(m_channelName));

    while (m_running.load())
    {
        if (!m_camera->getFrame(frame)) {
            m_consecutiveFailCount++;
            if (m_consecutiveFailCount == 1) {
                LOG_WARN(QString("[GrabThread-%1] First frame grab failed, consecutive: %2/%3")
                         .arg(m_channelName).arg(m_consecutiveFailCount).arg(MAX_FAIL_COUNT));
            } else if (m_consecutiveFailCount % 50 == 0) {
                LOG_WARN(QString("[GrabThread-%1] Consecutive failures: %2/%3")
                         .arg(m_channelName).arg(m_consecutiveFailCount).arg(MAX_FAIL_COUNT));
            }

            if (m_consecutiveFailCount >= MAX_FAIL_COUNT) {
                if (m_camera->isVideoFile()) {
                    LOG_INFO(QString("[GrabThread-%1] 视频文件读取完毕，共 %2 帧")
                                 .arg(m_channelName).arg(frameIndex));
                    emit videoFinished();
                } else {
                    LOG_ERR(QString("[GrabThread-%1] MAX_FAIL_COUNT reached, emitting deviceLost")
                            .arg(m_channelName));
                    emit deviceLost();
                }
                m_running.store(false);
                break;
            }
            QThread::msleep(50);
            continue;
        }

        // 成功获取帧，重置失败计数
        if (m_consecutiveFailCount > 0) {
            LOG_INFO(QString("[GrabThread-%1] Frame grab recovered after %2 failures")
                     .arg(m_channelName).arg(m_consecutiveFailCount));
            m_consecutiveFailCount = 0;
        }

        frameIndex++;

        // 发出原始 RGB QImage（用于显示/FPS/手动保存）
        {
            cv::Mat rgb;
            cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
            QImage img(rgb.data, rgb.cols, rgb.rows,
                       static_cast<int>(rgb.step), QImage::Format_RGB888);
            emit frameGrabbed(img.copy());
        }

        // 投递最新帧给检测线程（慢时自动丢帧，只处理最新一帧）
        m_slot->publish(frame);

        // 帧率日志（降低频率）
        if (frameIndex % 500 == 0) {
            qint64 elapsed = frameTimer.elapsed();
            LOG_INFO(QString("[GrabThread-%1] Frame #%2, size=%3x%4, interval=%5ms")
                     .arg(m_channelName)
                     .arg(frameIndex)
                     .arg(frame.cols).arg(frame.rows)
                     .arg(elapsed));
            frameTimer.restart();
        }
    }

    LOG_INFO(QString("[GrabThread-%1] Exited, total frames: %2")
             .arg(m_channelName).arg(frameIndex));
}
