#include "grabthread.h"
#include "logger.h"
#include <QImage>

GrabThread::GrabThread(CameraController* cam, const QString &channelName, QObject *parent)
    : QThread(parent)
    , m_camera(cam)
    , m_channelName(channelName)
{
    m_running = true;
    m_consecutiveFailCount = 0;
}

void GrabThread::stop()
{
    m_running = false;
}

void GrabThread::setDetector(PersonDetector *detector)
{
    m_detector = detector;
}

void GrabThread::setTracker(PersonTracker *tracker)
{
    m_tracker = tracker;
}

void GrabThread::setReidExtractor(ReIDExtractor *extractor)
{
    m_reidExtractor = extractor;
}

void GrabThread::run()
{
    cv::Mat frame;
    int frameIndex = 0;
    QElapsedTimer frameTimer;
    frameTimer.start();

    LOG_INFO(QString("[GrabThread-%1] Started grabbing frames").arg(m_channelName));

    while (m_running)
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
                LOG_ERR(QString("[GrabThread-%1] MAX_FAIL_COUNT reached, emitting deviceLost")
                        .arg(m_channelName));
                emit deviceLost();
                m_running = false;
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

        // 发出原始 BGR 帧信号（用于图像保存）
        // 显式 deep clone，避免接收端读取到被后续 getFrame 覆盖的数据
        emit rawFrameGrabbed(frame.clone());

        // 检测管线：如果设置了检测器，则执行检测和跟踪
        if (m_detector && m_detector->isLoaded() && m_tracker) {
            QVector<cv::Rect> detections = m_detector->detect(frame);
            // 将 frame 和 ReID 提取器传递给 tracker，由 tracker 内部完成
            // 卡曼预测 + IoU + ReID 多阶段匹配
            QVector<TrackedPerson> persons = m_tracker->update(detections, frame, m_reidExtractor);

            m_lastTrackedPersons = persons;

            // 将 BGR 转换为 RGB QImage
            cv::Mat rgbFrame;
            cv::cvtColor(frame, rgbFrame, cv::COLOR_BGR2RGB);
            QImage img(rgbFrame.data, rgbFrame.cols, rgbFrame.rows,
                       static_cast<int>(rgbFrame.step), QImage::Format_RGB888);

            // 发出处理后的帧和跟踪结果信号（frameProcessed 内部已处理显示和保存）
            emit frameProcessed(img.copy(), persons);
        } else {
            // 无检测管线时，仍然发出 frameProcessed（空 persons 列表）
            // 这样区域叠加和 UI 更新仍然正常工作
            cv::Mat rgbFrame;
            cv::cvtColor(frame, rgbFrame, cv::COLOR_BGR2RGB);
            QImage img(rgbFrame.data, rgbFrame.cols, rgbFrame.rows,
                       static_cast<int>(rgbFrame.step), QImage::Format_RGB888);
            emit frameProcessed(img.copy(), QVector<TrackedPerson>());
        }
    }

    LOG_INFO(QString("[GrabThread-%1] Exited, total frames: %2")
             .arg(m_channelName).arg(frameIndex));
}
