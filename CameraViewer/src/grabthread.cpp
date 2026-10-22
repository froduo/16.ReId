#include "grabthread.h"
#include "logger.h"
#include <QImage>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QCoreApplication>
#include <exception>

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

void GrabThread::setPersonClipRecording(bool enabled, const QString &dir, int gapMs, int fps)
{
    m_clipEnabled = enabled;
    m_clipDir = dir;
    m_clipGapMs = qMax(500, gapMs);
    if (fps > 1) m_clipFps = fps;

    LOG_INFO(QString("[GrabThread-%1] 人物片段录制: %2, 目录=%3, 无人间隔=%4ms, 起始fps=%5")
                 .arg(m_channelName)
                 .arg(enabled ? "启用" : "关闭")
                 .arg(dir).arg(m_clipGapMs).arg(m_clipFps, 0, 'f', 1));

    // 关闭录制功能时，收尾正在写出的片段
    if (!enabled && m_clipActive) closePersonClip();
}

void GrabThread::updatePersonClip(const cv::Mat &frame, bool personPresent)
{
    if (frame.empty()) return;

    if (!personPresent) {
        if (m_clipActive) closePersonClip();
        return;
    }

    // ===== 开启新片段 =====
    if (!m_clipActive || !m_clipWriter.isOpened()) {
        QString dir = m_clipDir;
        if (dir.isEmpty()) dir = QStringLiteral("./save_images/clips");
        if (!QDir(dir).isAbsolute()) {
            dir = QCoreApplication::applicationDirPath() + "/" + dir;
        }
        if (!QDir().mkpath(dir)) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                LOG_WARN("[GrabThread] 无法创建人物片段目录: " + dir);
            }
            return;
        }

        m_clipStart = QDateTime::currentDateTime();
        const QString baseName = QString("人物片段_%1").arg(m_clipStart.toString("yyyyMMdd_HHmmss"));

        // 优先 mp4(mp4v)，失败回退 avi(MJPG)
        QString path = QDir(dir).filePath(baseName + ".mp4");
        m_clipWriter.open(path.toStdString(),
                          cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                          m_clipFps, frame.size(), true);
        if (!m_clipWriter.isOpened()) {
            path = QDir(dir).filePath(baseName + ".avi");
            m_clipWriter.open(path.toStdString(),
                              cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                              m_clipFps, frame.size(), true);
        }
        if (!m_clipWriter.isOpened()) {
            LOG_WARN("[GrabThread] 人物片段写入器打开失败(编码不可用): " + path);
            m_clipActive = false;
            return;
        }

        m_clipFilePath = path;
        m_clipFrameSize = frame.size();
        m_clipFrames = 0;
        m_clipActive = true;

        LOG_INFO(QString("[GrabThread-%1] 开始录制人物片段: %2 (fps=%3, %4x%5)")
                     .arg(m_channelName, path).arg(m_clipFps, 0, 'f', 1)
                     .arg(frame.cols).arg(frame.rows));
    }

    // 分辨率变化时结束旧片段（下一帧按新尺寸自动重开）
    if (frame.size() != m_clipFrameSize) {
        LOG_WARN(QString("[GrabThread-%1] 人物片段分辨率变化 %2x%3 → %4x%5，结束当前片段")
                     .arg(m_channelName).arg(m_clipFrameSize.width).arg(m_clipFrameSize.height)
                     .arg(frame.cols).arg(frame.rows));
        closePersonClip();
        m_lastPersonMs = QDateTime::currentMSecsSinceEpoch();
        updatePersonClip(frame, true);   // 立即以新尺寸重开
        return;
    }

    m_clipWriter.write(frame);
    m_clipFrames++;
}

void GrabThread::closePersonClip()
{
    if (!m_clipActive) return;
    m_clipActive = false;

    if (m_clipWriter.isOpened()) m_clipWriter.release();
    if (m_clipFilePath.isEmpty() || m_clipFrames <= 0) {
        m_clipFilePath.clear();
        m_clipFrames = 0;
        return;
    }

    const QDateTime end = QDateTime::currentDateTime();
    const qint64 durMs = qMax<qint64>(1, m_clipStart.msecsTo(end));
    const double actualFps = m_clipFrames * 1000.0 / static_cast<double>(durMs);
    // 自校准：用实测写入速率作为下一段帧率，避免视频被加速/拖慢
    if (actualFps > 1.0 && actualFps < 120.0) m_clipFps = actualFps;

    // 文件名补上结束时间：人物片段_yyyyMMdd_HHmmss-HHmmss.mp4
    const QFileInfo fi(m_clipFilePath);
    const QString newBase = QString("人物片段_%1-%2")
                                .arg(m_clipStart.toString("yyyyMMdd_HHmmss"))
                                .arg(end.toString("HHmmss"));
    const QString newPath = QDir(fi.absolutePath()).filePath(newBase + "." + fi.suffix());

    QString finalPath = m_clipFilePath;
    if (newPath != m_clipFilePath) {
        if (QFile::exists(newPath)) QFile::remove(newPath);
        if (QFile::rename(m_clipFilePath, newPath)) finalPath = newPath;
    }

    LOG_INFO(QString("[GrabThread-%1] 人物片段已保存: %2 (帧数=%3, 时长=%4ms, 实测fps=%5)")
                 .arg(m_channelName, finalPath).arg(m_clipFrames).arg(durMs)
                 .arg(actualFps, 0, 'f', 1));

    m_clipFilePath.clear();
    m_clipFrames = 0;
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
                if (m_camera->isVideoFile()) {
                    LOG_INFO(QString("[GrabThread-%1] 视频文件读取完毕，共 %2 帧")
                                 .arg(m_channelName).arg(frameIndex));
                    emit videoFinished();
                } else {
                    LOG_ERR(QString("[GrabThread-%1] MAX_FAIL_COUNT reached, emitting deviceLost")
                            .arg(m_channelName));
                    emit deviceLost();
                }
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

        // 发出原始 BGR 帧信号（用于图像保存/绘制）。
        // 该信号会带来整帧深拷贝(约 6MB)，接收端还会再拷贝一次；若每帧发送会严重
        // 拖慢采集线程并造成延时累积。这里降频为每 5 帧发一次（手动保存最多滞后 ~200ms）。
        if (frameIndex % 5 == 0) {
            emit rawFrameGrabbed(frame.clone());
        }

        // 背景模型更新（固定摄像头：仅在画面基本无人时学习背景）
        if (m_bgModel) {
            m_bgModel->update(frame);
        }

        // 检测管线：如果设置了检测器，则执行检测和跟踪
        // 用 try/catch 兜底：任何 OpenCV/标准异常都不得逃出线程函数，
        // 否则会触发 std::terminate 使整个程序崩溃。
        try {
        if (m_detector && m_detector->isLoaded() && m_tracker) {
            // 耗时埋点：定位瓶颈在「检测」还是「跟踪+ReID」
            const qint64 t0 = QDateTime::currentMSecsSinceEpoch();
            QVector<cv::Rect> detections = m_detector->detect(frame);
            const qint64 tDetect = QDateTime::currentMSecsSinceEpoch() - t0;

            const qint64 t1 = QDateTime::currentMSecsSinceEpoch();
            // 将 frame 和 ReID 提取器传递给 tracker，由 tracker 内部完成
            // 卡曼预测 + IoU + ReID 多阶段匹配
            QVector<TrackedPerson> persons = m_tracker->update(detections, frame, m_reidExtractor);
            const qint64 tTrack = QDateTime::currentMSecsSinceEpoch() - t1;

            // 每 5 秒打印一次耗时，便于评估「换模型/调参」的效果
            static qint64 s_lastPerfMs = 0;
            const qint64 perfNow = QDateTime::currentMSecsSinceEpoch();
            if (perfNow - s_lastPerfMs > 5000) {
                s_lastPerfMs = perfNow;
                LOG_INFO(QString("[Perf] detect=%1ms, track+reid=%2ms, dets=%3, persons=%4")
                             .arg(tDetect).arg(tTrack).arg(detections.size()).arg(persons.size()));
            }

            m_lastTrackedPersons = persons;
            // 记录“最近检测到人”的时刻（人物片段录制的驱动信号）
            if (!persons.isEmpty()) {
                m_lastPersonMs = QDateTime::currentMSecsSinceEpoch();
            }

            // 将 BGR 转换为 RGB QImage
            cv::Mat rgbFrame;
            cv::cvtColor(frame, rgbFrame, cv::COLOR_BGR2RGB);
            QImage img(rgbFrame.data, rgbFrame.cols, rgbFrame.rows,
                       static_cast<int>(rgbFrame.step), QImage::Format_RGB888);

            // 发出处理后的帧和跟踪结果信号（frameProcessed 内部已处理显示和保存）
            // 背压控制：UI 尚未消费完上一帧时，丢弃本帧“显示”更新（检测已完成，不影响实时性），
            // 从而避免显示队列无限积压、画面越来越滞后。
            if (!m_uiBusy.load()) {
                m_uiBusy.store(true);
                emit frameProcessed(img.copy(), persons);
            }
        } else {
            // 无检测管线时，仍然发出 frameProcessed（空 persons 列表）
            // 这样区域叠加和 UI 更新仍然正常工作
            cv::Mat rgbFrame;
            cv::cvtColor(frame, rgbFrame, cv::COLOR_BGR2RGB);
            QImage img(rgbFrame.data, rgbFrame.cols, rgbFrame.rows,
                       static_cast<int>(rgbFrame.step), QImage::Format_RGB888);
            if (!m_uiBusy.load()) {
                m_uiBusy.store(true);
                emit frameProcessed(img.copy(), QVector<TrackedPerson>());
            }
        }
        } catch (const cv::Exception &e) {
            LOG_ERR(QString("[GrabThread-%1] cv::Exception in pipeline: %2")
                        .arg(m_channelName, QString::fromStdString(e.what())));
        } catch (const std::exception &e) {
            LOG_ERR(QString("[GrabThread-%1] std::exception in pipeline: %2")
                        .arg(m_channelName, QString::fromUtf8(e.what())));
        }

        // ===== 人物片段录制 =====
        // 每帧写入以保证画面流畅；“有人”状态由最近一次检测结果驱动
        // （检测降频时用 gapMs 保持窗口继续录制，避免频繁切段）。
        if (m_clipEnabled) {
            const qint64 nowMs = QDateTime::currentMSecsSinceEpoch();
            const bool personPresent =
                (m_lastPersonMs > 0) && ((nowMs - m_lastPersonMs) <= m_clipGapMs);
            updatePersonClip(frame, personPresent);
        }
    }

    // 线程退出：收尾当前片段，保证文件可播放
    closePersonClip();

    LOG_INFO(QString("[GrabThread-%1] Exited, total frames: %2")
             .arg(m_channelName).arg(frameIndex));
}
