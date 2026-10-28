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
    m_paused.store(false);   // 停止时退出暂停，避免线程 wait 阻塞
}

void GrabThread::pause()
{
    m_paused.store(true);
}

void GrabThread::resume()
{
    m_paused.store(false);
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

void GrabThread::setVideoSpeed(double speed)
{
    if (speed < 0.1) speed = 0.1;
    m_videoSpeed = speed;
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

        // 录制降频：写帧率按 EveryN 等比例下调，保证视频时长正确（不会加速播放）
        const double effFps = qMax(1.0, m_clipFps / static_cast<double>(m_clipEveryN));
        m_clipTick = 0;

        // 优先 mp4(mp4v)，失败回退 avi(MJPG)
        QString path = QDir(dir).filePath(baseName + ".mp4");
        m_clipWriter.open(path.toStdString(),
                          cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                          effFps, frame.size(), true);
        if (!m_clipWriter.isOpened()) {
            path = QDir(dir).filePath(baseName + ".avi");
            m_clipWriter.open(path.toStdString(),
                              cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                              effFps, frame.size(), true);
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

    // 录制降频：每 m_clipEveryN 帧写 1 帧，省视频编码开销（倍速播放提速）
    if ((m_clipTick++ % m_clipEveryN) != 0) return;
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
        // 暂停播放/采集：不读取帧、不处理，等待 resume() 后从当前位置接续
        if (m_paused.load()) {
            QThread::msleep(50);
            continue;
        }

        const qint64 nowMs = QDateTime::currentMSecsSinceEpoch();
        if (!m_camera->getFrame(frame)) {
            if (m_failStartMs == 0) m_failStartMs = nowMs;
            m_consecutiveFailCount++;
            const qint64 failElapsed = nowMs - m_failStartMs;

            if (m_consecutiveFailCount == 1) {
                LOG_WARN(QString("[GrabThread-%1] Frame grab failed (#1)，进入容错窗口（坏帧/抖动不立即判掉线）")
                         .arg(m_channelName));
            } else if (m_consecutiveFailCount % 50 == 0) {
                LOG_WARN(QString("[GrabThread-%1] Consecutive failures: %2 (elapsed %3ms)")
                         .arg(m_channelName).arg(m_consecutiveFailCount).arg(failElapsed));
            }

            // 掉线判定：连续失败「时长」超过阈值，或「次数」达到兜底上限。
            // 坏帧（read 立即失败）几秒内即恢复，不会误判；真正断流（read 阻塞
            // stimeout 后失败）约 2 个周期内判定掉线。
            const bool timedOut = failElapsed >= FAIL_TIMEOUT_MS;
            const bool tooMany  = m_consecutiveFailCount >= MAX_FAIL_COUNT;
            if (timedOut || tooMany) {
                if (m_camera->isVideoFile()) {
                    LOG_INFO(QString("[GrabThread-%1] 视频文件读取完毕，共 %2 帧")
                                 .arg(m_channelName).arg(frameIndex));
                    emit videoFinished();
                } else {
                    LOG_ERR(QString("[GrabThread-%1] 连续读帧失败 %2ms/%3 次，判定掉线")
                            .arg(m_channelName).arg(failElapsed).arg(m_consecutiveFailCount));
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
            LOG_INFO(QString("[GrabThread-%1] Frame grab recovered after %2 failures (%3ms)")
                     .arg(m_channelName).arg(m_consecutiveFailCount).arg(nowMs - m_failStartMs));
            m_consecutiveFailCount = 0;
            m_failStartMs = 0;
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

        // 背景模型更新（固定摄像头：仅在画面基本无人时学习背景）。
        // 视频文件检测/倍速播放不需要背景扣除，跳过以逼近目标倍速。
        if (m_bgModel && !m_camera->isVideoFile()) {
            m_bgModel->update(frame);
        }

        // 检测管线：如果设置了检测器，则执行检测和跟踪
        // 用 try/catch 兜底：任何 OpenCV/标准异常都不得逃出线程函数，
        // 否则会触发 std::terminate 使整个程序崩溃。
        QVector<TrackedPerson> persons;
        try {
        if (m_detector && m_detector->isLoaded() && m_tracker) {
            m_processing.store(true);   // 标记处理中（模型热重载前等待其空闲）
            // 耗时埋点：定位瓶颈在「检测」还是「跟踪+ReID」
            const qint64 t0 = QDateTime::currentMSecsSinceEpoch();

            // ===== 自适应检测间隔 =====
            // 有乘客/刚检测到乘客 → 逐帧检测（不漏穿越）；连续 idleFramesBeforeSkip 帧
            // 无乘客 → 降频到每 idleSkipEveryN 帧检测一次，空场景快速推进。
            // 乘客从「出现」到「跨门线」有 2~5 秒，而空闲跳帧间隔仅约 0.4 秒，
            // 再次检测到乘客立即切回逐帧即可覆盖完整穿越，无需回退/seek。
            const bool idle = (frameIndex - m_lastPersonFrame) > m_idleFramesBeforeSkip;
            const int effectiveEveryN = idle ? m_idleSkipEveryN : m_detectEveryN;
            const bool doDetect = (effectiveEveryN <= 1) || (frameIndex % effectiveEveryN == 0);
            QVector<cv::Rect> detections;
            if (doDetect) {
                detections = m_detector->detect(frame);
                if (!detections.isEmpty()) m_lastPersonFrame = frameIndex;
            }
            const qint64 tDetect = QDateTime::currentMSecsSinceEpoch() - t0;

            const qint64 t1 = QDateTime::currentMSecsSinceEpoch();
            // 将 frame 和 ReID 提取器传递给 tracker，由 tracker 内部完成
            // 卡曼预测 + IoU + ReID 多阶段匹配；跳过检测的帧不传 frame/reid，
            // 避免在外推帧上白白提取 DeepReID 特征（该帧无真实检测框）。
            persons = m_tracker->update(detections,
                                        doDetect ? frame : cv::Mat(),
                                        doDetect ? m_reidExtractor : nullptr);
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
            m_processing.store(false);   // 处理完成
        }
        } catch (const cv::Exception &e) {
            m_processing.store(false);
            persons.clear();
            LOG_ERR(QString("[GrabThread-%1] cv::Exception in pipeline: %2")
                        .arg(m_channelName, QString::fromStdString(e.what())));
        } catch (const std::exception &e) {
            m_processing.store(false);
            persons.clear();
            LOG_ERR(QString("[GrabThread-%1] std::exception in pipeline: %2")
                        .arg(m_channelName, QString::fromUtf8(e.what())));
        }

        // 1) 每帧检测结果（不背压）：主线程逐帧处理上下车判定 / 图像归档。
        //    倍速播放时检测仍逐帧进行，不能因 UI 显示慢而跳帧漏检。
        emit personsDetected(frame.cols, frame.rows, persons);

        // 2) 显示帧（背压）：UI 尚未消费完上一帧时丢弃本帧“显示”，
        //    避免显示队列无限积压、画面越来越滞后（检测结果不受影响）。
        //    直接传递 BGR 帧（cv::Mat 浅拷贝/COW），颜色转换放到主线程，
        //    采集线程不再做整帧 cvtColor，显著降低每帧开销以逼近目标倍速。
        if (!m_uiBusy.load()) {
            m_uiBusy.store(true);
            // 深拷贝：frame 是本线程复用的缓冲，跨线程(QueuedConnection)浅拷贝会与
            // 下一帧 m_cap.read(frame) 共享底层 buffer，导致 data race / use-after-free，
            // 表现为 av_frame_unref 释放时 free(): invalid pointer 崩溃。
            emit frameProcessed(frame.clone(), persons);
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

        // ===== 视频文件倍速播放（仅节奏控制，不改变检测流程）=====
        // 目标帧间隔 = 1000 / (fps × speed)。若解码+检测本身已超过目标间隔，
        // 则不再额外休眠 → 实际速度自然以「检测程序最大速率」为上限。
        if (m_camera->isVideoFile()) {
            const double fps = m_camera->fps();
            if (fps > 0.0 && m_videoSpeed > 0.0) {
                if (!m_paceTimer.isValid()) m_paceTimer.start();
                const qint64 targetMs = qMax<qint64>(1, static_cast<qint64>(1000.0 / (fps * m_videoSpeed)));
                const qint64 elapsedMs = m_paceTimer.elapsed();
                if (elapsedMs < targetMs) {
                    QThread::msleep(static_cast<unsigned long>(targetMs - elapsedMs));
                }
                m_paceTimer.restart();

                // 节奏诊断：elapsedMs 即「本帧处理耗时（不含 sleep）」
                static qint64 s_paceDbgTotal = 0;
                static int s_paceDbgFrames = 0;
                s_paceDbgTotal += elapsedMs;
                s_paceDbgFrames++;
                if (s_paceDbgFrames % 100 == 0) {
                    LOG_INFO(QString("[Pace] 处理耗时 avg=%1ms/帧 (target=%2ms) frames=%3")
                                 .arg(static_cast<double>(s_paceDbgTotal) / s_paceDbgFrames, 0, 'f', 2)
                                 .arg(targetMs).arg(s_paceDbgFrames));
                    s_paceDbgTotal = 0;
                    s_paceDbgFrames = 0;
                }
            }
        }
    }

    // 线程退出：收尾当前片段，保证文件可播放
    closePersonClip();

    LOG_INFO(QString("[GrabThread-%1] Exited, total frames: %2")
             .arg(m_channelName).arg(frameIndex));
}
