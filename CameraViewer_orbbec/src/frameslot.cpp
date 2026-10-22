#include "frameslot.h"

void LatestFrameSlot::publish(const cv::Mat &frame)
{
    if (frame.empty()) return;

    QMutexLocker lk(&m_mutex);
    m_frame = frame.clone();
    m_seq++;
    m_cond.wakeOne();
}

bool LatestFrameSlot::waitForFrame(cv::Mat &out, int timeoutMs, const std::atomic<bool> &running)
{
    QMutexLocker lk(&m_mutex);

    // 等待直到有新帧发布，或超时，或要求停止
    while (m_seq == m_consumedSeq && running.load()) {
        if (!m_cond.wait(&m_mutex, static_cast<unsigned long>(timeoutMs)))
            break;   // 超时
    }

    // 要求停止且没有新帧可处理 → 退出
    if (!running.load() && m_seq == m_consumedSeq)
        return false;

    m_consumedSeq = m_seq;
    out = m_frame.clone();
    return !out.empty();
}

void LatestFrameSlot::reset()
{
    QMutexLocker lk(&m_mutex);
    m_frame.release();
    m_seq = 0;
    m_consumedSeq = 0;
}
