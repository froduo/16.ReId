#ifndef FRAMESLOT_H
#define FRAMESLOT_H

#include <QMutex>
#include <QWaitCondition>
#include <opencv2/opencv.hpp>
#include <atomic>

/**
 * @brief 只保留最新帧的线程安全槽
 *
 * RGB 采集线程以流帧率（约 30fps）调用 publish() 写入最新一帧；
 * 检测线程调用 waitForFrame() 获取最新帧。检测线程处理不过来时，
 * 中间帧会被覆盖丢弃（只投递最新帧），避免队列积压造成画面越来越滞后。
 */
class LatestFrameSlot
{
public:
    /** 写入最新帧（覆盖旧帧），并唤醒等待中的检测线程 */
    void publish(const cv::Mat &frame);

    /**
     * @brief 等待并取出最新一帧
     * @param out       输出帧（深拷贝，调用方可安全使用）
     * @param timeoutMs 无新帧时的等待超时（毫秒）
     * @param running   运行标志；置 false 且无新帧时返回 false 以便线程退出
     * @return true=取到新帧；false=超时无新帧或已要求停止
     */
    bool waitForFrame(cv::Mat &out, int timeoutMs, const std::atomic<bool> &running);

    /** 清空槽内帧（停止/重连时调用） */
    void reset();

private:
    QMutex m_mutex;
    QWaitCondition m_cond;
    cv::Mat m_frame;
    quint64 m_seq{0};           // 已发布帧序号
    quint64 m_consumedSeq{0};   // 已消费帧序号（仅检测线程消费）
};

#endif // FRAMESLOT_H
