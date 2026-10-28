#ifndef FFMPEGCAPTURE_H
#define FFMPEGCAPTURE_H

#include <QString>
#include <opencv2/opencv.hpp>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

/**
 * @brief 基于 FFmpeg API 的 RTSP 拉流封装（替代 OpenCV VideoCapture 拉 RTSP）。
 *
 * 为什么要单独用 FFmpeg 拉 RTSP：
 *   OpenCV 的 CAP_FFMPEG 在遇到 H.264 坏帧（摄像头偶发输出损坏码流）时，
 *   m_cap.read() 会直接返回 false，导致 GrabThread 误判「掉线」并频繁重连；
 *   而 VLC/ffmpeg 命令行对坏帧做「错误隐藏(conceal)」，跳过坏帧继续播放，
 *   因此 VLC 基本无卡顿。
 *
 * 本类在解码器层设置 err_detect=ignore_err，坏帧会被 conceal 后继续输出
 * （画面局部花屏但不中断），仅当网络真正断流（av_read_frame 超时）才返回失败，
 * 从而与 VLC 行为对齐。
 */
class FfmpegCapture
{
public:
    FfmpegCapture();
    ~FfmpegCapture();

    /** 打开 RTSP/网络流（TCP 传输，stimeout=6s，timeout=3s） */
    bool open(const QString &url);

    /** 读取一帧（BGR cv::Mat）。坏帧被 conceal 跳过，仅真断流返回 false。 */
    bool read(cv::Mat &frame);

    void close();
    bool isOpened() const { return m_opened; }
    int width() const { return m_width; }
    int height() const { return m_height; }
    double fps() const { return m_fps; }

private:
    AVFormatContext *m_fmtCtx{nullptr};
    AVCodecContext  *m_decCtx{nullptr};
    SwsContext      *m_swsCtx{nullptr};
    AVFrame         *m_frame{nullptr};
    AVFrame         *m_bgrFrame{nullptr};
    AVPacket        *m_pkt{nullptr};
    int m_videoStreamIndex{-1};
    bool m_opened{false};
    int m_width{0};
    int m_height{0};
    double m_fps{0.0};
};

#endif // FFMPEGCAPTURE_H
