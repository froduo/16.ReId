#include "ffmpegcapture.h"
#include "logger.h"

FfmpegCapture::FfmpegCapture() = default;

FfmpegCapture::~FfmpegCapture()
{
    close();
}

bool FfmpegCapture::open(const QString &url)
{
    close();

    AVDictionary *opts = nullptr;
    // TCP 交错传输（本相机 UDP 会连接超时）；stimeout=6s 会话读超时（容忍抖动），
    // timeout=3s socket 连接超时（open 快速失败）。
    av_dict_set(&opts, "rtsp_transport", "tcp", 0);
    av_dict_set(&opts, "stimeout", "6000000", 0);
    av_dict_set(&opts, "timeout", "3000000", 0);

    int ret = avformat_open_input(&m_fmtCtx, url.toUtf8().constData(), nullptr, &opts);
    av_dict_free(&opts);
    if (ret < 0) {
        LOG_ERR(QString("[FfmpegCapture] avformat_open_input 失败(%1): %2")
                    .arg(ret).arg(url));
        close();
        return false;
    }

    ret = avformat_find_stream_info(m_fmtCtx, nullptr);
    if (ret < 0) {
        LOG_ERR("[FfmpegCapture] avformat_find_stream_info 失败");
        close();
        return false;
    }

    // 定位视频流
    for (unsigned i = 0; i < m_fmtCtx->nb_streams; ++i) {
        if (m_fmtCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            m_videoStreamIndex = static_cast<int>(i);
            break;
        }
    }
    if (m_videoStreamIndex < 0) {
        LOG_ERR("[FfmpegCapture] 未找到视频流");
        close();
        return false;
    }

    AVStream *vs = m_fmtCtx->streams[m_videoStreamIndex];
    const AVCodec *dec = avcodec_find_decoder(vs->codecpar->codec_id);
    if (!dec) {
        LOG_ERR("[FfmpegCapture] 未找到 H.264 解码器");
        close();
        return false;
    }

    m_decCtx = avcodec_alloc_context3(dec);
    if (!m_decCtx) {
        close();
        return false;
    }
    avcodec_parameters_to_context(m_decCtx, vs->codecpar);

    // 关键容错：忽略解码错误（conceal 错误隐藏）。摄像头偶发输出损坏的 H.264 码流，
    // 若不忽略，avcodec_receive_frame 会返回错误导致丢帧/中断（OpenCV 即因此反复掉线）；
    // 设 IGNORE_ERR 后，坏帧被 conceal 成可用帧继续输出，与 VLC 行为对齐。
    av_opt_set_int(m_decCtx, "err_detect", AV_EF_IGNORE_ERR, 0);
    // 低延迟：不缓冲参考帧的重排序延迟（实时场景优先出帧）。
    av_opt_set_int(m_decCtx, "flags2", AV_CODEC_FLAG2_FAST, 0);

    ret = avcodec_open2(m_decCtx, dec, nullptr);
    if (ret < 0) {
        LOG_ERR("[FfmpegCapture] avcodec_open2 失败");
        close();
        return false;
    }

    m_width = m_decCtx->width;
    m_height = m_decCtx->height;
    m_fps = av_q2d(vs->avg_frame_rate);
    if (m_fps <= 0.0 || m_fps > 120.0) m_fps = av_q2d(vs->r_frame_rate);
    if (m_fps <= 0.0 || m_fps > 120.0) m_fps = 25.0;

    m_frame = av_frame_alloc();
    m_bgrFrame = av_frame_alloc();
    m_pkt = av_packet_alloc();
    if (!m_frame || !m_bgrFrame || !m_pkt) {
        close();
        return false;
    }

    m_bgrFrame->format = AV_PIX_FMT_BGR24;
    m_bgrFrame->width = m_width;
    m_bgrFrame->height = m_height;
    if (av_frame_get_buffer(m_bgrFrame, 32) < 0) {
        LOG_ERR("[FfmpegCapture] BGR 帧缓冲分配失败");
        close();
        return false;
    }

    m_swsCtx = sws_getContext(m_width, m_height, m_decCtx->pix_fmt,
                              m_width, m_height, AV_PIX_FMT_BGR24,
                              SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!m_swsCtx) {
        LOG_ERR("[FfmpegCapture] sws_getContext 失败");
        close();
        return false;
    }

    m_opened = true;
    LOG_INFO(QString("[FfmpegCapture] RTSP 打开成功: %1x%2 @ %3 fps (FFmpeg 直接拉流)")
                 .arg(m_width).arg(m_height).arg(m_fps, 0, 'f', 2));
    return true;
}

bool FfmpegCapture::read(cv::Mat &frame)
{
    if (!m_opened) return false;

    while (true) {
        int ret = av_read_frame(m_fmtCtx, m_pkt);
        if (ret < 0) {
            // 网络断流 / stimeout 超时 / EOF：真正失败
            av_packet_unref(m_pkt);
            return false;
        }

        // 跳过非视频流（音频/数据）
        if (m_pkt->stream_index != m_videoStreamIndex) {
            av_packet_unref(m_pkt);
            continue;
        }

        ret = avcodec_send_packet(m_decCtx, m_pkt);
        av_packet_unref(m_pkt);
        if (ret < 0 && ret != AVERROR(EAGAIN)) {
            // 送包失败（多为解码器内部错误）：跳过该包继续读，不中断
            continue;
        }

        ret = avcodec_receive_frame(m_decCtx, m_frame);
        if (ret == AVERROR(EAGAIN)) {
            // 需要更多包才能出一帧：继续读
            continue;
        }
        if (ret < 0) {
            // 解码失败（坏帧）：跳过，继续读下一包
            av_frame_unref(m_frame);
            continue;
        }

        // 成功解码一帧 → 转 BGR
        sws_scale(m_swsCtx,
                  m_frame->data, m_frame->linesize, 0, m_height,
                  m_bgrFrame->data, m_bgrFrame->linesize);

        cv::Mat tmp(m_height, m_width, CV_8UC3,
                    m_bgrFrame->data[0], m_bgrFrame->linesize[0]);
        frame = tmp.clone();  // 深拷贝，避免与 FFmpeg 内部缓冲共享
        av_frame_unref(m_frame);
        return true;
    }
}

void FfmpegCapture::close()
{
    if (m_swsCtx) { sws_freeContext(m_swsCtx); m_swsCtx = nullptr; }
    if (m_bgrFrame) { av_frame_free(&m_bgrFrame); }
    if (m_frame) { av_frame_free(&m_frame); }
    if (m_pkt) { av_packet_free(&m_pkt); }
    if (m_decCtx) { avcodec_free_context(&m_decCtx); }
    if (m_fmtCtx) { avformat_close_input(&m_fmtCtx); }
    m_videoStreamIndex = -1;
    m_opened = false;
    m_width = m_height = 0;
    m_fps = 0.0;
}
