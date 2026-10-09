#include "orbbecsource.h"

#include <libobsensor/ObSensor.hpp>
#include <libobsensor/hpp/Filter.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

struct OrbbecSource::Impl
{
    std::shared_ptr<ob::Pipeline> pipe;
    std::shared_ptr<ob::Config> config;
    std::shared_ptr<ob::Align>  align;      ///< 软件 D2C 对齐滤镜（HW 不可用时使用）
    bool hwAlign{false};
    bool depthScaleKnown{false};
};

namespace {

cv::Mat colorToBGR(const std::shared_ptr<ob::ColorFrame> &f)
{
    if (!f)
        return cv::Mat();
    int w = static_cast<int>(f->getWidth());
    int h = static_cast<int>(f->getHeight());
    const uint8_t *d = reinterpret_cast<const uint8_t *>(f->getData());
    uint32_t n = f->getDataSize();

    switch (f->getFormat()) {
    case OB_FORMAT_MJPG: {
        // 直接包裹 MJPEG 缓冲，避免再拷贝一份（imdecode 只读）
        cv::Mat buf(1, static_cast<int>(n), CV_8U, (void *)d);
        return cv::imdecode(buf, cv::IMREAD_COLOR); // BGR
    }
    case OB_FORMAT_RGB: {
        cv::Mat m(h, w, CV_8UC3, (void *)d);
        cv::Mat b;
        cv::cvtColor(m, b, cv::COLOR_RGB2BGR);
        return b;
    }
    case OB_FORMAT_BGR: {
        cv::Mat m(h, w, CV_8UC3, (void *)d);
        return m.clone();
    }
    case OB_FORMAT_RGBA: {
        cv::Mat m(h, w, CV_8UC4, (void *)d);
        cv::Mat b;
        cv::cvtColor(m, b, cv::COLOR_RGBA2BGR);
        return b;
    }
    case OB_FORMAT_BGRA: {
        cv::Mat m(h, w, CV_8UC4, (void *)d);
        cv::Mat b;
        cv::cvtColor(m, b, cv::COLOR_BGRA2BGR);
        return b;
    }
    case OB_FORMAT_YUYV: {
        cv::Mat m(h, w, CV_8UC2, (void *)d);
        cv::Mat b;
        cv::cvtColor(m, b, cv::COLOR_YUV2BGR_YUYV);
        return b;
    }
    case OB_FORMAT_UYVY: {
        cv::Mat m(h, w, CV_8UC2, (void *)d);
        cv::Mat b;
        cv::cvtColor(m, b, cv::COLOR_YUV2BGR_UYVY);
        return b;
    }
    default:
        return cv::Mat();
    }
}

cv::Mat depthToMM(const std::shared_ptr<ob::DepthFrame> &f, float &scale)
{
    if (!f)
        return cv::Mat();
    scale = f->getValueScale();
    const int w = static_cast<int>(f->getWidth());
    const int h = static_cast<int>(f->getHeight());
    const size_t dataSize = f->getDataSize();

    // 行跨距（stride）按实际数据长度推断，兼容 SDK 缓冲区存在行填充的情况
    size_t step = (h > 0) ? (dataSize / static_cast<size_t>(h)) : static_cast<size_t>(w) * 2;
    if (step < static_cast<size_t>(w) * 2)
        step = static_cast<size_t>(w) * 2;

    cv::Mat raw(h, w, CV_16U, (void *)f->getData(), step);
    cv::Mat mm;
    if (std::fabs(scale - 1.0f) < 1e-6f)
        mm = raw.clone();                 // SDK 缓冲会被复用，必须拷贝
    else
        raw.convertTo(mm, CV_16U, static_cast<double>(scale));
    return mm;
}

} // namespace

OrbbecSource::OrbbecSource() : m_impl(new Impl) {}

OrbbecSource::~OrbbecSource()
{
    close();
}

bool OrbbecSource::open(int colorWidth, int colorHeight, int fps)
{
    close();
    m_err.clear();
    m_hasDepth = false;
    m_aligned  = false;
    m_depthW = m_depthH = 0;
    m_depthScale = 1.0f;

    // ===== D2C 对齐策略 =====
    //  HW : config.setAlignMode(ALIGN_D2C_HW_MODE)  —— 无 CPU 开销，最优先；
    //  SW : config 不对齐 + ob::Align 滤镜（强制输出到彩色分辨率）。
    //       注意：config 的 ALIGN_D2C_SW_MODE 实测不会改变深度分辨率（深度仍 848x480），
    //             因此必须用 Align 滤镜才能与彩色逐像素对应。
    // 每种方案都从零重建 Pipeline+Config 再启动；失败则换下一种。
    struct Plan { const char *name; OBAlignMode cfgAlign; bool useFilter; bool hw; };
    const Plan plans[2] = {
        {"HW", ALIGN_D2C_HW_MODE, false, true},
        {"SW(Align滤镜)", ALIGN_DISABLE, true, false},
    };

    std::string lastErr;

    for (const Plan &plan : plans) {
        try {
            auto pipe = std::make_shared<ob::Pipeline>();
            auto device = pipe->getDevice();
            if (!device) {
                m_err = "no device";
                return false;
            }
            auto info = device->getDeviceInfo();
            m_devName = std::string(info->getName()) + " sn=" + info->getSerialNumber();
            pipe->enableFrameSync(); // 按时间戳同步彩色/深度

            auto config = std::make_shared<ob::Config>();

            // 选择 profile：优先指定分辨率/帧率，找不到则用默认
            bool configured = false;
            if (colorWidth > 0 && colorHeight > 0) {
                auto colorList = pipe->getStreamProfileList(OB_SENSOR_COLOR);
                for (uint32_t i = 0; i < colorList->getCount(); ++i) {
                    auto sp = colorList->getProfile(i)->as<ob::VideoStreamProfile>();
                    if (static_cast<int>(sp->getWidth()) == colorWidth
                        && static_cast<int>(sp->getHeight()) == colorHeight
                        && (fps == 0 || static_cast<int>(sp->getFps()) == fps)) {
                        config->enableStream(sp);
                        configured = true;
                        break;
                    }
                }
            }
            if (!configured) {
                config->enableStream(OB_SENSOR_COLOR);
            }

            bool hasDepth = true;
            try {
                config->enableStream(OB_SENSOR_DEPTH);
            } catch (...) {
                hasDepth = false;
            }

            // 无深度流时无需 D2C 对齐
            const OBAlignMode useAlign = hasDepth ? plan.cfgAlign : ALIGN_DISABLE;

            try {
                config->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_ALL_TYPE_FRAME_REQUIRE);
            } catch (...) {
            }

            if (hasDepth) {
                try {
                    config->setAlignMode(useAlign);
                } catch (...) {
                    lastErr = "align mode unsupported";
                    try { pipe->stop(); } catch (...) {}
                    continue;                       // 换下一种对齐方案
                }
            }

            try {
                pipe->start(config);
            } catch (ob::Error &e) {
                lastErr = std::string(e.getName()) + ": " + e.getMessage();
                try { pipe->stop(); } catch (...) {}
                continue;                           // 换下一种对齐方案重试
            }

            // 成功：保存句柄
            m_impl->pipe = pipe;
            m_impl->config = config;
            m_impl->hwAlign = plan.hw;
            m_impl->align.reset();
            m_hasDepth = hasDepth;

            // 取宽高（启动后从首个 profile 推断）
            auto cp = pipe->getStreamProfileList(OB_SENSOR_COLOR);
            for (uint32_t i = 0; i < cp->getCount(); ++i) {
                auto sp = cp->getProfile(i)->as<ob::VideoStreamProfile>();
                if (colorWidth == 0 || static_cast<int>(sp->getWidth()) == colorWidth) {
                    m_w = static_cast<int>(sp->getWidth());
                    m_h = static_cast<int>(sp->getHeight());
                    break;
                }
            }

            // 软件对齐：创建 Align 滤镜（输出到彩色分辨率）
            if (hasDepth && plan.useFilter) {
                try {
                    m_impl->align = std::make_shared<ob::Align>(OB_STREAM_COLOR);
                    m_impl->align->setMatchTargetResolution(true);
                } catch (ob::Error &e) {
                    lastErr = std::string("Align filter: ") + e.getMessage();
                }
            }
            m_alignName = hasDepth ? plan.name : "OFF";
            m_aligned   = hasDepth;   // 深度已与彩色对齐（同分辨率）

            m_open = true;
            return true;
        } catch (ob::Error &e) {
            lastErr = std::string(e.getName()) + ": " + e.getMessage();
        } catch (std::exception &e) {
            lastErr = e.what();
        }
    }

    m_err = "pipeline start failed (HW/SW): " + lastErr;
    close();
    return false;
}

void OrbbecSource::close()
{
    try {
        if (m_impl && m_impl->pipe) {
            m_impl->pipe->stop();
            m_impl->pipe.reset();
        }
        if (m_impl)
            m_impl->config.reset();
    } catch (...) {
    }
    m_open = false;
}

bool OrbbecSource::getFrames(cv::Mat &colorBGR, cv::Mat &depthMM,
                             uint64_t &colorTsUs, uint64_t &depthTsUs, int timeoutMs)
{
    colorBGR.release();
    depthMM.release();
    colorTsUs = depthTsUs = 0;
    if (!m_open || !m_impl || !m_impl->pipe)
        return false;

    try {
        auto fs = m_impl->pipe->waitForFrameset(timeoutMs);
        if (!fs)
            return false;

        // 软件 D2C 对齐：把深度重采样到彩色分辨率（逐像素对应）
        if (m_impl->align) {
            try {
                auto out = m_impl->align->process(fs);
                if (out) {
                    auto ofs = out->as<ob::FrameSet>();
                    if (ofs)
                        fs = ofs;
                }
            } catch (ob::Error &e) {
                m_err = std::string("align: ") + e.getMessage();
            }
        }

        auto cf = fs->getColorFrame();
        if (cf) {
            colorBGR = colorToBGR(cf);
            colorTsUs = cf->getTimeStampUs();
        }
        float scale = 1.0f;
        auto df = fs->getDepthFrame();
        if (df) {
            depthMM = depthToMM(df, scale);
            depthTsUs = df->getTimeStampUs();
            m_depthW = static_cast<int>(df->getWidth());
            m_depthH = static_cast<int>(df->getHeight());
            if (!m_impl->depthScaleKnown) {
                m_depthScale = scale;
                m_impl->depthScaleKnown = true;
            }
        }
        return !colorBGR.empty() || !depthMM.empty();
    } catch (ob::Error &e) {
        m_err = std::string(e.getName()) + ": " + e.getMessage();
        return false;
    } catch (std::exception &e) {
        m_err = e.what();
        return false;
    }
}

cv::Mat OrbbecSource::depthToColor(const cv::Mat &depthMM, float minMM, float maxMM)
{
    if (depthMM.empty() || depthMM.type() != CV_16U)
        return cv::Mat();

    cv::Mat mask = depthMM > 0;
    cv::Mat f;
    depthMM.convertTo(f, CV_32F);

    cv::Mat norm;
    f.convertTo(norm, CV_32F, 1.0);
    norm = (norm - minMM) * (255.0f / std::max(1.0f, maxMM - minMM));
    cv::min(norm, 255.0f, norm);
    cv::max(norm, 0.0f, norm);
    cv::Mat u8;
    norm.convertTo(u8, CV_8U);

    cv::Mat color;
    cv::applyColorMap(u8, color, cv::COLORMAP_JET);
    color.setTo(cv::Scalar(0, 0, 0), ~mask);
    return color;
}

cv::Mat OrbbecSource::depthToColorAuto(const cv::Mat &depthMM)
{
    if (depthMM.empty() || depthMM.type() != CV_16U)
        return cv::Mat();
    // 收集有效深度像素（>0）并排序，用于 5%~95% 分位自动拉伸
    std::vector<uint16_t> v;
    v.reserve(static_cast<size_t>(depthMM.total()));
    for (int y = 0; y < depthMM.rows; ++y) {
        const uint16_t *p = depthMM.ptr<uint16_t>(y);
        for (int x = 0; x < depthMM.cols; ++x) {
            uint16_t d = p[x];
            if (d > 0)
                v.push_back(d);
        }
    }
    if (v.empty())
        return depthToColor(depthMM);
    std::sort(v.begin(), v.end());
    float lo = static_cast<float>(v[static_cast<size_t>(v.size() * 0.05)]);
    size_t hiIdx = std::min(v.size() - 1, static_cast<size_t>(v.size() * 0.95));
    float hi = static_cast<float>(v[hiIdx]);
    if (hi - lo < 100.0f) {
        lo = 300.0f;
        hi = 3000.0f;
    }
    return depthToColor(depthMM, lo, hi);
}
