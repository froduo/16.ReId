// Orbbec 硬件信息探针：输出设备信息、可用流 profile、彩色/深度内参与 FOV、
// 深度值单位(value scale)、D2C 对齐后深度分辨率等。
//
// 编译（在 CameraViewer_orbbec 目录）：
//   SDK=third_party/OrbbecSDK_v2.10.6_202609291004_575347c_linux_x86_64
//   g++ -std=c++17 tests/ob_info.cpp -I$SDK/include \
//       -L$SDK/lib -lOrbbecSDK \
//       -Wl,-rpath,$PWD/$SDK/lib \
//       -Wl,-rpath,$PWD/$SDK/lib/extensions/depthengine \
//       -Wl,-rpath,$PWD/$SDK/lib/extensions/filters \
//       -o tests/ob_info
#include <libobsensor/ObSensor.hpp>

#include <cmath>
#include <cstdio>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static const char *fmtName(OBFormat f)
{
    switch (f) {
    case OB_FORMAT_MJPG:  return "MJPG";
    case OB_FORMAT_YUYV:  return "YUYV";
    case OB_FORMAT_UYVY:  return "UYVY";
    case OB_FORMAT_RGB:   return "RGB";
    case OB_FORMAT_BGR:   return "BGR";
    case OB_FORMAT_RGBA:  return "RGBA";
    case OB_FORMAT_BGRA:  return "BGRA";
    case OB_FORMAT_Y8:    return "Y8";
    case OB_FORMAT_Y16:   return "Y16";
    case OB_FORMAT_Z16:   return "Z16";
    default:              return "OTHER";
    }
}

static void dumpProfile(const char *tag, const std::shared_ptr<ob::StreamProfile> &sp)
{
    auto vsp = sp->as<ob::VideoStreamProfile>();
    if (!vsp)
        return;
    printf("  [%s] %s %ux%u @%ufps", tag, fmtName(vsp->getFormat()),
           vsp->getWidth(), vsp->getHeight(), vsp->getFps());
    try {
        OBCameraIntrinsic in = vsp->getIntrinsic();
        OBCameraDistortion d = vsp->getDistortion();
        double hfov = 2.0 * std::atan2(in.width / 2.0, in.fx) * 180.0 / M_PI;
        double vfov = 2.0 * std::atan2(in.height / 2.0, in.fy) * 180.0 / M_PI;
        printf("  fx=%.2f fy=%.2f cx=%.2f cy=%.2f HFOV=%.1f VFOV=%.1f "
               "k1=%.4f k2=%.4f k3=%.4f p1=%.4f p2=%.4f",
               in.fx, in.fy, in.cx, in.cy, hfov, vfov,
               d.k1, d.k2, d.k3, d.p1, d.p2);
    } catch (std::exception &e) {
        printf("  (intrinsics n/a: %s)", e.what());
    }
    printf("\n");
}

int main()
{
    try {
        ob::Context ctx;
        auto devList = ctx.queryDeviceList();
        printf("==== 设备 ====\n");
        printf("device count = %u\n", devList->getCount());
        for (uint32_t i = 0; i < devList->getCount(); ++i) {
            auto info = devList->getDevice(i)->getDeviceInfo();
            printf("  dev[%u] name=%s sn=%s fw=%s pid=0x%04X vid=0x%04X conn=%s\n",
                   i, info->getName(), info->getSerialNumber(), info->getFirmwareVersion(),
                   info->getPid(), info->getVid(),
                   info->getConnectionType() ? "USB" : "?");
        }
        if (devList->getCount() == 0)
            return 1;

        auto pipe = std::make_shared<ob::Pipeline>();
        auto device = pipe->getDevice();

        printf("\n==== 可用流 profile ====\n");
        auto cl = pipe->getStreamProfileList(OB_SENSOR_COLOR);
        for (uint32_t i = 0; i < cl->getCount(); ++i)
            dumpProfile("COLOR", cl->getProfile(i));
        auto dl = pipe->getStreamProfileList(OB_SENSOR_DEPTH);
        for (uint32_t i = 0; i < dl->getCount(); ++i)
            dumpProfile("DEPTH", dl->getProfile(i));

        printf("\n==== D2C 对齐尝试（HW→SW→OFF）====\n");
        const char *names[3] = {"HW", "SW", "OFF"};
        OBAlignMode modes[3] = {ALIGN_D2C_HW_MODE, ALIGN_D2C_SW_MODE, ALIGN_DISABLE};
        for (int m = 0; m < 3; ++m) {
            try {
                auto p = std::make_shared<ob::Pipeline>();
                auto cfg = std::make_shared<ob::Config>();
                cfg->enableStream(OB_SENSOR_COLOR);
                cfg->enableStream(OB_SENSOR_DEPTH);
                try { cfg->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_ALL_TYPE_FRAME_REQUIRE); } catch (...) {}
                try { cfg->setAlignMode(modes[m]); } catch (ob::Error &e) {
                    printf("  [%s] setAlignMode 不支持: %s\n", names[m], e.getMessage());
                    continue;
                }
                p->start(cfg);
                auto fs = p->waitForFrameset(1500);
                int cw = 0, ch = 0, dw = 0, dh = 0;
                const char *cfmt = "-", *dfmt = "-";
                float dscale = 0.f;
                if (fs) {
                    auto c = fs->getColorFrame();
                    auto d = fs->getDepthFrame();
                    if (c) { cw = (int)c->getWidth(); ch = (int)c->getHeight(); cfmt = fmtName(c->getFormat()); }
                    if (d) { dw = (int)d->getWidth(); dh = (int)d->getHeight(); dfmt = fmtName(d->getFormat()); dscale = d->getValueScale(); }
                }
                printf("  [%s] 启动成功  彩色=%dx%d(%s)  深度=%dx%d(%s)  depthValueScale=%.4f mm/单位\n",
                       names[m], cw, ch, cfmt, dw, dh, dfmt, dscale);
                p->stop();
                break;   // 用第一个可用的对齐模式
            } catch (ob::Error &e) {
                printf("  [%s] 启动失败: %s\n", names[m], e.getMessage());
            }
        }
        return 0;
    } catch (ob::Error &e) {
        printf("[ERR] %s: %s\n", e.getName(), e.getMessage());
        return 2;
    } catch (std::exception &e) {
        printf("[ERR] %s\n", e.what());
        return 3;
    }
}
