// OrbbecSource 独立连通性测试：验证 D2C 对齐回退（HW→SW→OFF）后能成功打开并取帧。
// 编译示例（在 CameraViewer_orbbec 目录）：
//   g++ -std=c++17 tests/test_source.cpp src/orbbecsource.cpp \
//       -Isrc -Ithird_party/OrbbecSDK_v2.10.6_202609291004_575347c_linux_x86_64/include \
//       $(pkg-config --cflags opencv4) \
//       -Lthird_party/OrbbecSDK_v2.10.6_202609291004_575347c_linux_x86_64/lib -lOrbbecSDK \
//       $(pkg-config --libs opencv4) -o tests/test_source
#include "orbbecsource.h"

#include <cstdio>

int main()
{
    OrbbecSource src;
    if (!src.open()) {
        std::printf("[FAIL] open(): %s\n", src.lastError().c_str());
        return 1;
    }

    std::printf("[OK] opened: %s | color=%dx%d | depth=%s | align=%s | depthScale=%.3f\n",
                src.deviceName().c_str(), src.width(), src.height(),
                src.hasDepth() ? "on" : "off",
                src.alignMode().c_str(), src.depthScale());

    int okFrames = 0;
    int colorOk = 0;
    int depthOk = 0;
    for (int i = 0; i < 30; ++i) {
        cv::Mat color, depth;
        uint64_t cts = 0, dts = 0;
        if (src.getFrames(color, depth, cts, dts, 1000)) {
            ++okFrames;
            if (!color.empty()) ++colorOk;
            if (!depth.empty()) ++depthOk;
            if (okFrames == 1) {
                std::printf("[OK] first frame: color=%dx%d(%dch) depth=%dx%d(%dch) "
                            "aligned=%s\n",
                            color.cols, color.rows, color.channels(),
                            depth.cols, depth.rows, depth.channels(),
                            (depth.cols == color.cols && depth.rows == color.rows)
                                ? "yes(pixel-wise)" : "NO");
            }
        }
    }

    std::printf("[RESULT] frames=%d color=%d depth=%d\n", okFrames, colorOk, depthOk);
    src.close();
    return okFrames > 0 ? 0 : 2;
}
