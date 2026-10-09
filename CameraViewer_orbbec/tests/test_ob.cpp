// OrbbecSDK v2 连通性测试：枚举设备
#include <libobsensor/ObSensor.hpp>
#include <iostream>

int main() {
    try {
        ob::Context ctx;
        auto devList = ctx.queryDeviceList();
        auto n = devList->getCount();
        std::cout << "[OK] device count = " << n << std::endl;
        for (uint32_t i = 0; i < n; ++i) {
            auto dev = devList->getDevice(i);
            auto info = dev->getDeviceInfo();
            std::cout << "  dev[" << i << "] name=" << info->getName()
                      << " sn=" << info->getSerialNumber()
                      << " fw=" << info->getFirmwareVersion()
                      << " pid=0x" << std::hex << info->getPid()
                      << " vid=0x" << info->getVid() << std::dec << std::endl;
        }
        return n > 0 ? 0 : 1;
    } catch (ob::Error &e) {
        std::cerr << "[ERR] " << e.getName() << ": " << e.getMessage() << std::endl;
        return 2;
    } catch (std::exception &e) {
        std::cerr << "[ERR] " << e.what() << std::endl;
        return 3;
    }
}
