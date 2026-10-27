#ifndef APP_VERSION_H
#define APP_VERSION_H

/**
 * @file version.h
 * @brief 程序版本号（由 gen_version.sh 自动生成，请勿手动修改）
 *
 * 版本号 = 主.次.构建号，构建号每次编译自动 +1，
 * 保证「每一次程序更新修改」版本号都会自动变化。
 */

#define APP_VERSION_MAJOR 1
#define APP_VERSION_MINOR 1
#define APP_VERSION_PATCH 144

#define APP_VERSION_STRING "1.1.144"

#define APP_BUILD_DATE "2026-10-27"
#define APP_BUILD_TIME "14:12:49"

// 显示用版本号，例如 "v1.1.144"
#define APP_VERSION_DISPLAY ("v" APP_VERSION_STRING)

#define APP_PRODUCT_NAME "HX-CCD23 人员检测 ReID 系统"

// 数值版本（用于比较）：MAJOR*10000 + MINOR*100 + PATCH
#define APP_VERSION_CODE \
    (APP_VERSION_MAJOR * 10000 + APP_VERSION_MINOR * 100 + APP_VERSION_PATCH)

#endif // APP_VERSION_H
