#!/bin/sh
# ============================================================================
# gen_version.sh —— 自动生成 src/version.h
#
# 每次构建（qmake/make 触发）都会：
#   1) 读取并自增构建号（存于 CameraViewer/.build_number）
#   2) 写入当前构建日期/时间
#   3) 重新生成 src/version.h
#
# 因此「每一次程序更新修改」都会自动产生一个新的版本号，
# 无需手动修改任何版本信息。
#
# 如需变更主/次版本号（不兼容改动 / 功能新增），
# 只需修改下面的 MAJOR / MINOR 两个常量。
# ============================================================================

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SRC="$SCRIPT_DIR/src/version.h"
NUM="$SCRIPT_DIR/.build_number"

# ---- 手工维护的语义化版本主/次号 ----
MAJOR=1
MINOR=1

# ---- 自增构建号（即 PATCH 位）----
if [ -r "$NUM" ]; then
    n=$(tr -dc '0-9' < "$NUM" 2>/dev/null || true)
else
    n=""
fi
[ -z "${n:-}" ] && n=0
n=$((n + 1))
printf '%s\n' "$n" > "$NUM"

BUILD_DATE=$(date +%Y-%m-%d)
BUILD_TIME=$(date +%H:%M:%S)

cat > "$SRC" <<EOF
#ifndef APP_VERSION_H
#define APP_VERSION_H

/**
 * @file version.h
 * @brief 程序版本号（由 gen_version.sh 自动生成，请勿手动修改）
 *
 * 版本号 = 主.次.构建号，构建号每次编译自动 +1，
 * 保证「每一次程序更新修改」版本号都会自动变化。
 */

#define APP_VERSION_MAJOR $MAJOR
#define APP_VERSION_MINOR $MINOR
#define APP_VERSION_PATCH $n

#define APP_VERSION_STRING "$MAJOR.$MINOR.$n"

#define APP_BUILD_DATE "$BUILD_DATE"
#define APP_BUILD_TIME "$BUILD_TIME"

// 显示用版本号，例如 "v$MAJOR.$MINOR.$n"
#define APP_VERSION_DISPLAY ("v" APP_VERSION_STRING)

#define APP_PRODUCT_NAME "HX-CCD23 人员检测 ReID 系统"

// 数值版本（用于比较）：MAJOR*10000 + MINOR*100 + PATCH
#define APP_VERSION_CODE \\
    (APP_VERSION_MAJOR * 10000 + APP_VERSION_MINOR * 100 + APP_VERSION_PATCH)

#endif // APP_VERSION_H
EOF

echo "[gen_version] version v$MAJOR.$MINOR.$n ($BUILD_DATE $BUILD_TIME) -> $SRC"
