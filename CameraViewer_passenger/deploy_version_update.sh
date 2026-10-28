#!/bin/sh
# ============================================================================
# deploy_version_update.sh
#   把「版本号 / 更新」相关文件部署到指定的程序目录。
#
# 用法：
#   ./deploy_version_update.sh [目标程序目录]
#   默认目标目录： build-CameraViewer-Desktop-Debug
#
# 部署内容：
#   1) style.qss        —— 界面样式（程序运行需要）
#   2) VERSION.txt      —— 当前版本记录（从 src/version.h 读取）
#   3) config.ini       —— [Update]/UpdateUrl（若已存在则保留，不覆盖）
#   4) update.json      —— 本地更新清单模板（若已存在则保留，不覆盖）
# ============================================================================

set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DEST=${1:-"$ROOT/build-CameraViewer-Desktop-Debug"}
SRC="$ROOT/CameraViewer/src/version.h"

if [ ! -d "$DEST" ]; then
    echo "[deploy] 目标目录不存在：$DEST" >&2
    exit 1
fi

# ---- 版本信息：优先从「已构建的二进制」读取，保证与程序一致；否则回退 version.h ----
VER_STR=""
BIN="$DEST/CameraViewer"
if [ -f "$BIN" ]; then
    VER_STR=$(grep -aoE 'v[0-9]+\.[0-9]+\.[0-9]+' "$BIN" 2>/dev/null | head -1)
fi

B_DATE=$(sed -n 's/^#define APP_BUILD_DATE  *"\([^"]*\)".*/\1/p' "$SRC")
B_TIME=$(sed -n 's/^#define APP_BUILD_TIME  *"\([^"]*\)".*/\1/p' "$SRC")
H_VER="v$(sed -n 's/^#define APP_VERSION_MAJOR  *\([0-9]*\).*/\1/p' "$SRC").$(sed -n 's/^#define APP_VERSION_MINOR  *\([0-9]*\).*/\1/p' "$SRC").$(sed -n 's/^#define APP_VERSION_PATCH  *\([0-9]*\).*/\1/p' "$SRC")"

[ -z "$VER_STR" ] && VER_STR="$H_VER"

VER_NUM=${VER_STR#v}
VER_MAJOR=$(printf '%s' "$VER_NUM" | cut -d. -f1)
VER_MINOR=$(printf '%s' "$VER_NUM" | cut -d. -f2)
VER_PATCH=$(printf '%s' "$VER_NUM" | cut -d. -f3)

echo "[deploy] 版本 $VER_STR ($B_DATE $B_TIME) -> $DEST"

# ---- 1) 样式 ----
if [ -f "$ROOT/CameraViewer/style.qss" ]; then
    cp -f "$ROOT/CameraViewer/style.qss" "$DEST/style.qss"
    echo "[deploy] 已部署 style.qss"
fi

# ---- 2) 版本记录 ----
cat > "$DEST/VERSION.txt" <<EOF
========================================================================
 CameraViewer  版本信息 / Version Info
========================================================================
产品名称 : HX-CCD23 人员检测 ReID 系统
版本号   : $VER_STR
主版本   : $VER_MAJOR
次版本   : $VER_MINOR
构建号   : $VER_PATCH
构建日期 : $B_DATE $B_TIME

说明：
  - 版本号由 gen_version.sh 每次构建自动生成并自增。
  - 程序右下角显示：版本: $VER_STR
  - 菜单「关于 → 检查更新」读取远端 UpdateUrl 或本地 update.json。
========================================================================
EOF
echo "[deploy] 已写入 VERSION.txt"

# ---- 3) config.ini（保留既有内容，仅确保存在 Update 段）----
if [ ! -f "$DEST/config.ini" ]; then
    cat > "$DEST/config.ini" <<'EOF'
[Update]
; 更新清单(manifest)地址：http(s)://host/path/version.json
; 留空时仅使用程序目录下的本地 update.json（离线/U盘升级）。
UpdateUrl=
EOF
    echo "[deploy] 已创建 config.ini"
else
    echo "[deploy] config.ini 已存在，保留不动"
fi

# ---- 4) update.json（模板，已存在则保留）----
if [ ! -f "$DEST/update.json" ]; then
    cat > "$DEST/update.json" <<EOF
{
  "version": "$VER_MAJOR.$VER_MINOR.$VER_PATCH",
  "url": "",
  "notes": "本地更新清单示例：修改 version 为更高版本号并填写 url 后，程序「检查更新」会提示有更新。"
}
EOF
    echo "[deploy] 已创建 update.json 模板"
else
    echo "[deploy] update.json 已存在，保留不动"
fi

echo "[deploy] 完成。程序目录：$DEST"
