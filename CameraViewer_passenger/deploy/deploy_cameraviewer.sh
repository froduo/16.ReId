#!/usr/bin/env bash
# =============================================================================
# deploy_cameraviewer.sh —— 在 192.168.10.88 的 “test” 盘上部署并运行 CameraViewer
#
# 用法（在 88 上执行，推荐 bash 而非 sh）：
#   bash deploy_cameraviewer.sh                 # 自动找 test 盘 -> 拷贝源码 -> 编译 -> 运行
#   bash deploy_cameraviewer.sh --install-deps  # 额外尝试用 apt 安装基础依赖
#   bash deploy_cameraviewer.sh --in-place      # 已在源码目录内，原地编译（不拷贝）
#   bash deploy_cameraviewer.sh --no-run        # 只编译，不启动
#   TEST_ROOT=/mnt/test bash deploy_cameraviewer.sh
#   SRC_DIR=/path/to/CameraViewer bash deploy_cameraviewer.sh
#
# 前置依赖（缺哪个脚本会明确提示）：
#   - g++ / make (build-essential)
#   - Qt6 开发工具(qmake + Widgets/Network/Sql)  例：/home/xxx/Qt/6.2.4/gcc_64/bin/qmake
#   - OpenCV 4（建议 CUDA 版，/usr/local/include/opencv4）；无 CUDA 版时脚本自动给 .pro 打补丁
#   - CUDA Toolkit  (/usr/local/cuda)
#   - TensorRT 10.3  (/usr/local/TensorRT-10.3)
#   - NVIDIA 驱动 + 显卡
# =============================================================================
set -uo pipefail

C_BLUE='\033[1;36m'; C_YEL='\033[1;33m'; C_RED='\033[1;31m'; C_GRN='\033[1;32m'; C_RST='\033[0m'
log(){  printf "${C_BLUE}[deploy]${C_RST} %s\n" "$*"; }
ok(){   printf "${C_GRN}[  ok  ]${C_RST} %s\n" "$*"; }
warn(){ printf "${C_YEL}[ warn ]${C_RST} %s\n" "$*" >&2; }
die(){  printf "${C_RED}[ fail ]${C_RST} %s\n" "$*" >&2; exit 1; }

INSTALL_DEPS=0; DO_RUN=1; IN_PLACE=0
for a in "$@"; do
  case "$a" in
    --install-deps) INSTALL_DEPS=1;;
    --no-run)       DO_RUN=0;;
    --in-place)     IN_PLACE=1;;
    -h|--help)      sed -n '2,22p' "$0"; exit 0;;
    *) warn "忽略未知参数: $a";;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- 1. 定位源码（含 CameraViewer.pro） ----------
find_src(){
  local c
  for c in "${SRC_DIR:-}" "$SCRIPT_DIR/.." "$SCRIPT_DIR" "$PWD"; do
    [ -n "$c" ] && [ -f "$c/CameraViewer.pro" ] && { (cd "$c" && pwd); return 0; }
  done
  find "$HOME" /opt /srv /mnt /media /data -maxdepth 5 -name CameraViewer.pro 2>/dev/null | head -1
}
SRC="$(find_src)"
[ -n "$SRC" ] || die "未找到 CameraViewer.pro。请设置 SRC_DIR=/path/to/CameraViewer 再运行。"
log "源码目录: $SRC"

# ---------- 2. 定位 test 盘 ----------
if [ "$IN_PLACE" = 1 ]; then
  DEST="$SRC"
else
  if [ -n "${TEST_ROOT:-}" ]; then
    ROOT="$TEST_ROOT"
  else
    ROOT="$(lsblk -o LABEL,MOUNTPOINT -nr 2>/dev/null | awk 'tolower($0)~/test/{print $NF}' | grep -v '^$' | head -1)"
    if [ -z "${ROOT:-}" ]; then
      for m in /mnt/test* /media/*/test* /mnt/*test* /data/test* /test; do
        [ -d "$m" ] && { ROOT="$m"; break; }
      done
    fi
  fi
  if [ -z "${ROOT:-}" ] || [ ! -d "${ROOT:-}" ]; then
    echo "---- 当前块设备（找名为 test 的盘）----"
    lsblk -o NAME,SIZE,FSTYPE,LABEL,MOUNTPOINT
    die "未找到 test 盘。请用 TEST_ROOT=/mnt/xxx 指定挂载点后重试。"
  fi
  DEST="$ROOT/CameraViewer"
fi
log "部署目标目录: $DEST"

# ---------- 3. 拷贝源码（排除构建产物） ----------
if [ "$IN_PLACE" != 1 ]; then
  log "同步源码到 $DEST ..."
  mkdir -p "$DEST"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a \
      --exclude 'build/' --exclude 'build-*/' --exclude '.git/' \
      --exclude '*.o' --exclude 'moc_*.cpp' --exclude 'moc_*.o' \
      --exclude 'Makefile' --exclude '.qmake.stash' --exclude '*.pro.user*' \
      --exclude 'config.ini' \
      "$SRC"/ "$DEST"/
  else
    ( cd "$SRC" && tar --exclude='./build' --exclude='./build-*' --exclude='./.git' \
        --exclude='*.o' --exclude='moc_*' --exclude='./Makefile' \
        --exclude='./.qmake.stash' --exclude='./config.ini' -cf - . ) \
      | ( cd "$DEST" && tar -xf - )
  fi
  chmod +x "$DEST/gen_version.sh" 2>/dev/null || true
fi

# ---------- 4. 依赖检查 ----------
MISSING=()
command -v g++  >/dev/null 2>&1 || MISSING+=("g++ (build-essential)")
command -v make >/dev/null 2>&1 || MISSING+=("make")

QMAKE_BIN="${QMAKE_BIN:-}"
if [ -z "$QMAKE_BIN" ]; then
  for q in qmake6 qmake; do
    if command -v "$q" >/dev/null 2>&1; then QMAKE_BIN="$(command -v "$q")"; break; fi
  done
fi
if [ -z "$QMAKE_BIN" ]; then
  QMAKE_BIN="$(ls -1 /home/*/Qt/6.*/gcc_64/bin/qmake /opt/Qt/6.*/gcc_64/bin/qmake 2>/dev/null | head -1)"
fi
[ -n "$QMAKE_BIN" ] && [ -x "$QMAKE_BIN" ] || MISSING+=("Qt6 开发工具(qmake)")

if [ -d /usr/local/include/opencv4 ] || { command -v pkg-config >/dev/null 2>&1 && pkg-config --exists opencv4 2>/dev/null; }; then :; else MISSING+=("OpenCV4 开发库"); fi
[ -d /usr/local/cuda ] || MISSING+=("CUDA(/usr/local/cuda)")
[ -d /usr/local/TensorRT-10.3 ] || MISSING+=("TensorRT-10.3(/usr/local/TensorRT-10.3)")

if [ "${#MISSING[@]}" -gt 0 ]; then
  warn "以下依赖缺失："
  for m in "${MISSING[@]}"; do warn "   - $m"; done
  if [ "$INSTALL_DEPS" = 1 ] && command -v apt-get >/dev/null 2>&1; then
    log "尝试安装基础依赖（apt）: build-essential qt6-base-dev libopencv-dev ..."
    SUDO=""; [ "$(id -u)" != 0 ] && SUDO="sudo"
    $SUDO apt-get update || true
    $SUDO apt-get install -y build-essential qt6-base-dev qt6-base-dev-tools libopencv-dev pkg-config || true
    warn "注意：apt 版 OpenCV 通常不含 CUDA 模块；CUDA 与 TensorRT 需按 NVIDIA 官方方式单独安装。"
  else
    warn "可用 --install-deps 尝试 apt 安装基础依赖；CUDA/TensorRT 需手动安装。"
  fi
fi

# ---------- 4b. OpenCV 是否含 CUDA：无则给 .pro 打补丁 ----------
PRO="$DEST/CameraViewer.pro"
OCV_CUDA_OK=0
if ldconfig -p 2>/dev/null | grep -qi 'libopencv_cudaarithm'; then OCV_CUDA_OK=1; fi
if [ "$OCV_CUDA_OK" = 0 ]; then
  warn "未检测到 OpenCV CUDA 模块(libopencv_cudaarithm)：关闭 USE_OPENCV_CUDA 与 cuda 链接库。"
  cp -f "$PRO" "$PRO.orig_$(date +%Y%m%d_%H%M%S)"
  sed -i 's|^DEFINES += USE_OPENCV_CUDA|#DEFINES += USE_OPENCV_CUDA  # 部署时自动关闭(无CUDA版OpenCV)|' "$PRO"
  sed -i '/-lopencv_cuda\(arithm\|imgproc\|filters\|warping\)/d' "$PRO"
fi

# ---------- 5. 编译 ----------
BUILD="$DEST/build"
mkdir -p "$BUILD"
log "qmake: $QMAKE_BIN"
( cd "$BUILD" && "$QMAKE_BIN" ../CameraViewer.pro -spec linux-g++ CONFIG+=release ) || die "qmake 失败（检查 Qt6 是否安装/路径是否正确）"
log "make -j$(nproc) ..."
( cd "$BUILD" && make -j"$(nproc)" ) || die "编译失败（多为 OpenCV/CUDA/TensorRT 路径或版本问题，见上方错误）"
[ -x "$BUILD/CameraViewer" ] || die "未生成可执行文件: $BUILD/CameraViewer"
ok "编译完成: $BUILD/CameraViewer"

# ---------- 6. 运行期文件：模型 / style.qss / config.ini ----------
mkdir -p "$BUILD/model"
log "拷贝模型文件到 $BUILD/model ..."
cp -f "$DEST"/model/* "$BUILD/model/" 2>/dev/null || true
cp -f "$DEST"/style.qss "$BUILD/" 2>/dev/null || true

CFG_SRC="$SCRIPT_DIR/config.ini"
if [ -f "$CFG_SRC" ]; then
  if [ ! -f "$BUILD/config.ini" ] || [ "${FORCE_CFG:-0}" = 1 ]; then
    cp -f "$CFG_SRC" "$BUILD/config.ini"
    ok "已写入 config.ini"
  else
    log "build/config.ini 已存在，保留（覆盖请：FORCE_CFG=1 重跑）"
  fi
fi
[ -f "$BUILD/config.ini" ] || warn "缺少 build/config.ini，程序将使用内置默认值。"

# ---------- 7. 运行 ----------
if [ "$DO_RUN" = 1 ]; then
  if [ -z "${DISPLAY:-}" ]; then
    warn "未检测到 DISPLAY（当前可能是 SSH 会话）。GUI 需要图形桌面："
    warn "   请在 88 的桌面终端运行，或  DISPLAY=:0 $BUILD/CameraViewer"
    exit 0
  fi
  log "启动 CameraViewer（工作目录=$BUILD）"
  cd "$BUILD"
  exec ./CameraViewer
else
  log "已跳过运行。手动启动： cd $BUILD && ./CameraViewer"
fi
