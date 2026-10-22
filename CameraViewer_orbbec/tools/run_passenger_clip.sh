#!/bin/sh
# ============================================================================
# run_passenger_clip.sh —— 一键构建并运行「乘客上下车片段」离线批处理
#
# 用法：
#   ./tools/run_passenger_clip.sh [输入目录] [输出目录]
#
# 默认：
#   输入 = /run/user/1000/gvfs/sftp:host=192.168.200.8/share/vision/25.乘客视频
#   输出 = <项目根>/passenger_clips
#
# 说明：
#   - 输入目录含 .rar(含分卷) 时，工具会自动解压（需要 unrar/unar/7z）。
#     项目内已安装 unrar 于 .conda_envs/unrar/bin，本脚本自动加入 PATH。
#   - 输入目录已解压视频时，可直接传该目录。
# ============================================================================
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build-passenger-tool"
QMAKE="${QMAKE:-/home/froduo/Qt/6.2.4/gcc_64/bin/qmake}"

IN="${1:-/run/user/1000/gvfs/sftp:host=192.168.200.8/share/vision/25.乘客视频}"
OUT="${2:-$ROOT/passenger_clips}"

# 让工具能找到 RAR5 解压器（项目内 conda 环境安装的 unrar）
if [ -d "$ROOT/.conda_envs/unrar/bin" ]; then
    PATH="$ROOT/.conda_envs/unrar/bin:$PATH"
    export PATH
fi

echo "[run] 项目根 : $ROOT"
echo "[run] 输入目录: $IN"
echo "[run] 输出目录: $OUT"

mkdir -p "$BUILD"
cd "$BUILD"
if [ ! -f Makefile ]; then
    "$QMAKE" "$ROOT/tools/passenger_clip_tool.pro"
fi
make -j"$(nproc)"

exec "$BUILD/passenger_clip_tool" \
    --input  "$IN" \
    --output "$OUT" \
    --model  "$ROOT/model/yolo26n_trt10_x86.engine" \
    --deep   "$ROOT/model/osnet_x1_0_head_finetuned.onnx" \
    --gap 12 --pad 3
