#!/bin/sh
# ============================================================================
# copy_models.sh —— 将模型文件复制到构建输出目录
#
# 用法: copy_models.sh <src_dir> <dest_dir>
#   - 检测模型: head_yolo11n_trt10_x86.engine / head_yolo11n.onnx
#   - 乘客检测: passenger_yolo11n_trt10_x86.engine / passenger_yolo11n.onnx
#               （YOLO11n 单类 passenger，1x3x640x640 -> 1x5x8400）
#   - ReID 模型: osnet_x1_0_head_finetuned.onnx / osnet_x1_0_msmt17.onnx
#
# 仅复制存在的文件；目标目录不存在则创建。
# ============================================================================

set -eu

SRC="${1:-}"
DEST="${2:-}"

if [ -z "$SRC" ] || [ -z "$DEST" ]; then
    echo "[copy_models] 用法: copy_models.sh <src_dir> <dest_dir>" >&2
    exit 1
fi

mkdir -p "$DEST"

for m in \
    head_yolo11n_trt10_x86.engine \
    head_yolo11n.onnx \
    passenger_yolo11n_trt10_x86.engine \
    passenger_yolo11n.onnx \
    yolo26n_trt10_x86.engine \
    yolov8n.onnx \
    osnet_x1_0_head_finetuned.onnx \
    osnet_x1_0_msmt17.onnx
do
    if [ -f "$SRC/$m" ]; then
        cp -f "$SRC/$m" "$DEST/$m"
        echo "[copy_models] $m -> $DEST"
    else
        echo "[copy_models] 跳过(缺失): $SRC/$m" >&2
    fi
done
