#!/bin/bash
# =============================================================================
# 训练完成检测 + 一键部署脚本
# -----------------------------------------------------------------------------
# 等待 finetune_osnet_dataset.py 训练进程结束后，自动：
#   1) 复制导出的 osnet_x1_0_finetuned.onnx 到「源码 + 构建」两个 model 目录
#   2) 更新 build config.ini 的 DeepModelPath 指向新 ONNX
#   3) 删除旧的 osnet_x1_0_head_finetuned.engine（让程序自动重建 TensorRT 引擎）
#
# 用法：
#   ./deploy_finetuned_model.sh          # 前台：等待训练结束并部署
#   nohup ./deploy_finetuned_model.sh > /tmp/deploy_finetuned.log 2>&1 &   # 后台
# =============================================================================
set -u

ROOT="/media/tianheng/test/16.reId"
APP="$ROOT/CameraViewer_passenger"
BUILD="$ROOT/build-CameraViewer_passenger-Desktop-Debug"
DATASET="$ROOT/passenger_reid_dataset"

SRC_MODEL="$APP/model"
BUILD_MODEL="$BUILD/model"
CONFIG="$BUILD/config.ini"
ONNX="$DATASET/osnet_x1_0_finetuned.onnx"
TRAIN_LOG="/tmp/finetune_train2.log"
TRAIN_PATTERN="finetune_osnet_dataset.py"

log() { echo "[deploy] $(date '+%m-%d %H:%M:%S') $*"; }

# ---- 1. 等待训练进程结束 ----
if pgrep -f "$TRAIN_PATTERN" > /dev/null 2>&1; then
    log "检测到训练进程仍在运行，等待其结束（每 20s 检测一次）..."
    while pgrep -f "$TRAIN_PATTERN" > /dev/null 2>&1; do
        sleep 20
    done
fi
log "训练进程已结束。"

# ---- 2. 校验导出的 ONNX ----
if [ ! -f "$ONNX" ]; then
    log "错误：未找到导出的 ONNX：$ONNX" >&2
    log "请检查训练日志：$TRAIN_LOG" >&2
    exit 1
fi
log "找到新 ONNX：$ONNX ($(stat -c%s "$ONNX") bytes)"

# ---- 3. 复制 ONNX 到两个 model 目录 ----
cp -f "$ONNX" "$SRC_MODEL/osnet_x1_0_finetuned.onnx"
cp -f "$ONNX" "$BUILD_MODEL/osnet_x1_0_finetuned.onnx"
log "已复制 ONNX 到："
log "  $SRC_MODEL/osnet_x1_0_finetuned.onnx"
log "  $BUILD_MODEL/osnet_x1_0_finetuned.onnx"

# ---- 4. 更新 config.ini 的 DeepModelPath ----
if grep -q '^DeepModelPath=' "$CONFIG"; then
    sed -i 's|^DeepModelPath=.*|DeepModelPath=./model/osnet_x1_0_finetuned.onnx|' "$CONFIG"
    log "已更新 config.ini DeepModelPath："
    grep '^DeepModelPath=' "$CONFIG" | sed 's/^/  /'
else
    log "警告：config.ini 未找到 DeepModelPath 行，请手动检查：$CONFIG" >&2
fi

# ---- 5. 删除旧的 head_finetuned engine（让程序重建 TensorRT） ----
OLD_ENGINE="$BUILD_MODEL/osnet_x1_0_head_finetuned.engine"
if [ -f "$OLD_ENGINE" ]; then
    rm -f "$OLD_ENGINE"
    log "已删除旧 engine：$OLD_ENGINE"
else
    log "未发现旧 engine（无需删除）"
fi

# ---- 6. 提示训练验证结果（供阈值标定参考） ----
if [ -f "$TRAIN_LOG" ]; then
    BEST=$(grep -oE 'val same=[0-9.]+ diff=[0-9.]+ gap=[+-][0-9.]+' "$TRAIN_LOG" | tail -5)
    if [ -n "$BEST" ]; then
        log "训练验证结果（最近几轮）："
        echo "$BEST" | sed 's/^/  /'
        log "提示：如需重标定 DeepMatchThreshold，请根据 same/diff 分布人工确定"
    fi
fi

log "部署完成。重启程序即可加载新模型（首次运行会自动重建 TensorRT engine）。"
