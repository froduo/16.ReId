#!/bin/bash
# =============================================================================
# OSNet 重启训练 + 质量对比 + 条件部署 + 报告
# -----------------------------------------------------------------------------
# 1) 在 passenger_reid_dataset_v3 上重启训练（从 v3/osnet_best.pth 续训）
# 2) 解析训练 best gap（same - diff，越大越好）
# 3) 与基线 gap（上一次训练）比较
# 4) 有提升则部署新 onnx（复制 model 目录 + 更新 config.ini + 删旧 engine）
# 5) 输出训练报告 auto_pipeline/training_report.txt（供次日人工阅读）
#
# 用法：
#   bash tools/retrain_and_deploy.sh            # 立即执行
#   （由 cron 在指定时间触发，见 retrain 调度）
# =============================================================================
set -u

ROOT="/media/tianheng/test/16.reId"
APP="$ROOT/CameraViewer_passenger"
BUILD="$ROOT/build-CameraViewer_passenger-Desktop-Debug"
TOOLS="$APP/tools"
DATASET="$ROOT/CameraViewer_passenger/passenger_reid_dataset_v3"
PYTHON="/home/tianheng/miniconda3/envs/yolotest/bin/python"

WORK="$ROOT/CameraViewer_passenger/auto_pipeline"
mkdir -p "$WORK"
BASELINE_FILE="$WORK/baseline_gap.txt"
REPORT="$WORK/training_report.txt"
TS=$(date +%Y%m%d_%H%M%S)
TRAIN_LOG="/tmp/retrain_${TS}.log"

ONNX="$DATASET/osnet_x1_0_finetuned.onnx"
SRC_MODEL="$APP/model"
BUILD_MODEL="$BUILD/model"
CONFIG="$BUILD/config.ini"

# 一次性任务：执行后立即移除本脚本的 crontab 条目，避免重复执行
crontab -l 2>/dev/null | grep -v 'retrain_and_deploy.sh' | crontab - 2>/dev/null

log() { echo "[retrain] $(date '+%m-%d %H:%M:%S') $*" | tee -a "$REPORT"; }

# ---- 读取基线（上一次训练 best gap）----
BASELINE="0.310"
if [ -f "$BASELINE_FILE" ]; then
    BASELINE=$(tr -d '[:space:]' < "$BASELINE_FILE")
fi

: > "$REPORT"
log "=========================================================="
log "OSNet 重启训练报告"
log "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
log "数据集: $DATASET"
log "续训权重: $DATASET/osnet_best.pth"
log "基线 gap（上一次训练）: $BASELINE"
log "=========================================================="

log "训练前 GPU 状态:"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/    /' | tee -a "$REPORT"

# ---- 训练 ----
log "开始训练（60 epochs，P=8 K=4，lr=3.5e-4）..."
"$PYTHON" "$TOOLS/finetune_osnet_dataset.py" \
    --root "$DATASET" \
    --weights "$DATASET/osnet_best.pth" \
    --epochs 60 --p 8 --k 4 > "$TRAIN_LOG" 2>&1
TRAIN_EXIT=$?
log "训练进程结束，退出码=$TRAIN_EXIT"

# ---- 解析 best gap（训练内 val 指标：gap = same - diff）----
NEW_GAP=$(grep -oE 'gap=[+-][0-9.]+' "$TRAIN_LOG" 2>/dev/null | sed 's/gap=//; s/^+//' | sort -g | tail -1)
if [ -z "$NEW_GAP" ]; then
    log "错误：未能从训练日志解析出 gap（训练可能失败）"
    log "训练日志: $TRAIN_LOG"
    log "日志末尾："
    tail -20 "$TRAIN_LOG" 2>/dev/null | sed 's/^/    /' | tee -a "$REPORT"
    exit 1
fi
log "本次训练 best gap = $NEW_GAP"
log "最近几轮验证指标（val same/diff/gap/idacc）："
grep 'val same=' "$TRAIN_LOG" 2>/dev/null | tail -3 | sed 's/^/    /' | tee -a "$REPORT"

# ---- 质量对比 ----
UPGRADE=0
if awk "BEGIN{exit !($NEW_GAP > $BASELINE)}"; then
    UPGRADE=1
    log "对比结果: 有提升（$NEW_GAP > $BASELINE）→ 更新模型"
else
    log "对比结果: 无提升（$NEW_GAP <= $BASELINE）→ 不更新模型"
fi

# ---- 部署（仅在有提升时）----
if [ "$UPGRADE" -eq 1 ]; then
    if [ -f "$ONNX" ]; then
        cp -f "$ONNX" "$SRC_MODEL/osnet_x1_0_finetuned.onnx"
        cp -f "$ONNX" "$BUILD_MODEL/osnet_x1_0_finetuned.onnx"
        if grep -q '^DeepModelPath=' "$CONFIG"; then
            sed -i 's|^DeepModelPath=.*|DeepModelPath=./model/osnet_x1_0_finetuned.onnx|' "$CONFIG"
        else
            echo "DeepModelPath=./model/osnet_x1_0_finetuned.onnx" >> "$CONFIG"
        fi
        rm -f "$BUILD_MODEL/osnet_x1_0_finetuned.engine"
        log "已更新模型文件：ONNX 已复制到 model/ 与 build/model/，config.ini 已更新，旧 engine 已删除"
    else
        log "警告：未找到导出 ONNX $ONNX，跳过部署"
        UPGRADE=0
    fi
fi

# ---- 更新基线（仅提升时更新，保持历史最佳）----
if [ "$UPGRADE" -eq 1 ]; then
    echo "$NEW_GAP" > "$BASELINE_FILE"
    log "基线已更新: $BASELINE_FILE -> $NEW_GAP"
else
    log "基线保持: $BASELINE_FILE = $BASELINE"
fi

log "=========================================================="
log "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
log "训练日志: $TRAIN_LOG"
log "报告文件: $REPORT"
log "=========================================================="
