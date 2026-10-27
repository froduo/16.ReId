#!/bin/bash
# =============================================================================
# ReID 全自动流水线：训练完成 → 部署模型 → 依次检测视频 → 比对 → 准确度评估
# -----------------------------------------------------------------------------
# 流程：
#   1) 等待 finetune_osnet_dataset.py 训练结束
#   2) 部署训练导出的 osnet_x1_0_finetuned.onnx 到程序 model 目录 + 更新 config.ini
#   3) 备份旧 session_state.json（让程序新建干净 session 目录）
#   4) 依次用 CameraViewer_passenger --autoplay 检测 passenger_videos/*.mp4（6x 倍速）
#   5) 用 osnet_best.pth 对生成的 person 目录做上车↔下车独立比对
#   6) 生成准确度评估报告 + 下一步完善方案
#
# 无人干预运行（后台）：
#   nohup bash auto_reid_pipeline.sh > /tmp/auto_pipeline_stdout.log 2>&1 &
#
# 断点续跑：已完成的视频记录在 $WORK/completed_videos.txt，重复执行会跳过。
# =============================================================================
set -u

ROOT="/media/tianheng/test/16.reId"
APP="$ROOT/CameraViewer_passenger"
BUILD="$ROOT/build-CameraViewer_passenger-Desktop-Debug"
EXE="$BUILD/CameraViewer_passenger"
VIDEOS_DIR="$APP/passenger_videos"
DATASET="$ROOT/passenger_reid_dataset"
TOOLS="$APP/tools"
PYTHON="/home/tianheng/miniconda3/envs/yolotest/bin/python"

TRAIN_PATTERN="finetune_osnet_dataset.py"
ONNX="$DATASET/osnet_x1_0_finetuned.onnx"
BEST_PTH="$DATASET/osnet_best.pth"
CONFIG="$BUILD/config.ini"
BUILD_MODEL="$BUILD/model"
SRC_MODEL="$APP/model"

DISPLAY_ENV="${DISPLAY:-:1}"

WORK="$ROOT/auto_pipeline"
mkdir -p "$WORK"
PIPE_LOG="$WORK/pipeline.log"
DONE_FILE="$WORK/completed_videos.txt"
touch "$DONE_FILE"

log() { echo "[pipeline] $(date '+%m-%d %H:%M:%S') $*" | tee -a "$PIPE_LOG"; }

# ===== 阶段0：单实例检查（精确匹配可执行文件完整路径，避免误匹配脚本自身/训练进程） =====
if pgrep -f "$EXE" > /dev/null 2>&1; then
    log "错误：CameraViewer_passenger 正在运行，请先关闭再跑流水线"
    exit 1
fi

# ===== 阶段1：等待训练结束 =====
if pgrep -f "$TRAIN_PATTERN" > /dev/null 2>&1; then
    log "检测到训练进程仍在运行，等待其结束（每 60s 检测一次）..."
    while pgrep -f "$TRAIN_PATTERN" > /dev/null 2>&1; do sleep 60; done
fi
log "训练已结束"

# ===== 阶段2：部署模型 =====
log "===== 部署模型 ====="
if [ ! -f "$ONNX" ]; then
    log "错误：未找到训练导出的 ONNX：$ONNX"
    exit 1
fi
cp -f "$ONNX" "$SRC_MODEL/osnet_x1_0_finetuned.onnx"
cp -f "$ONNX" "$BUILD_MODEL/osnet_x1_0_finetuned.onnx"
if grep -q '^DeepModelPath=' "$CONFIG"; then
    sed -i 's|^DeepModelPath=.*|DeepModelPath=./model/osnet_x1_0_finetuned.onnx|' "$CONFIG"
else
    echo "DeepModelPath=./model/osnet_x1_0_finetuned.onnx" >> "$CONFIG"
fi
rm -f "$BUILD_MODEL/osnet_x1_0_head_finetuned.engine"
log "已部署新 ONNX 并更新 config.ini DeepModelPath"

# ===== 阶段3：准备干净 session 目录 =====
log "===== 准备干净 session 目录 ====="
SESSION_STATE="$BUILD/visitor_log/session_state.json"
if [ -f "$SESSION_STATE" ]; then
    mv "$SESSION_STATE" "$WORK/session_state.bak_$(date +%H%M%S).json"
    log "已备份旧 session_state.json（程序将新建 session 目录）"
else
    log "无旧 session_state.json，直接新建"
fi

# ===== 阶段4：依次检测视频 =====
log "===== 依次检测视频 ====="
mapfile -t VIDEOS < <(ls "$VIDEOS_DIR"/*.mp4 2>/dev/null | sort)
if [ "${#VIDEOS[@]}" -eq 0 ]; then
    log "错误：$VIDEOS_DIR 下没有 .mp4 视频"
    exit 1
fi
log "共 ${#VIDEOS[@]} 个视频待检测"
for v in "${VIDEOS[@]}"; do
    bname=$(basename "$v")
    if grep -qF "$bname" "$DONE_FILE" 2>/dev/null; then
        log "跳过已完成: $bname"
        continue
    fi
    log "开始检测: $bname"
    t0=$(date +%s)
    DISPLAY="$DISPLAY_ENV" "$EXE" --autoplay "$v" 6 >> "$PIPE_LOG" 2>&1
    code=$?
    t1=$(date +%s)
    log "检测完成: $bname (退出码 $code, 耗时 $(( (t1-t0)/60 )) 分 $(( (t1-t0)%60 )) 秒)"
    if [ "$code" -eq 0 ]; then
        echo "$bname" >> "$DONE_FILE"
    else
        log "警告：退出码非 0，不标记完成，下次可重跑"
    fi
done

# ===== 阶段5：定位最新 session 目录 =====
log "===== 定位 session 目录 ====="
SESSION=$(ls -td "$BUILD/save_images"/session_* 2>/dev/null | head -1)
if [ -z "$SESSION" ]; then
    log "错误：未找到 session 目录（$BUILD/save_images/session_*）"
    exit 1
fi
log "最新 session: $SESSION"

# ===== 阶段6：OSNet 独立比对 =====
log "===== OSNet 独立比对 ====="
"$PYTHON" "$TOOLS/match_person_folders.py" \
    --sessions "$SESSION" \
    --model "$BEST_PTH" \
    --out "$WORK/match_result.json" \
    --threshold 0.60 >> "$PIPE_LOG" 2>&1
log "比对完成 → $WORK/match_result.json"

# ===== 阶段7：准确度评估报告 =====
log "===== 生成准确度评估报告 ====="
"$PYTHON" "$TOOLS/evaluate_accuracy.py" \
    --session "$SESSION" \
    --match "$WORK/match_result.json" \
    --out "$WORK/accuracy_report.txt" >> "$PIPE_LOG" 2>&1
log "评估报告 → $WORK/accuracy_report.txt"

log "===== 全部完成 ====="
log "报告目录: $WORK"
log "  匹配结果: $WORK/match_result.json"
log "  准确度报告: $WORK/accuracy_report.txt"
log "  运行日志: $PIPE_LOG"
