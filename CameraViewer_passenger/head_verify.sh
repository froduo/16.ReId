#!/bin/bash
# 模型识别准确性验证工具 —— 启动脚本
cd "$(dirname "$0")" || exit 1
exec python3 head_verify_gui.py "$@"
