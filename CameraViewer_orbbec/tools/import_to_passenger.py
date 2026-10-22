#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增量导入：把批量识别生成的「图片 + YOLO 标注」导入 12.ultralytics 平台的数据目录，
使网页标注界面（http://localhost:8000/ui/）中「passenger」项目能浏览并人工审核标记框。

平台项目机制：
  - 图片/标注统一存放在 backend/data/images/{split}/ 与 backend/data/labels/{split}/；
  - 通过文件名前缀 __proj_passenger__ 归属到 passenger 项目；
  - 前端项目列表来自浏览器 localStorage（需在 UI 里新建一次 passenger 项目）。

本脚本持续运行：每 --interval 秒扫描一次识别输出，发现新完成的视频数据就增量导入。

用法:
  python3 -u tools/import_to_passenger.py [--interval 10]
"""

import argparse
import os
import shutil
import time

PLATFORM_DATA = "/media/tianheng/test/12.ultralytics/ultralytics/backend/data"
PROJECT_NAME = "passenger"
MARKER = "__proj_passenger__"
SPLIT = "train"

SINGLE_STEM = "ch01_20260930_063419_074220topspeed"
SINGLE_ROOT = "passenger_dataset"
BATCH_ROOT = "passenger_dataset_all"

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
MIN_AGE_SEC = 5   # 只导入「写入完成超过 5 秒」的文件，避免复制到半成品


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def import_dir(stem, root, img_dir, lbl_dir):
    src_img = os.path.join(root, "images")
    src_lbl = os.path.join(root, "labels")
    if not os.path.isdir(src_img):
        return 0

    cnt = 0
    for fn in sorted(os.listdir(src_img)):
        if not fn.lower().endswith(IMG_EXTS):
            continue
        src = os.path.join(src_img, fn)
        if time.time() - os.path.getmtime(src) < MIN_AGE_SEC:
            continue
        dst = os.path.join(img_dir, f"{MARKER}{stem}_{fn}")
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            cnt += 1
        # 对应标注
        label_fn = os.path.splitext(fn)[0] + ".txt"
        src_txt = os.path.join(src_lbl, label_fn)
        dst_txt = os.path.join(lbl_dir, f"{MARKER}{stem}_{label_fn}")
        if os.path.exists(src_txt) and not os.path.exists(dst_txt):
            shutil.copy2(src_txt, dst_txt)
            cnt += 1
    return cnt


def collect_sources():
    sources = []
    if os.path.isdir(SINGLE_ROOT):
        sources.append((SINGLE_STEM, SINGLE_ROOT))
    if os.path.isdir(BATCH_ROOT):
        for stem in sorted(os.listdir(BATCH_ROOT)):
            p = os.path.join(BATCH_ROOT, stem)
            if os.path.isdir(p) and os.path.isdir(os.path.join(p, "images")):
                sources.append((stem, p))
    return sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=10)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    img_dir = os.path.join(PLATFORM_DATA, "images", SPLIT)
    lbl_dir = os.path.join(PLATFORM_DATA, "labels", SPLIT)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    log(f"平台数据目录: {PLATFORM_DATA} (split={SPLIT}, 项目={PROJECT_NAME}, 前缀={MARKER})")

    while True:
        sources = collect_sources()
        total = 0
        for stem, root in sources:
            total += import_dir(stem, root, img_dir, lbl_dir)
        if total:
            log(f"新增导入 {total} 个文件")
        if args.once:
            break
        time.sleep(args.interval)

    log(f"完成，当前 passenger 项目 train 集图片 {sum(1 for n in os.listdir(img_dir) if n.startswith(MARKER))} 张, "
        f"标注 {sum(1 for n in os.listdir(lbl_dir) if n.startswith(MARKER))} 个")


if __name__ == "__main__":
    main()
