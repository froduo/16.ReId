#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出 ReID 种子数据集：
  把人工确认的乘客目录（如 B/C/D 的 copy 文件夹）复制到统一的 ReID 数据集目录，
  自动编号为 passenger_001 / passenger_002 / ... ，支持增量追加。

特性：
  - 自动扫描已有 passenger_NNN，从最大编号+1 继续编号；
  - manifest.json 记录「源目录 -> passenger_NNN」映射，重复导入同一目录会自动跳过；
  - 复制时保留原文件名，重名自动加后缀，避免覆盖。

用法（yolotest 环境）：
  /home/tianheng/miniconda3/envs/yolotest/bin/python export_seed_dataset.py \
      --out /path/to/reid_dataset \
      --src "目录B" "目录C" "目录D"
"""
import argparse
import json
import os
import re
import shutil

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def existing_max_id(out):
    """扫描 out 下已有的 passenger_NNN 目录，返回最大编号（0 表示还没有）。"""
    if not os.path.isdir(out):
        return 0
    mx = 0
    for name in os.listdir(out):
        m = re.match(r'^passenger_(\d+)$', name)
        if m and os.path.isdir(os.path.join(out, name)):
            mx = max(mx, int(m.group(1)))
    return mx


def main():
    ap = argparse.ArgumentParser(description='导出 ReID 种子数据集（增量追加）')
    ap.add_argument('--out', required=True, help='数据集根目录')
    ap.add_argument('--src', nargs='+', required=True, help='人工确认的乘客目录（可多个）')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, 'manifest.json')
    manifest = {}
    if os.path.isfile(manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
    sources = manifest.setdefault('sources', {})

    next_id = existing_max_id(args.out) + 1

    for src in args.src:
        src = os.path.abspath(src)
        if not os.path.isdir(src):
            print(f'跳过（目录不存在）: {src}')
            continue
        if src in sources:
            print(f'跳过（已导入过 -> {sources[src]}）: {src}')
            continue

        pid = f'passenger_{next_id:03d}'
        dst = os.path.join(args.out, pid)
        os.makedirs(dst, exist_ok=True)

        files = sorted(f for f in os.listdir(src)
                       if f.lower().endswith(IMAGE_EXTS)
                       and os.path.isfile(os.path.join(src, f)))
        copied = 0
        for fn in files:
            target = os.path.join(dst, fn)
            base, ext = os.path.splitext(fn)
            k = 1
            while os.path.exists(target):           # 重名保护
                target = os.path.join(dst, f'{base}_{k}{ext}')
                k += 1
            shutil.copy2(os.path.join(src, fn), target)
            copied += 1

        sources[src] = pid
        print(f'{pid}: {copied} 张  <-  {os.path.basename(src)}')
        next_id += 1

    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print('-' * 60)
    print(f'数据集目录: {args.out}')
    for pid in sorted(k for k in os.listdir(args.out)
                      if re.match(r'^passenger_\d+$', k) and os.path.isdir(os.path.join(args.out, k))):
        n = len([x for x in os.listdir(os.path.join(args.out, pid))
                 if x.lower().endswith(IMAGE_EXTS)])
        print(f'  {pid}: {n} 张')
    print(f'manifest 已更新: {manifest_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
