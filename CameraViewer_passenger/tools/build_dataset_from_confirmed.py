#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于人工确认配对（confirmed_pairs.json）新建 ReID 训练数据集（重头训练）
======================================================================
- 读取 match_review.html 导出的 confirmed_pairs.json（人工确认「同一人」的配对）
- 用并查集把每对 person_a + person_b 合并为同一身份
- 导出到【全新】数据集目录（默认 passenger_reid_dataset_v2，不碰旧数据集，
  避免 append 增量追加导致的历史重复数据污染训练）

用法（yolotest 环境）：
  /home/tianheng/miniconda3/envs/yolotest/bin/python build_dataset_from_confirmed.py \
      --pairs confirmed_pairs.json \
      --sessions <session目录> \
      --out /media/tianheng/test/16.reId/CameraViewer_passenger/passenger_reid_dataset_v2 \
      --min-images 10 [--dry-run]
"""
import argparse
import json
import os
import re
import shutil
from collections import defaultdict

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
PERSON_RE = re.compile(r'^person_\d+_')


class DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def list_images(crop_dir):
    if not os.path.isdir(crop_dir):
        return []
    return sorted(os.path.join(crop_dir, f) for f in os.listdir(crop_dir)
                  if f.lower().endswith(IMAGE_EXTS))


def main():
    ap = argparse.ArgumentParser(description='按人工确认配对新建 ReID 数据集')
    ap.add_argument('--pairs', required=True, help='confirmed_pairs.json 路径')
    ap.add_argument('--sessions', nargs='+', required=True, help='一个或多个 session 目录')
    ap.add_argument('--out', default='/media/tianheng/test/16.reId/CameraViewer_passenger/passenger_reid_dataset_v2')
    ap.add_argument('--min-images', type=int, default=10, help='每个身份最少图片数')
    ap.add_argument('--dry-run', action='store_true', help='只打印计划，不写文件')
    args = ap.parse_args()

    pairs = json.load(open(args.pairs, encoding='utf-8'))
    print(f'读取 {len(pairs)} 对人工确认配对')

    # 收集所有 person 图片
    persons = {}
    for sess in args.sessions:
        if not os.path.isdir(sess):
            print(f'[warn] session 不存在: {sess}')
            continue
        for name in sorted(os.listdir(sess)):
            pdir = os.path.join(sess, name)
            if not os.path.isdir(pdir) or not PERSON_RE.match(name):
                continue
            files = list_images(os.path.join(pdir, '乘客图片'))
            if files:
                persons[name] = {'session': sess, 'files': files}

    # 校验配对双方存在
    valid = []
    for p in pairs:
        a, b = p['person_a'], p['person_b']
        if a not in persons:
            print(f'[warn] 跳过（person 不在 sessions）: {a}')
            continue
        if b not in persons:
            print(f'[warn] 跳过（person 不在 sessions）: {b}')
            continue
        valid.append(p)
    if not valid:
        print('无有效配对')
        return 1

    # 并查集合并配对
    dsu = DSU()
    for p in valid:
        dsu.union(p['person_a'], p['person_b'])
    for name in persons:
        dsu.find(name)

    groups = defaultdict(list)
    for name in sorted(persons):
        groups[dsu.find(name)].append(name)

    # 只导出「包含确认配对」的组（忽略其它未配对的 person）
    paired = set()
    for p in valid:
        paired.add(p['person_a'])
        paired.add(p['person_b'])

    plan = []
    for root in sorted(groups.keys(), key=lambda r: -len(groups[r])):
        names = sorted(groups[root])
        if not any(n in paired for n in names):
            continue
        all_files = []
        for name in names:
            all_files.extend(persons[name]['files'])
        if len(all_files) < args.min_images:
            print(f'[skip] {"+".join(names)} 仅 {len(all_files)} 张 < {args.min_images}')
            continue
        plan.append((names, all_files))

    print('=' * 70)
    print(f'将新建数据集 {args.out}，导出 {len(plan)} 个身份')
    for i, (names, files) in enumerate(plan, 1):
        print(f'  passenger_{i:03d}: {len(files)} 张 <- {" + ".join(names)}')

    if args.dry_run:
        print('[dry-run] 未写入')
        return 0

    # 新建数据集目录（清空旧内容，重头训练）
    if os.path.isdir(args.out):
        print(f'[warn] 目标目录已存在，将清空重建: {args.out}')
        shutil.rmtree(args.out)
    os.makedirs(args.out)

    manifest = {'sources': {}, 'identities': {}}
    total = 0
    for i, (names, all_files) in enumerate(plan, 1):
        pid = f'passenger_{i:03d}'
        dst = os.path.join(args.out, pid)
        os.makedirs(dst)
        copied = 0
        for fp in all_files:
            person = os.path.basename(os.path.dirname(os.path.dirname(fp)))
            fn = f'{person}__{os.path.basename(fp)}'
            base, ext = os.path.splitext(fn)
            target = os.path.join(dst, fn)
            k = 1
            while os.path.exists(target):
                target = os.path.join(dst, f'{base}_{k}{ext}')
                k += 1
            shutil.copy2(fp, target)
            copied += 1
        manifest['identities'][pid] = names
        total += copied
        print(f'{pid}: {copied} 张 <- {" + ".join(names)}')

    with open(os.path.join(args.out, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print('=' * 70)
    print(f'完成：新建数据集 {args.out} 共 {len(plan)} 个身份 / {total} 张图')
    print('下一步训练：')
    print(f'  python finetune_osnet_dataset.py --root {args.out} '
          f'--weights <msmt17 或 osnet_best.pth> --epochs 60 --p 8 --k 4')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
