#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于人工确认配对（manual_pairs.json）增量追加 ReID 数据集
=========================================================
与 rebuild_dataset_manual.py 的区别：本脚本【不清空】既有身份，
从现有最大编号 +1 继续追加新配对（并查集把 board↔alight 合并为同一身份）。

- 读 manual_pairs.json（人工确认的 board↔alight 配对）
- 并查集把每对合并为同一身份
- 追加导出到 passenger_reid_dataset（从 passenger_091 起）
- 更新 manifest.json（追加，不覆盖旧记录）

用法（yolotest 环境）：
  /home/tianheng/miniconda3/envs/yolotest/bin/python append_pairs_dataset.py \
      --manual /media/tianheng/test/16.reId/manual_pairs_new.json \
      --sessions <session目录1> <session目录2> ... \
      --out /media/tianheng/test/16.reId/passenger_reid_dataset \
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
    """并查集。"""

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


def existing_max_id(out):
    mx = 0
    if not os.path.isdir(out):
        return mx
    for name in os.listdir(out):
        m = re.match(r'^passenger_(\d+)$', name)
        if m and os.path.isdir(os.path.join(out, name)):
            mx = max(mx, int(m.group(1)))
    return mx


def main():
    ap = argparse.ArgumentParser(description='按人工配对增量追加 ReID 数据集')
    ap.add_argument('--manual', required=True, help='manual_pairs.json 路径（人工确认配对）')
    ap.add_argument('--sessions', nargs='+', required=True, help='一个或多个 session 目录')
    ap.add_argument('--out', default='/media/tianheng/test/16.reId/passenger_reid_dataset')
    ap.add_argument('--min-images', type=int, default=10, help='每个身份最少图片数')
    ap.add_argument('--dry-run', action='store_true', help='只打印计划，不实际写文件')
    args = ap.parse_args()

    manual = json.load(open(args.manual, encoding='utf-8'))

    # 收集所有 person（跨 session）
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
            if not files:
                continue
            persons[name] = {'session': sess, 'files': files}

    # 校验：配对双方都应在扫描到的 person 中，否则跳过该对
    valid_pairs = []
    for p in manual:
        b, a = p['board'], p['alight']
        if b not in persons:
            print(f'[warn] 配对 person 不在 sessions 中，跳过: {b}')
            continue
        if a not in persons:
            print(f'[warn] 配对 person 不在 sessions 中，跳过: {a}')
            continue
        valid_pairs.append(p)
    if not valid_pairs:
        print('没有有效的配对可追加')
        return 1
    manual = valid_pairs

    # 并查集合并人工配对
    dsu = DSU()
    for p in manual:
        dsu.union(p['board'], p['alight'])
    for name in persons:
        dsu.find(name)

    groups = defaultdict(list)
    for name in sorted(persons):
        groups[dsu.find(name)].append(name)

    paired_names = set()
    for p in manual:
        paired_names.add(p['board'])
        paired_names.add(p['alight'])

    start_id = existing_max_id(args.out) + 1
    next_id = start_id

    plan = []  # (pid, names, all_files)
    for root in sorted(groups.keys(), key=lambda r: -len(groups[r])):
        names = sorted(groups[root])
        # 只处理「与本次配对相关」的组（跳过其它独立 person）
        if not any(n in paired_names for n in names):
            continue
        all_files = []
        for name in names:
            all_files.extend(persons[name]['files'])
        if len(all_files) < args.min_images:
            print(f'[skip] {"+".join(names)} 仅 {len(all_files)} 张 < {args.min_images}')
            continue
        plan.append((f'passenger_{next_id:03d}', names, all_files))
        next_id += 1

    print('=' * 70)
    print(f'将从 passenger_{start_id:03d} 起追加 {len(plan)} 个身份')
    for pid, names, files in plan:
        print(f'  {pid}: {len(files)} 张 <- {" + ".join(names)}')

    if args.dry_run:
        print('[dry-run] 未写入任何文件')
        return 0

    if not plan:
        print('无可追加的身份')
        return 0

    # 追加导出
    for pid, names, all_files in plan:
        dst = os.path.join(args.out, pid)
        os.makedirs(dst, exist_ok=True)
        used = set()
        copied = 0
        for fp in all_files:
            person = os.path.basename(os.path.dirname(os.path.dirname(fp)))
            fn = f'{person}__{os.path.basename(fp)}'
            base, ext = os.path.splitext(fn)
            target = os.path.join(dst, fn)
            k = 1
            while target in used:
                target = os.path.join(dst, f'{base}_{k}{ext}')
                k += 1
            used.add(target)
            shutil.copy2(fp, target)
            copied += 1
        print(f'{pid}: {copied} 张 <- {" + ".join(names)}')

    # 更新 manifest（追加，不覆盖旧 identities）
    manifest_path = os.path.join(args.out, 'manifest.json')
    manifest = json.load(open(manifest_path, encoding='utf-8')) if os.path.isfile(manifest_path) else {}
    identities = manifest.setdefault('identities', {})
    for pid, names, _ in plan:
        identities[pid] = names
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    total_ids = sum(1 for n in os.listdir(args.out)
                    if re.match(r'^passenger_\d+$', n) and os.path.isdir(os.path.join(args.out, n)))
    total_imgs = sum(len(list_images(os.path.join(args.out, d))) for d in os.listdir(args.out)
                     if re.match(r'^passenger_\d+$', d) and os.path.isdir(os.path.join(args.out, d)))
    print('=' * 70)
    print(f'追加完成：数据集 {args.out} 现共 {total_ids} 个身份 / {total_imgs} 张图')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
