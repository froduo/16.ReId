#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按 match_result.json 的高置信匹配自动合并身份，导出到 passenger_reid_dataset
========================================================================
- 读 match_result.json（match_person_folders.py 的输出）
- 高置信匹配（sim >= --min-sim，默认 0.85）用并查集合并为同一身份
- 低置信匹配 + 未匹配的 person 各自独立（留待人工确认）
- 每个身份收集其所有 person 的「乘客图片」，导出为 passenger_XXX（增量编号）
- 更新 manifest.json（复用 export_seed_dataset.py 的编号机制，从最大编号+1 继续）

用法（yolotest 环境）：
  /home/tianheng/miniconda3/envs/yolotest/bin/python export_matched_identities.py \
      --match /media/tianheng/test/16.reId/CameraViewer_passenger/match_result.json \
      --sessions <session目录1> <session目录2> ... \
      --out /media/tianheng/test/16.reId/CameraViewer_passenger/passenger_reid_dataset \
      --min-sim 0.85 --min-images 10
"""
import argparse
import json
import os
import re
import shutil
from collections import defaultdict

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


class DSU:
    """并查集：把高置信匹配的 person 合并为同一身份。"""

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


def existing_max_id(out):
    """扫描 out 下已有 passenger_NNN，返回最大编号（0 表示还没有）。"""
    mx = 0
    if os.path.isdir(out):
        for name in os.listdir(out):
            m = re.match(r'^passenger_(\d+)$', name)
            if m and os.path.isdir(os.path.join(out, name)):
                mx = max(mx, int(m.group(1)))
    return mx


def list_images(crop_dir):
    if not os.path.isdir(crop_dir):
        return []
    return sorted(os.path.join(crop_dir, f) for f in os.listdir(crop_dir)
                  if f.lower().endswith(IMAGE_EXTS))


def main():
    ap = argparse.ArgumentParser(description='按高置信匹配合并身份并导出 ReID 数据集')
    ap.add_argument('--match', required=True, help='match_result.json 路径')
    ap.add_argument('--sessions', nargs='+', required=True, help='一个或多个 session 目录')
    ap.add_argument('--out', default='/media/tianheng/test/16.reId/CameraViewer_passenger/passenger_reid_dataset')
    ap.add_argument('--min-sim', type=float, default=0.85, help='高置信匹配阈值，>= 此值自动合并')
    ap.add_argument('--min-images', type=int, default=10, help='每个身份最少图片数，低于则跳过')
    args = ap.parse_args()

    data = json.load(open(args.match, encoding='utf-8'))
    matches = data.get('matches', [])
    unmatched = data.get('unmatched', [])

    # name -> session 映射
    name2session = {}
    for m in matches:
        name2session[m['person_a']] = m['a_session']
        name2session[m['person_b']] = m['b_session']
    for name in unmatched:
        for sess in args.sessions:
            if os.path.isdir(os.path.join(sess, name)):
                name2session[name] = os.path.basename(sess.rstrip('/'))
                break

    sess_map = {os.path.basename(s.rstrip('/')): s for s in args.sessions}

    # 并查集合并高置信匹配
    dsu = DSU()
    all_names = set(name2session.keys())
    for m in matches:
        if m['similarity'] >= args.min_sim:
            dsu.union(m['person_a'], m['person_b'])
    for name in all_names:
        dsu.find(name)

    # 按根分组
    groups = defaultdict(list)
    for name in sorted(all_names):
        groups[dsu.find(name)].append(name)

    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, 'manifest.json')
    manifest = json.load(open(manifest_path, encoding='utf-8')) if os.path.isfile(manifest_path) else {}
    sources = manifest.setdefault('sources', {})

    next_id = existing_max_id(args.out) + 1
    exported = 0
    skipped = []
    identity_map = {}

    for root in sorted(groups.keys(), key=lambda r: -len(groups[r])):
        names = sorted(groups[root])
        all_files = []
        for name in names:
            sess_dir = sess_map.get(name2session.get(name, ''))
            if not sess_dir:
                continue
            all_files.extend(list_images(os.path.join(sess_dir, name, '乘客图片')))

        if len(all_files) < args.min_images:
            skipped.append(f"{'+'.join(names)}({len(all_files)}张)")
            continue

        pid = f'passenger_{next_id:03d}'
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

        sources[names[0]] = pid
        identity_map[pid] = names
        print(f'{pid}: {copied} 张 <- {" + ".join(names)}')
        next_id += 1
        exported += 1

    manifest['sources'] = sources
    manifest.setdefault('identities', {}).update(identity_map)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print('=' * 70)
    print(f'导出 {exported} 个身份，跳过 {len(skipped)} 个（图片不足）')
    if skipped:
        print('跳过:', skipped)
    print(f'数据集目录: {args.out}（共 {existing_max_id(args.out)} 个身份）')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

