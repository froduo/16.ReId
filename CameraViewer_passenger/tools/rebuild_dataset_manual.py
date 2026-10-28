#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于人工配对结果（manual_pairs.json）重建 ReID 数据集
====================================================
- 读 manual_pairs.json（人工确认的 board↔alight 配对）
- 并查集把每对合并为同一身份
- 其余未配对的 person 各自独立
- 清空旧导出（passenger_004 及之后），重新增量导出到 passenger_reid_dataset

用法（yolotest 环境）：
  /home/tianheng/miniconda3/envs/yolotest/bin/python rebuild_dataset_manual.py \
      --manual /media/tianheng/test/16.reId/CameraViewer_passenger/manual_pairs.json \
      --sessions <session目录1> <session目录2> ... \
      --out /media/tianheng/test/16.reId/CameraViewer_passenger/passenger_reid_dataset \
      --min-images 10 --keep-max-id 3
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


def main():
    ap = argparse.ArgumentParser(description='按人工配对重建 ReID 数据集')
    ap.add_argument('--manual', required=True, help='manual_pairs.json 路径')
    ap.add_argument('--sessions', nargs='+', required=True, help='一个或多个 session 目录')
    ap.add_argument('--out', default='/media/tianheng/test/16.reId/CameraViewer_passenger/passenger_reid_dataset')
    ap.add_argument('--min-images', type=int, default=10, help='每个身份最少图片数')
    ap.add_argument('--keep-max-id', type=int, default=3, help='保留的既有身份编号上限（如 3=保留 passenger_001~003）')
    args = ap.parse_args()

    manual = json.load(open(args.manual, encoding='utf-8'))

    # 收集所有 person
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

    # 并查集合并人工配对
    dsu = DSU()
    for p in manual:
        dsu.union(p['board'], p['alight'])
    for name in persons:
        dsu.find(name)

    # 按根分组
    groups = defaultdict(list)
    for name in sorted(persons):
        groups[dsu.find(name)].append(name)

    # 清空旧导出（passenger_004 及之后）
    removed = 0
    for name in os.listdir(args.out):
        m = re.match(r'^passenger_(\d+)$', name)
        if m and int(m.group(1)) > args.keep_max_id and os.path.isdir(os.path.join(args.out, name)):
            shutil.rmtree(os.path.join(args.out, name))
            removed += 1
    print(f'已清空 {removed} 个旧身份目录（保留 passenger_001~{args.keep_max_id:03d}）')

    # 重新导出
    next_id = args.keep_max_id + 1
    exported = 0
    skipped = []
    new_identities = {}

    for root in sorted(groups.keys(), key=lambda r: -len(groups[r])):
        names = sorted(groups[root])
        all_files = []
        for name in names:
            all_files.extend(persons[name]['files'])

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

        new_identities[pid] = names
        print(f'{pid}: {copied} 张 <- {" + ".join(names)}')
        next_id += 1
        exported += 1

    # 更新 manifest：清理指向已删身份的记录，写入新 identities
    manifest_path = os.path.join(args.out, 'manifest.json')
    manifest = json.load(open(manifest_path, encoding='utf-8')) if os.path.isfile(manifest_path) else {}

    def pid_num(v):
        m = re.match(r'^passenger_(\d+)$', v)
        return int(m.group(1)) if m else 0

    sources = {k: v for k, v in manifest.get('sources', {}).items() if pid_num(v) <= args.keep_max_id}
    manifest['sources'] = sources
    manifest['identities'] = new_identities
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    total_ids = sum(1 for n in os.listdir(args.out) if re.match(r'^passenger_\d+$', n))
    total_imgs = sum(len(files) for files in
                     [list_images(os.path.join(args.out, d)) for d in os.listdir(args.out)
                      if re.match(r'^passenger_\d+$', d)])

    print('=' * 70)
    print(f'导出 {exported} 个身份，跳过 {len(skipped)} 个（图片不足）')
    if skipped:
        print('跳过:', skipped)
    print(f'数据集: {args.out} 共 {total_ids} 个身份 / {total_imgs} 张图')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

