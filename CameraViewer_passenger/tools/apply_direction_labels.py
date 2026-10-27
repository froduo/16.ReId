#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把「方向人工标注」结果回写到 person 目录的 match.txt
====================================================
读取 labels.json（render_unknown_labeler.py 导出的标注结果，格式：
  {"person_XXX_时间": "上车", ...}），
把对应 person 目录 match.txt 的「方向」字段改写为人工标注值，
并刷新 session 目录下汇总 match.txt 的「方向=xxx」。

用法：
  /home/tianheng/miniconda3/envs/yolotest/bin/python apply_direction_labels.py \
      --session <session目录> --labels <labels.json>
"""
import argparse
import json
import os
import re

PERSON_RE = re.compile(r'^person_\d+_')

DIR_MAP = {
    '上车': '上车', '下车': '下车',
    '上车折返': '上车折返', '下车折返': '下车折返',
    '其他': '其他', '未判定': '未判定',
}


def rewrite_marker(person_dir, direction):
    mt = os.path.join(person_dir, 'match.txt')
    if not os.path.isfile(mt):
        print(f'  [warn] 无 match.txt: {person_dir}')
        return False
    with open(mt, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    changed = False
    for i, line in enumerate(lines):
        if line.startswith('方向'):
            lines[i] = f'方向: {direction}\n'
            changed = True
            break
    if not changed:
        lines.insert(0, f'方向: {direction}\n')
    with open(mt, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    return True


def main():
    ap = argparse.ArgumentParser(description='回写人工标注方向到 match.txt')
    ap.add_argument('--session', required=True, help='session 目录')
    ap.add_argument('--labels', required=True, help='labels.json 标注结果')
    args = ap.parse_args()

    labels = json.load(open(args.labels, encoding='utf-8'))
    print(f'读取 {len(labels)} 条标注')

    applied = 0
    for name, direction in labels.items():
        if direction not in DIR_MAP:
            print(f'  [skip] {name}: 未知方向 {direction}')
            continue
        pdir = os.path.join(args.session, name)
        if not os.path.isdir(pdir):
            print(f'  [warn] {name}: 目录不存在')
            continue
        if rewrite_marker(pdir, direction):
            applied += 1
            print(f'  {name} → {direction}')

    print(f'已回写 {applied}/{len(labels)} 个 person 的方向')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
