#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 manual_pairs（人工配对结果）拆成「匹配正确」与「匹配错误」两组
=============================================================
- 读 manual_pairs JSON（[{board, alight}, ...]）
- 读错误配对清单（文本：每行 "board_num alight_num"，如 "208 236"）
- 按「无序配对」匹配：manual 对 {board_id, alight_id} == 错误对 {a, b} 即判为错误
- 输出：
    --valid-out   匹配正确（喂给 append_pairs_dataset.py 增量追加继续训练）
    --invalid-out 匹配错误（单独保存，便于人工重新核对）

用法（yolotest 环境）：
  /home/tianheng/miniconda3/envs/yolotest/bin/python split_manual_pairs.py \
      --manual /media/tianheng/test/16.reId/manual_pairs_20261023.json \
      --wrong  /media/tianheng/test/16.reId/CameraViewer_passenger/tools/wrong_pairs_20261023.txt \
      --valid-out   /media/tianheng/test/16.reId/manual_pairs_20261023_valid.json \
      --invalid-out /media/tianheng/test/16.reId/manual_pairs_20261023_invalid.json
"""
import argparse
import json
import re


PERSON_ID_RE = re.compile(r'person_(\d+)')


def person_id(name):
    m = PERSON_ID_RE.search(name or '')
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser(description='拆分人工配对：正确 / 错误')
    ap.add_argument('--manual', required=True, help='manual_pairs.json 路径')
    ap.add_argument('--wrong', required=True, help='错误配对清单（每行 "board alight"）')
    ap.add_argument('--valid-out', required=True, help='匹配正确输出的 JSON 路径')
    ap.add_argument('--invalid-out', required=True, help='匹配错误输出的 JSON 路径')
    args = ap.parse_args()

    manual = json.load(open(args.manual, encoding='utf-8'))
    if isinstance(manual, dict):
        manual = manual.get('pairs', [])
    if not isinstance(manual, list):
        print('[error] manual_pairs 应为数组'); return 1

    wrong = []
    for line in open(args.wrong, encoding='utf-8'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.replace('；', ' ').replace(';', ' ').replace('，', ' ').replace(',', ' ').split()
        if len(parts) >= 2:
            try:
                wrong.append(frozenset((int(parts[0]), int(parts[1]))))
            except ValueError:
                print(f'[warn] 忽略无法解析的错误配对行: {line}')

    valid, invalid = [], []
    for p in manual:
        bid, aid = person_id(p.get('board')), person_id(p.get('alight'))
        pair = frozenset(x for x in (bid, aid) if x is not None)
        if pair in wrong:
            invalid.append(p)
        else:
            valid.append(p)

    with open(args.valid_out, 'w', encoding='utf-8') as f:
        json.dump(valid, f, ensure_ascii=False, indent=2)
    with open(args.invalid_out, 'w', encoding='utf-8') as f:
        json.dump(invalid, f, ensure_ascii=False, indent=2)

    print(f'总配对: {len(manual)}')
    print(f'匹配正确(→训练集): {len(valid)}  → {args.valid_out}')
    print(f'匹配错误(需重核): {len(invalid)}  → {args.invalid_out}')
    if invalid:
        print('匹配错误清单:')
        for p in invalid:
            print(f'  {p.get("board")} ↔ {p.get("alight")}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
