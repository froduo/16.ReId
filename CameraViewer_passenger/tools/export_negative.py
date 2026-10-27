#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
静态误检负样本导出
==================
读取 postprocess_persons.py 报告中的 static_false_positive（非乘客/静态物），
把这些 person 的「乘客图片」复制到 12.ultralytics 的 passenger_new 项目，
并生成空 label（负样本：图中无 person），供下次 YOLO 训练时加入，
让模型学习「这些不是乘客」。

用法：
  python export_negative.py --report <postprocess_report.json> \
      --img-dir <passenger_new images 目录> --label-dir <passenger_new labels 目录>
"""
import argparse
import json
import os
import shutil


def main():
    ap = argparse.ArgumentParser(description='导出静态误检负样本')
    ap.add_argument('--report', required=True, help='postprocess_persons.py 报告')
    ap.add_argument('--save-root', required=True, help='save_images 根目录')
    ap.add_argument('--img-dir', required=True, help='passenger_new images 目录')
    ap.add_argument('--label-dir', required=True, help='passenger_new labels 目录')
    ap.add_argument('--max-per-person', type=int, default=30, help='每个 person 最多导出图片数')
    args = ap.parse_args()

    report = json.load(open(args.report, encoding='utf-8'))
    negatives = report.get('static_false_positive', [])
    print(f'待导出静态误检 person: {len(negatives)} 个')

    total = 0
    for item in negatives:
        session = item['session']
        person = item['person']
        src = os.path.join(args.save_root, session, person, '乘客图片')
        if not os.path.isdir(src):
            print(f'  [跳过] 目录不存在: {src}')
            continue
        files = sorted(f for f in os.listdir(src) if f.lower().endswith(('.jpg', '.png')))
        # 均匀采样，避免单个 person 图片过多
        if len(files) > args.max_per_person:
            step = len(files) / args.max_per_person
            files = [files[int(i * step)] for i in range(args.max_per_person)]
        n = 0
        for fn in files:
            stem = f'__proj_passenger_new__falsepos__{session}__{person}__{fn}'
            dst_img = os.path.join(args.img_dir, stem)
            dst_lbl = os.path.join(args.label_dir, os.path.splitext(stem)[0] + '.txt')
            shutil.copy2(os.path.join(src, fn), dst_img)
            # 空 label = 负样本（图中无 person）
            open(dst_lbl, 'w', encoding='utf-8').close()
            n += 1
        total += n
        print(f'  {person}: 导出 {n} 张')

    print(f'\n完成：共导出 {total} 张负样本图片到 {args.img_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
