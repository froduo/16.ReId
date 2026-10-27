#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对重新检测得到的人体裁剪图，用 OSNet 特征做聚类分组，
把不同乘客分到不同文件夹（cluster_XX/），供人工确认与后续训练使用。

用法（用 yolotest 环境）：
  /home/tianheng/miniconda3/envs/yolotest/bin/python cluster_person_crops.py \
      --model <osnet_x1_0_head_finetuned.onnx> \
      --crops <重新检测_乘客图片目录> \
      --out   <分组输出目录> \
      --threshold 0.60 [--k 3]
"""
import argparse
import os
import shutil

import cv2
import numpy as np
import onnxruntime as ort
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IN_W, IN_H = 128, 256
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def preprocess(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IN_W, IN_H), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = img.transpose(2, 0, 1)[None]
    return np.ascontiguousarray(img)


def embed(sess, in_name, img_bgr):
    blob = preprocess(img_bgr)
    out = sess.run(None, {in_name: blob})[0]
    if out.ndim == 4:
        feat = out.mean(axis=(2, 3))[0]
    elif out.ndim == 2:
        feat = out[0]
    else:
        feat = out.reshape(-1)
    feat = feat.astype(np.float32)
    n = np.linalg.norm(feat)
    if n > 1e-12:
        feat = feat / n
    return feat


def main():
    ap = argparse.ArgumentParser(description='OSNet 特征聚类分组')
    ap.add_argument('--model', required=True, help='osnet onnx 路径')
    ap.add_argument('--crops', required=True, help='裁剪图目录')
    ap.add_argument('--out', required=True, help='分组输出目录')
    ap.add_argument('--k', type=int, default=0, help='强制分组数（0=按相似度阈值自动）')
    ap.add_argument('--threshold', type=float, default=0.60, help='同一人相似度阈值（自动分组用）')
    args = ap.parse_args()

    # 只取顶层 jpg（排除 _标注整帧 等子目录）
    files = sorted(f for f in os.listdir(args.crops)
                   if f.lower().endswith(IMAGE_EXTS)
                   and os.path.isfile(os.path.join(args.crops, f)))
    if not files:
        print('错误：没有找到裁剪图')
        return 1
    print(f'裁剪图 {len(files)} 张')

    sess = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    in_name = sess.get_inputs()[0].name

    feats, paths = [], []
    for fn in files:
        p = os.path.join(args.crops, fn)
        img = cv2.imread(p)
        if img is None:
            continue
        feats.append(embed(sess, in_name, img))
        paths.append(p)

    F = np.stack(feats)                      # Nx512
    sim = np.clip(F @ F.T, -1.0, 1.0)        # 余弦相似度
    dist = 1.0 - sim

    Z = linkage(squareform(dist, checks=False), method='average')
    if args.k > 0:
        labels = fcluster(Z, t=args.k, criterion='maxclust')
    else:
        labels = fcluster(Z, t=1.0 - args.threshold, criterion='distance')

    groups = {}
    for i, lab in enumerate(labels):
        groups.setdefault(int(lab), []).append(i)
    order = sorted(groups.keys(), key=lambda g: -len(groups[g]))

    print('=' * 70)
    print(f'聚类结果：{len(order)} 组（阈值={args.threshold}）')
    for new_idx, g in enumerate(order, 1):
        idxs = groups[g]
        bases = sorted({os.path.basename(paths[i]).split('_det')[0] for i in idxs})
        print(f'  组{new_idx}: {len(idxs)} 张 | 帧范围 {bases[0]} ~ {bases[-1]}')
        for i in idxs:
            print(f'      {os.path.basename(paths[i])}')

    # 创建文件夹并复制
    os.makedirs(args.out, exist_ok=True)
    for new_idx, g in enumerate(order, 1):
        d = os.path.join(args.out, f'cluster_{new_idx:02d}')
        os.makedirs(d, exist_ok=True)
        for i in groups[g]:
            shutil.copy2(paths[i], os.path.join(d, os.path.basename(paths[i])))

    print('=' * 70)
    print(f'已创建 {len(order)} 个文件夹: {args.out}')
    for new_idx, g in enumerate(order, 1):
        print(f'  cluster_{new_idx:02d}/ : {len(groups[g])} 张')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
