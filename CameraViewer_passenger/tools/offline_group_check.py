#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线检查分组：严格复现 C++ 程序的匹配逻辑
  - DeepReID::embed      ：128x256, BGR->RGB, ImageNet 归一化, 全局平均池化, L2 归一化
  - FolderMatchWorker    ：目录级 best-avg（probe 每张取与 template 目录的最大余弦，再平均）

用法（yolotest 环境）：
  /home/tianheng/miniconda3/envs/yolotest/bin/python offline_group_check.py \
      --model <osnet.onnx> --dirs <候选目录1> <候选目录2> ...
"""
import argparse
import os

import cv2
import numpy as np
import onnxruntime as ort

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def embed(sess, in_name, img_bgr):
    """复现 DeepReID::embed：整图 128x256 -> 512 维 L2 归一化特征。"""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 256), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = img.transpose(2, 0, 1)[None]
    out = sess.run(None, {in_name: np.ascontiguousarray(img)})[0]
    if out.ndim == 4:
        f = out.mean(axis=(2, 3))[0]   # 全局平均池化
    else:
        f = out.reshape(-1)
    f = f.astype(np.float32)
    n = np.linalg.norm(f)
    return f / n if n > 1e-12 else f


def embed_dir(sess, in_name, d):
    feats = []
    for fn in sorted(os.listdir(d)):
        if not fn.lower().endswith(IMAGE_EXTS):
            continue
        img = cv2.imread(os.path.join(d, fn))
        if img is None:
            continue
        feats.append(embed(sess, in_name, img))
    return feats


def best_avg(probe, templ):
    """复现 FolderMatchWorker::bestAvgSimilarity。"""
    if not probe or not templ:
        return 0.0
    s = 0.0
    for p in probe:
        s += max(float(np.dot(p, t)) for t in templ)
    return s / len(probe)


def intra_consistency(feats):
    """组内一致性：所有两两相似度（排除自身）的均值，作为“同一人”的参考下限。"""
    if len(feats) < 2:
        return 1.0
    s, c = 0.0, 0
    for i in range(len(feats)):
        for j in range(len(feats)):
            if i == j:
                continue
            s += float(np.dot(feats[i], feats[j]))
            c += 1
    return s / c if c else 1.0


def main():
    ap = argparse.ArgumentParser(description='复现 C++ FolderMatch 的目录级 best-avg 比对')
    ap.add_argument('--model', required=True, help='osnet onnx 路径')
    ap.add_argument('--dirs', nargs='+', required=True, help='候选目录列表')
    ap.add_argument('--threshold', type=float, default=0.60, help='匹配阈值（对应 DeepMatchThreshold）')
    args = ap.parse_args()

    sess = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    in_name = sess.get_inputs()[0].name

    feats = {d: embed_dir(sess, in_name, d) for d in args.dirs}
    names = [os.path.basename(d.rstrip('/')) for d in args.dirs]

    print('各候选目录特征数:')
    for i, d in enumerate(args.dirs):
        print(f'  {names[i]}: {len(feats[d])} 张 | 组内一致性={intra_consistency(feats[d]):.3f}')

    n = len(args.dirs)
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            mat[i, j] = best_avg(feats[args.dirs[i]], feats[args.dirs[j]])

    print('\nbest-avg 相似度矩阵（行=probe，列=template）:')
    hdr = '          ' + '  '.join(f'{names[j][:14]:>14}' for j in range(n))
    print(hdr)
    for i in range(n):
        row = f'{names[i][:10]:>10}'
        for j in range(n):
            row += f'{mat[i, j]:>14.3f}'
        print(row)

    print('\n对称化相似度（(a->b + b->a)/2，便于判断两两是否同一人）:')
    print('          ' + '  '.join(f'{names[j][:14]:>14}' for j in range(n)))
    for i in range(n):
        row = f'{names[i][:10]:>10}'
        for j in range(n):
            row += f'{(mat[i, j] + mat[j, i]) / 2:>14.3f}'
        print(row)

    print(f'\n说明：组内一致性 vs 组间 best-avg 的差距越大，越能区分不同人。')
    print(f'阈值参考：C++ 配置 DeepMatchThreshold={args.threshold}（best-avg 高于此值判为同一人）。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
