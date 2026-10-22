#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ReID 图像验证脚本：复现 CameraViewer DeepReID 的特征提取流程
（OSNet + BGR→RGB + 128x256 + ImageNet 归一化 + 全局平均池化 + L2 归一化），
对「下车目录」与「上车目录」两组图像计算目录级 best-avg 相似度，
用于根据已保存图像验证 ReID 算法识别同一人的准确性。

用法：
  python reid_verify.py --model osnet_x1_0_head_finetuned.onnx \
      --probe   save_images/person_2_xxx/乘客图片 \
      --template save_images/person_1_xxx/乘客图片 \
      --threshold 0.60 [--max-images 200]
"""
import argparse
import os

import cv2
import numpy as np
import onnxruntime as ort

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IN_W, IN_H = 128, 256
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def preprocess(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IN_W, IN_H), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    # HWC -> CHW -> NCHW
    img = img.transpose(2, 0, 1)[None, ...]
    return np.ascontiguousarray(img)


def embed(session, input_name, img_bgr):
    """返回 L2 归一化的 512 维特征向量。"""
    blob = preprocess(img_bgr)
    out = session.run(None, {input_name: blob})[0]
    if out.ndim == 4:
        feat = out.mean(axis=(2, 3))[0]     # NCHW -> 全局平均池化
    elif out.ndim == 2:
        feat = out[0]
    else:
        feat = out.reshape(-1)
    feat = feat.astype(np.float32)
    n = np.linalg.norm(feat)
    if n > 1e-12:
        feat = feat / n
    return feat


def image_files(path, max_images=0):
    if os.path.isfile(path):
        return [path]
    files = sorted(
        os.path.join(path, f) for f in os.listdir(path)
        if f.lower().endswith(IMAGE_EXTS)
    )
    return files if max_images <= 0 else files[:max_images]


def best_avg_sim(session, input_name, probe_files, templ_files):
    templ_feats = []
    for f in templ_files:
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        if img is None:
            continue
        templ_feats.append(embed(session, input_name, img))
    if not templ_feats:
        return 0.0, 0

    sims = []
    for f in probe_files:
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        if img is None:
            continue
        p = embed(session, input_name, img)
        best = max(float(np.dot(p, t)) for t in templ_feats)
        sims.append(best)
    if not sims:
        return 0.0, 0
    return float(np.mean(sims)), len(sims)


def main():
    ap = argparse.ArgumentParser(description='ReID 上下车图像相似度验证')
    ap.add_argument('--model', required=True, help='OSNet ONNX 模型路径')
    ap.add_argument('--probe', required=True, help='下车目录（或单张图）')
    ap.add_argument('--template', required=True, help='上车目录（或单张图）')
    ap.add_argument('--threshold', type=float, default=0.60)
    ap.add_argument('--max-images', type=int, default=200)
    args = ap.parse_args()

    sess = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name

    probe_files = image_files(args.probe, args.max_images)
    templ_files = image_files(args.template, args.max_images)
    if not probe_files or not templ_files:
        print('错误：未找到可用图像（probe 或 template 为空）')
        return 1

    sim, counted = best_avg_sim(sess, input_name, probe_files, templ_files)
    verdict = '匹配（同一人）' if sim >= args.threshold else '不匹配'
    print('=' * 64)
    print(f'模型     : {args.model}')
    print(f'下车图   : {args.probe}（{len(probe_files)} 张）')
    print(f'上车图   : {args.template}（{len(templ_files)} 张）')
    print(f'相似度   : {sim:.4f}（best-avg，有效 probe {counted} 张）')
    print(f'阈值     : {args.threshold:.3f}')
    print(f'结论     : {verdict}')
    print('=' * 64)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
