#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对 person_N/原始图片 里的整帧重新做「人体检测 + 裁剪」（onnxruntime 直接推理，不依赖 ultralytics），
生成「重新检测_乘客图片/」目录，用于人工确认该 person 目录是否混入了多个人。

模型：passenger_yolov8m（输入 640x640，输出 1x5x8400，[cx,cy,w,h,conf]，nc=1 人体）。

用法（用 yolotest 环境）：
  /home/tianheng/miniconda3/envs/yolotest/bin/python detect_person_crops.py \
      --model <passenger_yolov8m.onnx> \
      --raw   <person_N/原始图片> \
      --out   <输出目录> \
      --conf 0.30 [--annotated]
"""
import argparse
import os

import cv2
import numpy as np
import onnxruntime as ort

IN_SIZE = 640
PAD_COLOR = 114


def preprocess(img):
    """等比缩放 + 灰色填充到 640x640，返回 blob(1,3,640,640) 与坐标映射参数。"""
    h, w = img.shape[:2]
    scale = min(IN_SIZE / h, IN_SIZE / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    pad_left = (IN_SIZE - new_w) // 2
    pad_top = (IN_SIZE - new_h) // 2
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((IN_SIZE, IN_SIZE, 3), PAD_COLOR, dtype=np.uint8)
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    blob = canvas[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB
    blob = blob.transpose(2, 0, 1)[None]                  # 1,3,640,640
    return np.ascontiguousarray(blob), scale, pad_left, pad_top


def nms(boxes, scores, iou_thr):
    """简单 NMS，boxes: Nx4 xyxy，scores: N，返回保留索引。"""
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area2 = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / (area1 + area2 - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def main():
    ap = argparse.ArgumentParser(description='重新检测 person 目录原始图片中的所有人并裁剪')
    ap.add_argument('--model', required=True, help='passenger_yolov8m ONNX 路径')
    ap.add_argument('--raw', required=True, help='原始图片目录')
    ap.add_argument('--out', required=True, help='输出裁剪目录')
    ap.add_argument('--conf', type=float, default=0.30, help='检测置信度阈值')
    ap.add_argument('--iou', type=float, default=0.50, help='NMS IoU 阈值')
    ap.add_argument('--annotated', action='store_true', help='同时输出带框标注的整帧')
    args = ap.parse_args()

    if not os.path.isdir(args.raw):
        print(f'错误：原始图片目录不存在: {args.raw}')
        return 1
    if not os.path.isfile(args.model):
        print(f'错误：模型不存在: {args.model}')
        return 1

    os.makedirs(args.out, exist_ok=True)
    ann_dir = os.path.join(args.out, '_标注整帧') if args.annotated else None
    if ann_dir:
        os.makedirs(ann_dir, exist_ok=True)

    sess = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    in_name = sess.get_inputs()[0].name

    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    files = sorted(f for f in os.listdir(args.raw) if f.lower().endswith(exts))
    print(f'原始图片 {len(files)} 张，conf={args.conf}')

    total = 0
    per_frame = []
    for i, fn in enumerate(files):
        img = cv2.imread(os.path.join(args.raw, fn))
        if img is None:
            continue
        blob, scale, pl, pt = preprocess(img)
        out = sess.run(None, {in_name: blob})[0]   # 1,5,8400
        pred = out[0].T                             # 8400,5

        idx = np.where(pred[:, 4] >= args.conf)[0]
        n = 0
        if idx.size > 0:
            cx, cy, w, h = pred[idx, 0], pred[idx, 1], pred[idx, 2], pred[idx, 3]
            x1 = (cx - w / 2 - pl) / scale
            y1 = (cy - h / 2 - pt) / scale
            x2 = (cx + w / 2 - pl) / scale
            y2 = (cy + h / 2 - pt) / scale
            xyxy = np.stack([x1, y1, x2, y2], axis=1)
            keep = nms(xyxy, pred[idx, 4], args.iou)
            xyxy = xyxy[keep]
            n = len(xyxy)

            for j, b in enumerate(xyxy):
                bx1, by1, bx2, by2 = [int(round(v)) for v in b]
                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(img.shape[1], bx2), min(img.shape[0], by2)
                if bx2 <= bx1 or by2 <= by1:
                    continue
                crop = img[by1:by2, bx1:bx2]
                base = os.path.splitext(fn)[0]
                cv2.imwrite(os.path.join(args.out, f'{base}_det{j:02d}.jpg'), crop)

            if ann_dir:
                ann = img.copy()
                for b in xyxy:
                    bx1, by1, bx2, by2 = [int(round(v)) for v in b]
                    cv2.rectangle(ann, (bx1, by1), (bx2, by2), (0, 0, 255), 3)
                cv2.imwrite(os.path.join(ann_dir, fn), ann)

        total += n
        per_frame.append((fn, n))
        flag = '' if n == 1 else f'  <== {n} 人'
        print(f'[{i+1}/{len(files)}] {fn}: {n} 人{flag}')

    print('=' * 70)
    print(f'总计 {total} 个人体裁剪，输出目录: {args.out}')
    zero = sum(1 for _, n in per_frame if n == 0)
    single = sum(1 for _, n in per_frame if n == 1)
    multi = [(fn, n) for fn, n in per_frame if n > 1]
    print(f'帧统计：0人={zero} 帧，1人={single} 帧，多人={len(multi)} 帧')
    for fn, n in multi:
        print(f'  {fn}: {n} 人')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
