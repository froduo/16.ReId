#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断：按时间段分组，计算各组平均 OSNet 特征相似度，结果写入 /tmp/diag_result.txt。"""
import os
import cv2
import numpy as np
import onnxruntime as ort

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def embed(sess, in_name, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 256), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = img.transpose(2, 0, 1)[None]
    out = sess.run(None, {in_name: np.ascontiguousarray(img)})[0]
    if out.ndim == 4:
        f = out.mean(axis=(2, 3))[0]
    else:
        f = out.reshape(-1)
    f = f.astype(np.float32)
    n = np.linalg.norm(f)
    return f / n if n > 1e-12 else f


def main():
    crops = '/media/tianheng/test/16.reId/build-CameraViewer_passenger-Desktop-Debug/save_images/ch01_20260930_063419_074220topspeed_20261022_094525/person_8_20261022_095139/重新检测_乘客图片'
    model = '/media/tianheng/test/16.reId/CameraViewer_passenger/model/osnet_x1_0_head_finetuned.onnx'
    out_path = '/tmp/diag_result.txt'
    lines = []
    sess = ort.InferenceSession(model, providers=['CPUExecutionProvider'])
    in_name = sess.get_inputs()[0].name

    files = sorted(f for f in os.listdir(crops)
                   if f.lower().endswith('.jpg') and os.path.isfile(os.path.join(crops, f)))
    feats, names = [], []
    for fn in files:
        img = cv2.imread(os.path.join(crops, fn))
        if img is None:
            continue
        feats.append(embed(sess, in_name, img))
        names.append(fn)
    lines.append(f'共 {len(feats)} 张特征')

    groups = {}
    for f, n in zip(feats, names):
        # frame_NNNNN_HHMMSS_zzz_detX.jpg
        idx = int(n.split('_')[1])
        det = int(n.split('_det')[1].split('.')[0])
        if idx <= 6:
            t = 'T1'
        elif idx <= 32:
            t = 'T2'
        else:
            t = 'T3'
        key = f'{t}_det{det}'
        groups.setdefault(key, []).append(f)

    keys = sorted(groups)
    for k in keys:
        lines.append(f'  {k}: {len(groups[k])} 张')

    lines.append('组间平均相似度矩阵:')
    lines.append('        ' + '  '.join(f'{k:>10}' for k in keys))
    for a in keys:
        row = [f'{a:>8}']
        for b in keys:
            s = np.mean([float(np.dot(x, y)) for x in groups[a] for y in groups[b]])
            row.append(f'{s:>10.3f}')
        lines.append(' '.join(row))

    text = '\n'.join(lines)
    with open(out_path, 'w') as fh:
        fh.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
