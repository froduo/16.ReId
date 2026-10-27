#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
person_N 乘客目录「初步匹配」脚本（同一人上下车配对，供人工确认）
================================================================
- 对每个 session 目录下的 person_N，读取「乘客图片」提取 OSNet 512 维特征；
- 目录级特征 = 该 person 所有图特征均值 + L2 归一化（与程序目录全图比对同思路）；
- 按方向分组（上车 vs 下车），计算两两余弦相似度；
- 贪心一对一匹配（阈值可调），输出 JSON 供后期人工确认 / 导出训练集。

用法（yolotest 环境，torch GPU）：
  /home/tianheng/miniconda3/envs/yolotest/bin/python match_person_folders.py \
      --sessions <session目录1> <session目录2> ... \
      --model /media/tianheng/test/16.reId/passenger_reid_dataset/osnet_best.pth \
      --out match_result.json [--threshold 0.60]
"""
import argparse
import importlib.util
import json
import os
import re

import cv2
import numpy as np
import torch
import torch.nn.functional as F

OSNET_PY = "/media/tianheng/test/deep-person-reid/torchreid/models/osnet.py"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IN_W, IN_H = 128, 256
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
PERSON_RE = re.compile(r'^person_\d+_')
BATCH = 32


def load_osnet_extractor(weights):
    """用 torch 加载 osnet_x1_0（去分类头），返回 eval 模式的 feature 提取器。"""
    spec = importlib.util.spec_from_file_location("osnet_mod", OSNET_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.osnet_x1_0(num_classes=1000, pretrained=False, loss="softmax")
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    sd = {k: v for k, v in sd.items() if not k.startswith("classifier")}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def embed_batch(model, img_paths, device):
    """批量提取多张图特征，返回 Nx512（每图 L2 归一化）。"""
    feats = []
    for i in range(0, len(img_paths), BATCH):
        batch = []
        for p in img_paths[i:i + BATCH]:
            img = cv2.imread(p)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IN_W, IN_H), interpolation=cv2.INTER_LINEAR)
            img = img.astype(np.float32) / 255.0
            img = (img - MEAN) / STD
            batch.append(img.transpose(2, 0, 1))
        if not batch:
            continue
        x = torch.from_numpy(np.stack(batch)).to(device)
        with torch.no_grad():
            fm = model.featuremaps(x)
            v = F.adaptive_avg_pool2d(fm, 1).flatten(1)
            v = F.normalize(v, dim=1)
        feats.append(v.cpu().numpy())
    return np.concatenate(feats) if feats else np.zeros((0, 512), np.float32)


def read_direction(person_dir):
    """读取 person 目录下 match.txt 的「方向」字段（上车/下车/未判定）。"""
    mt = os.path.join(person_dir, 'match.txt')
    if os.path.isfile(mt):
        try:
            with open(mt, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('方向'):
                        return line.split(':', 1)[1].strip()
        except Exception:
            pass
    return '未判定'


def collect_persons(sessions):
    persons = []
    for sess in sessions:
        if not os.path.isdir(sess):
            print(f'[warn] session 不存在: {sess}')
            continue
        for name in sorted(os.listdir(sess)):
            pdir = os.path.join(sess, name)
            if not os.path.isdir(pdir) or not PERSON_RE.match(name):
                continue
            crop_dir = os.path.join(pdir, '乘客图片')
            if not os.path.isdir(crop_dir):
                continue
            files = sorted(os.path.join(crop_dir, f) for f in os.listdir(crop_dir)
                           if f.lower().endswith(IMAGE_EXTS))
            if not files:
                continue
            persons.append({
                'name': name,
                'session': os.path.basename(sess.rstrip('/')),
                'dir': os.path.abspath(pdir),
                'files': files,
                'direction': read_direction(pdir),
            })
    return persons


def main():
    ap = argparse.ArgumentParser(description='person_N 目录级 ReID 初步匹配（同一人上下车配对）')
    ap.add_argument('--sessions', nargs='+', required=True, help='一个或多个 session 目录')
    ap.add_argument('--model',
                    default='/media/tianheng/test/16.reId/passenger_reid_dataset/osnet_best.pth',
                    help='osnet x1_0 权重路径（.pth，默认新训练的 90 身份微调模型）')
    ap.add_argument('--out', default='match_result.json', help='匹配结果 JSON 输出路径')
    ap.add_argument('--threshold', type=float, default=0.60, help='同一人判定阈值 [0,1]')
    args = ap.parse_args()

    persons = collect_persons(args.sessions)
    if not persons:
        print('未找到任何 person_N 目录')
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'共发现 {len(persons)} 个 person，设备={device}，加载模型 ...', flush=True)
    model = load_osnet_extractor(args.model).to(device)

    emb = np.zeros((len(persons), 512), np.float32)
    for i, p in enumerate(persons):
        f = embed_batch(model, p['files'], device)
        if len(f) > 0:
            v = f.mean(axis=0)
            n = np.linalg.norm(v)
            if n > 1e-12:
                v = v / n
            emb[i] = v
        print(f"[{i+1}/{len(persons)}] {p['name']}: {len(f)} 张, 方向={p['direction']}", flush=True)

    sim = np.clip(emb @ emb.T, -1.0, 1.0)

    board = [i for i, p in enumerate(persons) if p['direction'] == '上车']
    alight = [i for i, p in enumerate(persons) if p['direction'] == '下车']
    unknown = [i for i, p in enumerate(persons) if p['direction'] == '未判定']

    # 候选：上车↔下车；未判定也参与（与上车、下车都比）
    candidates = []
    for a in board:
        for b in alight:
            candidates.append((sim[a, b], a, b))
    for u in unknown:
        for a in board:
            candidates.append((sim[u, a], u, a))
        for b in alight:
            candidates.append((sim[u, b], u, b))

    candidates.sort(key=lambda x: -x[0])
    used = set()
    matches = []
    for s, a, b in candidates:
        if s < args.threshold:
            continue
        if a in used or b in used:
            continue
        used.add(a)
        used.add(b)
        matches.append({
            'person_a': persons[a]['name'],
            'a_dir': persons[a]['direction'],
            'a_session': persons[a]['session'],
            'person_b': persons[b]['name'],
            'b_dir': persons[b]['direction'],
            'b_session': persons[b]['session'],
            'similarity': round(float(s), 4),
        })

    unmatched = [persons[i]['name'] for i in range(len(persons)) if i not in used]

    result = {
        'model': args.model,
        'threshold': args.threshold,
        'total_persons': len(persons),
        'matched_pairs': len(matches),
        'matches': matches,
        'unmatched': unmatched,
    }
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print('=' * 70)
    print(f'匹配结果：{len(matches)} 对（阈值 {args.threshold}），未匹配 {len(unmatched)} 人')
    for m in matches:
        print(f"  {m['person_a']}({m['a_dir']}) <-> {m['person_b']}({m['b_dir']}) sim={m['similarity']:.4f}")
    print(f'结果已保存: {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

