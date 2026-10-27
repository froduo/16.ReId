#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
person 归档后处理分析（离线，dry-run，不修改任何文件）
========================================================
用最新 OSNet 模型对 save_images 下所有 person_N 做三件事：
  1) 静态误检检测：相邻帧像素差 < 阈值 且 组内一致性极高 -> 标记为「非乘客/静态物」
  2) 碎片检测：图片数 < min_fragment -> 与同 session 时间相邻的 person 做 best-avg
     比对，相似度高则建议合并（解决 person_101/102、512/514 类 track 断裂分裂）
  3) 多人混存检测：类内高阈值聚类 -> 出现时间连续的异类子集 -> 建议拆分
     （解决 person_507/519 类折返窗口混入多人）

输出：JSON 报告（每类问题清单 + 建议动作），供人工确认后再执行纠正。

用法（yolotest 环境）：
  python postprocess_persons.py --root <save_images> --model <osnet.onnx> \
      --out report.json [--limit N] [--min-fragment 20] [--split-thr 0.75]
"""
import argparse
import json
import os
import re

import cv2
import numpy as np
import torch
import importlib.util

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
PERSON_RE = re.compile(r'^person_(\d+)_(\d{8}_\d{6})$')

_MODEL = None
_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
_OSNET_PY = "/media/tianheng/test/deep-person-reid/torchreid/models/osnet.py"


def load_model(pth_path):
    global _MODEL
    spec = importlib.util.spec_from_file_location("osnet_mod", _OSNET_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.osnet_x1_0(num_classes=1000, pretrained=False, loss="softmax")
    ck = torch.load(pth_path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    sd = {k: v for k, v in sd.items() if not k.startswith("classifier")}
    model.load_state_dict(sd, strict=False)
    model.eval().to(_DEVICE)
    _MODEL = model
    return model


def embed(img_bgr):
    """128x256 -> 512 维 L2 归一化特征（GPU/CPU）。"""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 256), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    x = torch.from_numpy(img.transpose(2, 0, 1)[None]).to(_DEVICE)
    with torch.no_grad():
        fm = _MODEL.featuremaps(x)
        v = torch.nn.functional.normalize(
            torch.nn.functional.adaptive_avg_pool2d(fm, 1).flatten(1), dim=1)
    return v[0].cpu().numpy()


def list_jpg(d):
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.lower().endswith(IMAGE_EXTS))


def load_features(d, max_n=15):
    files = list_jpg(d)
    if len(files) > max_n:
        step = len(files) / max_n
        files = [files[int(i * step)] for i in range(max_n)]
    feats = []
    for fn in files:
        img = cv2.imread(os.path.join(d, fn))
        if img is None:
            continue
        feats.append(embed(img))
    return np.array(feats), files


def intra_consistency(feats):
    if len(feats) < 2:
        return 1.0
    sim = feats @ feats.T
    n = len(feats)
    return float((sim.sum() - np.trace(sim)) / (n * (n - 1)))


def best_avg(probe, templ):
    if len(probe) == 0 or len(templ) == 0:
        return 0.0
    s = 0.0
    for p in probe:
        s += max(float(np.dot(p, t)) for t in templ)
    return s / len(probe)


def sym_sim(a, b):
    return (best_avg(a, b) + best_avg(b, a)) / 2


def cluster(feats, files, thr):
    n = len(feats)
    if n == 0:
        return []
    sim = feats @ feats.T
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] > thr:
                union(i, j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = []
    for g in groups.values():
        idxs = sorted(g)
        out.append({'count': len(g), 'files': [files[i] for i in idxs]})
    out.sort(key=lambda x: -x['count'])
    return out


def static_pixdiff(d, max_n=20):
    files = list_jpg(d)[:max_n]
    imgs = []
    for f in files:
        img = cv2.imread(os.path.join(d, f))
        if img is not None:
            imgs.append(cv2.resize(img, (128, 256)))
    if len(imgs) < 2:
        return 0.0
    diffs = [np.abs(imgs[i].astype(float) - imgs[i + 1].astype(float)).mean()
             for i in range(len(imgs) - 1)]
    return float(np.mean(diffs))


def main():
    ap = argparse.ArgumentParser(description='person 归档后处理分析（dry-run）')
    ap.add_argument('--root', required=True, help='save_images 根目录')
    ap.add_argument('--model', default='/media/tianheng/test/16.reId/passenger_reid_dataset_v3/osnet_best.pth',
                    help='OSNet 模型路径（pth，默认用 v3 最新训练）')
    ap.add_argument('--out', default='postprocess_report.json', help='报告输出路径')
    ap.add_argument('--limit', type=int, default=0, help='最多处理 person 数（0=全部）')
    ap.add_argument('--min-fragment', type=int, default=20, help='碎片判定图片数阈值')
    ap.add_argument('--split-thr', type=float, default=0.75, help='多人拆分聚类阈值')
    ap.add_argument('--merge-thr', type=float, default=0.60, help='碎片合并相似度阈值')
    ap.add_argument('--static-pixdiff', type=float, default=6.0, help='静态误检像素差阈值')
    args = ap.parse_args()

    load_model(args.model)
    print(f'模型加载完成，设备={_DEVICE}', flush=True)

    sessions = []
    for sname in sorted(os.listdir(args.root)):
        sdir = os.path.join(args.root, sname)
        if not os.path.isdir(sdir):
            continue
        persons = []
        for pname in sorted(os.listdir(sdir)):
            m = PERSON_RE.match(pname)
            if not m:
                continue
            persons.append({'name': pname, 'n': int(m.group(1)),
                            'ts': m.group(2), 'dir': os.path.join(sdir, pname)})
        if persons:
            sessions.append({'name': sname, 'persons': persons})

    total = sum(len(s['persons']) for s in sessions)
    print(f'扫描到 {len(sessions)} 个 session，共 {total} 个 person')

    feat_cache = {}
    pixdiff_cache = {}
    ic_cache = {}

    def get_feats(p):
        key = p['dir']
        if key not in feat_cache:
            feat_cache[key] = load_features(os.path.join(key, '乘客图片'), 15)
        return feat_cache[key]

    def get_pixdiff(p):
        key = p['dir']
        if key not in pixdiff_cache:
            pixdiff_cache[key] = static_pixdiff(os.path.join(key, '乘客图片'), 5)
        return pixdiff_cache[key]

    def get_ic(p):
        key = p['dir']
        if key not in ic_cache:
            feats, _ = get_feats(p)
            ic_cache[key] = intra_consistency(feats)
        return ic_cache[key]

    static_list = []
    fragment_merge = []
    multi_list = []
    processed = 0

    for s in sessions:
        persons = s['persons']
        for i, p in enumerate(persons):
            if args.limit and processed >= args.limit:
                break
            processed += 1
            if processed % 100 == 0:
                print(f'  进度: {processed}/{total} person', flush=True)
            crop = os.path.join(p['dir'], '乘客图片')
            n_img = len(list_jpg(crop))
            if n_img == 0:
                continue

            pix = get_pixdiff(p)

            # ---- 1. 静态误检：像素差 + 图数判断（无需 OSNet，快）----
            if pix < args.static_pixdiff and n_img >= 30:
                static_list.append({
                    'session': s['name'], 'person': p['name'], 'n_img': n_img,
                    'pixdiff': round(pix, 2),
                    'action': '排除（非乘客/静态物）',
                })
                continue

            # ---- 2. 碎片合并：图少的 person 用 OSNet 与相邻比对 ----
            if n_img < args.min_fragment:
                neighbors = []
                for j in (i - 1, i + 1):
                    if 0 <= j < len(persons):
                        nn = len(list_jpg(os.path.join(persons[j]['dir'], '乘客图片')))
                        if nn >= args.min_fragment:
                            neighbors.append(persons[j])
                best = None
                for nb in neighbors:
                    fa, _ = get_feats(p)
                    fb, _ = get_feats(nb)
                    sim = sym_sim(fa, fb)
                    if best is None or sim > best[1]:
                        best = (nb, sim)
                if best and best[1] > args.merge_thr:
                    fragment_merge.append({
                        'session': s['name'], 'person': p['name'], 'n_img': n_img,
                        'merge_to': best[0]['name'], 'similarity': round(best[1], 3),
                        'action': '合并到 ' + best[0]['name'],
                    })

            # ---- 3. 多人拆分：图特别多的 person 用 OSNet 聚类（>=300 才可疑）----
            if n_img >= 300:
                feats, files = load_features(crop, 40)
                groups = cluster(feats, files, args.split_thr)
                if len(groups) > 1:
                    alien = [g for g in groups[1:] if g['count'] >= 3]
                    if alien:
                        multi_list.append({
                            'session': s['name'], 'person': p['name'], 'n_img': n_img,
                            'groups': [{'count': g['count'],
                                        'range': [g['files'][0], g['files'][-1]]}
                                       for g in groups],
                            'action': '拆分（混入多人）',
                        })

    report = {
        'summary': {
            'total_persons': total,
            'static_false_positive': len(static_list),
            'fragment_merge': len(fragment_merge),
            'multi_person_split': len(multi_list),
        },
        'static_false_positive': static_list,
        'fragment_merge': fragment_merge,
        'multi_person_split': multi_list,
    }

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print('\n===== 汇总 =====')
    print(f"  静态误检（非乘客）: {len(static_list)}")
    print(f"  碎片待合并: {len(fragment_merge)}")
    print(f"  多人待拆分: {len(multi_list)}")
    print(f"\n报告已写入: {args.out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
