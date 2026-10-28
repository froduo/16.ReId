#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线复核脚本：对「人物片段」视频复现 CameraViewer 的上下车判定逻辑
（完整穿越 + 区域级半程接力 + 区域外接近点接力），用于验证算法效果。

用法：
  python3 verify_offline_clip.py <视频文件或目录> [--step 2] [--limit N] [--save-annot]

依赖：opencv-python（无需 CUDA；模型用 model/yolov8n.onnx）
"""
import os
import sys
import glob
import json
import argparse
import cv2
import numpy as np

MODEL = 'CameraViewer/model/yolov8n.onnx'
REGIONS = 'build-CameraViewer-main-Desktop-Debug/regions.json'
IN_SIZE = 640
CONF, NMS = 0.45, 0.5

# 与 C++ PersonDetector 一致的三层过滤（比例 + 绝对像素 + 宽高比）
MIN_H_RATIO, MIN_AREA_RATIO = 0.08, 0.005
MIN_BOX_W, MIN_BOX_H = 30, 90
MIN_AR, MAX_AR = 0.12, 1.00

# 与 ZoneMonitor 一致的门槛
MIN_OVERLAP_AREA = 0.0002      # 启动采样所需的最小交叠面积(归一化)
CONFIRM_LEAVE_FRAMES = 3       # 离开确认帧
PENDING_GRACE_S = 8.0          # 半程有效期
APPROACH_GRACE_S = 15.0        # 接近点有效期
APPROACH_NEAR_X = 0.25         # 接近点水平接近阈值
MAX_MISSED = 25                # 跟踪丢失容忍(处理帧)


def load_zone(path):
    polys = json.load(open(path))[0]['polygon']
    xs = [p['x'] for p in polys]
    ys = [p['y'] for p in polys]
    return dict(region=json.load(open(path))[0]['id'],
                leftX=min(xs), rightX=max(xs), topY=min(ys), bottomY=max(ys))


def detect(net, img):
    h, w = img.shape[:2]
    s = min(IN_SIZE / w, IN_SIZE / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    dw, dh = (IN_SIZE - nw) // 2, (IN_SIZE - nh) // 2

    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (IN_SIZE, IN_SIZE), (0, 0, 0), True, False)
    net.setInput(blob)
    out = net.forward()
    d = np.squeeze(out).T                      # (8400, 84)
    if d.shape[1] < 5:
        return []
    cls = d[:, 4:]
    cid = cls.argmax(1)
    conf = cls.max(1)
    m = (cid == 0) & (conf >= CONF)
    d, conf = d[m], conf[m]
    if len(d) == 0:
        return []

    boxes, scores = [], []
    for (cx, cy, bw, bh), c in zip(d[:, :4], conf):
        x = (cx - bw / 2 - dw) / s
        y = (cy - bh / 2 - dh) / s
        boxes.append([float(x), float(y), float(bw / s), float(bh / s)])
        scores.append(float(c))

    idx = cv2.dnn.NMSBoxes(boxes, scores, CONF, NMS)
    res = []
    for i in np.array(idx).flatten():
        x, y, bw, bh = boxes[i]
        if bw < MIN_BOX_W or bh < MIN_BOX_H:
            continue
        ar = bw / max(1.0, bh)
        if ar < MIN_AR or ar > MAX_AR:
            continue
        if bh / h < MIN_H_RATIO:
            continue
        if bw * bh / (w * h) < MIN_AREA_RATIO:
            continue
        res.append((x, y, bw, bh, scores[i]))
    return res


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / (aw * ah + bw * bh - inter)


def track_frame(tracks, dets, frame_idx):
    """极简关联：IoU 优先 → 中心距离兜底；返回 {tid: box}"""
    used = set()
    assign = {}
    # 1) IoU
    for tid, tr in tracks.items():
        best, bj = 0.0, -1
        for j, d in enumerate(dets):
            if j in used:
                continue
            v = iou(tr['box'], d[:4])
            if v > best:
                best, bj = v, j
        if bj >= 0 and best >= 0.15:
            assign[tid] = bj
            used.add(bj)
    # 2) 中心距离（放宽，减少 ID 切换）
    for tid, tr in tracks.items():
        if tid in assign:
            continue
        tcx, tcy = tr['box'][0] + tr['box'][2] / 2, tr['box'][1] + tr['box'][3] / 2
        best, bj = 1e9, -1
        for j, d in enumerate(dets):
            if j in used:
                continue
            cx, cy = d[0] + d[2] / 2, d[1] + d[3] / 2
            dist = ((cx - tcx) ** 2 + (cy - tcy) ** 2) ** 0.5
            if dist < best:
                best, bj = dist, j
        # 距离需小于 1.5 倍目标高度
        if bj >= 0 and best < 1.5 * max(40.0, tr['box'][3]):
            assign[tid] = bj
            used.add(bj)

    next_id = (max(tracks.keys()) + 1) if tracks else 1
    updated = {}
    for tid, tr in tracks.items():
        if tid in assign:
            d = dets[assign[tid]]
            updated[tid] = dict(box=d[:4], missed=0, last=frame_idx)
        else:
            updated[tid] = dict(box=tr['box'], missed=tr['missed'] + 1, last=tr['last'])
    for j, d in enumerate(dets):
        if j not in used:
            updated[next_id] = dict(box=d[:4], missed=0, last=frame_idx)
            next_id += 1
    return {k: v for k, v in updated.items() if v['missed'] <= MAX_MISSED}


def judge_video(path, net, zone, step=2, save_annot=False):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print('  无法打开视频')
        return None
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if save_annot:
        writer = cv2.VideoWriter(path + '.judge.avi',
                                 cv2.VideoWriter_fourcc(*'MJPG'),
                                 max(5.0, src_fps / step), (W, H))

    tracks = {}
    traj = {}            # tid -> {'ys':[], 'xs':[], 't0':..,'t1':..}
    approach = None      # {'y','x','dir','t'}
    pending = None       # {'regionY'...} 区域级半程
    state = {}           # tid -> {'cap':bool,'startY':float,'startX':float,'enterT':..,'enterF':..,'miss':int}
    events = []

    fi = 0
    proc = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        if fi % step:
            continue
        proc += 1
        tsec = fi / src_fps

        dets = detect(net, frame)
        tracks = track_frame(tracks, dets, proc)

        # 记录轨迹
        for tid, tr in tracks.items():
            x, y, w, h = tr['box']
            cx, cy = (x + w / 2) / W, (y + h / 2) / H
            traj.setdefault(tid, {'ys': [], 'xs': [], 't0': tsec, 't1': tsec, 'frames': 0})
            traj[tid]['ys'].append(cy)
            traj[tid]['xs'].append(cx)
            traj[tid]['t1'] = tsec
            traj[tid]['frames'] += 1

        # ==== 判定（复现 ZoneMonitor）====
        zone_h = zone['bottomY'] - zone['topY']
        min_box_h = max(0.10, zone_h * 0.35)
        active_ids = set(tracks.keys())

        for tid, tr in tracks.items():
            x, y, w, h = tr['box']
            boxN = (x / W, y / H, w / W, h / H)
            # 与区域矩形求交
            ix1, iy1 = max(boxN[0], zone['leftX']), max(boxN[1], zone['topY'])
            ix2, iy2 = min(boxN[0] + boxN[2], zone['rightX']), min(boxN[1] + boxN[3], zone['bottomY'])
            overlap = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            cy = (y + h / 2) / H
            cx = (x + w / 2) / W
            st = state.setdefault(tid, {'cap': False, 'startY': None, 'startX': cx,
                                        'enterT': tsec, 'miss': 0, 'contact': 0})

            # 区域外「接近点」：靠近通道口 + 方向推断
            if overlap <= 0:
                prev_cy = traj[tid]['ys'][-2] if len(traj[tid]['ys']) > 1 else cy
                dy = cy - prev_cy
                dirg = 1 if dy > 0.001 else (-1 if dy < -0.001 else (-1 if cy > (zone['topY'] + zone['bottomY']) / 2 else 1))
                outside = (cy <= zone['topY'] - zone_h * 0.05) if dirg > 0 else (cy >= zone['bottomY'] + zone_h * 0.05)
                near = (cy >= zone['topY'] - zone_h) if dirg > 0 else (cy <= zone['bottomY'] + zone_h)
                if outside and near:
                    if (approach is None or approach['dir'] != dirg or
                            (dirg > 0 and cy < approach['y']) or (dirg < 0 and cy > approach['y'])):
                        approach = {'y': cy, 'x': cx, 'dir': dirg, 't': tsec}

            # 启动/继续采样
            big_enough = h / H >= min_box_h
            if overlap >= MIN_OVERLAP_AREA and big_enough:
                if not st['cap']:
                    st['cap'] = True
                    st['enterT'] = tsec
                    st['miss'] = 0
                    # 起点：进入区域前的位置（若本轨迹没有，则用接近点接力）
                    startY = None
                    if len(traj[tid]['ys']) > 1 and traj[tid]['ys'][-2] < zone['topY'] + zone_h * 1.5:
                        startY = traj[tid]['ys'][-2]
                    if approach and approach['dir'] == dirg and abs(approach['x'] - cx) <= APPROACH_NEAR_X \
                            and (tsec - approach['t']) <= APPROACH_GRACE_S:
                        startY = approach['y']
                        st['resumed'] = 'approach'
                    if pending and abs(pending['x'] - cx) <= 0.25 and (tsec - pending['t']) <= PENDING_GRACE_S \
                            and pending['dir'] == dirg:
                        startY = pending['y']
                        st['resumed'] = 'zone-pending'
                    st['startY'] = startY if startY is not None else cy
                    st['startX'] = cx
                    st['dir'] = dirg
                st['contact'] = 0
                st['endY'] = cy
            else:
                if st['cap']:
                    st['contact'] += 1
                    if st['contact'] >= CONFIRM_LEAVE_FRAMES:
                        startY, endY = st['startY'], st.get('endY', cy)
                        dyt = endY - startY
                        boarding = dyt > 0 and startY <= zone['topY'] and endY >= zone['bottomY']
                        alighting = dyt < 0 and startY >= zone['bottomY'] and endY <= zone['topY']
                        if boarding or alighting:
                            events.append(dict(tid=tid, type='boarding' if boarding else 'alighting',
                                               startY=round(startY, 4), endY=round(endY, 4),
                                               dy=round(dyt, 4), t=round(tsec, 2),
                                               resumed=st.get('resumed', '')))
                            pending = None
                        else:
                            # 未完成 → 记录区域级半程（供跨 ID 接力）
                            if (st['dir'] > 0 and startY <= zone['topY']) or (st['dir'] < 0 and startY >= zone['bottomY']):
                                pending = {'y': startY, 'x': st['startX'], 'dir': st['dir'], 't': tsec}
                        st['cap'] = False
                        st['resumed'] = ''
                # 未启动采样但轨迹在区域内且框够大（属于"从区域内部出现"）——也尝试接力启动
                elif overlap >= MIN_OVERLAP_AREA and big_enough and not st['cap']:
                    pass

        # 过期
        if pending and (tsec - pending['t']) > PENDING_GRACE_S:
            pending = None
        if approach and (tsec - approach['t']) > APPROACH_GRACE_S:
            approach = None

        if writer is not None:
            vis = frame.copy()
            cv2.rectangle(vis, (int(zone['leftX'] * W), int(zone['topY'] * H)),
                          (int(zone['rightX'] * W), int(zone['bottomY'] * H)), (0, 255, 0), 2)
            for tid, tr in tracks.items():
                x, y, w, h = [int(v) for v in tr['box']]
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 255), 2)
                cv2.putText(vis, f'ID{tid}', (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            writer.write(vis)

    cap.release()
    if writer is not None:
        writer.release()

    print(f'  帧数: 读取 {fi} / 检测 {proc}  |  轨迹数: {len(traj)}')
    for tid in sorted(traj.keys()):
        ys, xs = traj[tid]['ys'], traj[tid]['xs']
        print(f'    ID{tid}: y[{min(ys):.3f}~{max(ys):.3f}] x[{min(xs):.3f}~{max(xs):.3f}] '
              f'frames={traj[tid]["frames"]} t={traj[tid]["t0"]:.1f}~{traj[tid]["t1"]:.1f}s')
    for e in events:
        print(f'    ★ {e["type"]}: ID{e["tid"]} startY={e["startY"]} endY={e["endY"]} '
              f'dy={e["dy"]} t={e["t"]}s' + (f' (接力:{e["resumed"]})' if e['resumed'] else ''))

    nb = sum(1 for e in events if e['type'] == 'boarding')
    na = sum(1 for e in events if e['type'] == 'alighting')
    if nb and not na:
        verdict = '上车（从上往下完整穿越）'
    elif na and not nb:
        verdict = '下车（从下往上完整穿越）'
    elif not nb and not na:
        verdict = '未检测到完整穿越'
    else:
        verdict = f'多次穿越（上{nb}/下{na}）'
    print(f'  结论: {verdict}')
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('target')
    ap.add_argument('--step', type=int, default=2)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--save-annot', action='store_true')
    args = ap.parse_args()

    zone = load_zone(REGIONS)
    print(f'区域: x[{zone["leftX"]:.3f}~{zone["rightX"]:.3f}] y[{zone["topY"]:.3f}~{zone["bottomY"]:.3f}]')
    net = cv2.dnn.readNetFromONNX(MODEL)

    files = sorted(glob.glob(os.path.join(args.target, '*.mp4'))) if os.path.isdir(args.target) else [args.target]
    if args.limit > 0:
        files = files[-args.limit:]
    for f in files:
        print('=' * 70)
        print('视频:', os.path.basename(f))
        judge_video(f, net, zone, step=args.step, save_annot=args.save_annot)


if __name__ == '__main__':
    main()
