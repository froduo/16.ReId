#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
乘客识别程序（TensorRT + CUDA）
================================
使用当前最新模型 passenger_yolov8n_trt10_x86_20261013_180200.engine：
  输入  : 1x3x640x640, float32, RGB, letterbox(114灰边), /255 归一化
  输出  : 1x5x8400 (cx, cy, w, h, score)，单类 person（class 0）

识别到乘客后，输出一个可直接用于 YOLO 训练的数据集：
  <output>/images/*.jpg   干净整帧全图（不绘制标注，避免模型学到画框）
  <output>/labels/*.txt   YOLO 格式标注：每行 class cx cy w h（归一化 0~1）
  <output>/data.yaml      数据集描述文件（nc=1, names=['person']）
当一帧内有多名乘客时仅保存一张全图（对应标注文件内为多行框）。
可选 --vis 额外输出带框预览图到 <output>/vis/ 子目录，便于人工抽检。

用法:
  python3 tools/passenger_detect_trt.py \
      --input  /path/to/ch01_xxx.mp4 \
      --model  model/passenger_yolov8n_trt10_x86_20261013_180200.engine \
      --output ./passenger_detected \
      --conf 0.30

依赖: numpy, opencv-python, tensorrt, cuda-python (from cuda.bindings import runtime)
"""

import argparse
import os
import sys
import time

import numpy as np
import cv2

import tensorrt as trt
from cuda.bindings import runtime as cudart

INPUT_W = 640
INPUT_H = 640
CONF_THRESH = 0.30
NMS_THRESH = 0.45
PAD_COLOR = 114


class TrtYolo:
    """TensorRT YOLO 检测封装（与 C++ TrtYolo 的预处理/后处理一致）"""

    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        if engine_data[:4] != b"ftrt":
            raise RuntimeError(f"不是有效的 TensorRT 引擎文件: {engine_path}")
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError(f"反序列化引擎失败: {engine_path}")
        self.context = self.engine.create_execution_context()

        self.input_name = None
        self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            elif self.output_name is None:
                self.output_name = name
        if not self.input_name or not self.output_name:
            raise RuntimeError("未找到引擎输入/输出张量")

        in_shape = self.engine.get_tensor_shape(self.input_name)
        self.input_h = int(in_shape[2])
        self.input_w = int(in_shape[3])
        self.context.set_input_shape(self.input_name, (1, 3, self.input_h, self.input_w))

        out_shape = self.engine.get_tensor_shape(self.output_name)
        self.out_shape = tuple(int(d) for d in out_shape)
        self.out_size = int(np.prod(self.out_shape))
        self.in_size = 1 * 3 * self.input_h * self.input_w

        err, self.d_in = cudart.cudaMalloc(self.in_size * np.dtype(np.float32).itemsize)
        self._check(err, "cudaMalloc d_in")
        err, self.d_out = cudart.cudaMalloc(self.out_size * np.dtype(np.float32).itemsize)
        self._check(err, "cudaMalloc d_out")
        err, self.stream = cudart.cudaStreamCreate()
        self._check(err, "cudaStreamCreate")

        self.context.set_tensor_address(self.input_name, self.d_in)
        self.context.set_tensor_address(self.output_name, self.d_out)

        self.h_in = np.empty((self.in_size,), np.float32)
        self.h_out = np.empty((self.out_size,), np.float32)

        print(f"[TrtYolo] 引擎加载成功: {engine_path}")
        print(f"[TrtYolo] input={self.input_w}x{self.input_h}, output_shape={self.out_shape}, "
              f"out_size={self.out_size}")

    @staticmethod
    def _check(err, what):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"{what} 失败: {err}")

    def _preprocess(self, bgr):
        h, w = bgr.shape[:2]
        scale = min(self.input_w / w, self.input_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(bgr, (new_w, new_h))
        padded = np.full((self.input_h, self.input_w, 3), PAD_COLOR, np.uint8)
        pad_x = (self.input_w - new_w) // 2
        pad_y = (self.input_h - new_h) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        rgb = padded[:, :, ::-1]  # BGR -> RGB
        f = rgb.astype(np.float32) / 255.0
        chw = np.ascontiguousarray(np.transpose(f, (2, 0, 1)))
        return chw.reshape(-1), scale, pad_x, pad_y, (w, h)

    def detect(self, bgr, conf_thresh=CONF_THRESH):
        """返回 [(x, y, w, h, score), ...] 在原图坐标系下的检测框"""
        inp, scale, pad_x, pad_y, orig = self._preprocess(bgr)

        cudart.cudaMemcpy(self.d_in, inp.ctypes.data, inp.nbytes,
                          cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        if not self.context.execute_async_v3(self.stream):
            raise RuntimeError("TensorRT 推理失败")
        cudart.cudaStreamSynchronize(self.stream)
        cudart.cudaMemcpy(self.h_out.ctypes.data, self.d_out, self.h_out.nbytes,
                          cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        out = self.h_out
        d1, d2 = self.out_shape[1], self.out_shape[2]
        boxes, scores = [], []

        if d2 == 6 and d1 <= 300:
            # yolo26 格式: [1, numDet, 6] => x1,y1,x2,y2,conf,class
            for i in range(d1):
                conf = float(out[i * 6 + 4])
                cls = int(out[i * 6 + 5])
                if conf < conf_thresh or cls != 0:
                    continue
                x1 = (out[i * 6 + 0] - pad_x) / scale
                y1 = (out[i * 6 + 1] - pad_y) / scale
                x2 = (out[i * 6 + 2] - pad_x) / scale
                y2 = (out[i * 6 + 3] - pad_y) / scale
                x, y = max(0, int(x1)), max(0, int(y1))
                bw, bh = int(x2 - x1), int(y2 - y1)
                if bw > 0 and bh > 0:
                    boxes.append([x, y, bw, bh])
                    scores.append(conf)
        else:
            is_layout_a = (d1 >= 5 and d1 <= 512 and d2 > d1)
            num_attr = d1 if is_layout_a else d2
            num_pred = d2 if is_layout_a else d1
            for i in range(num_pred):
                if is_layout_a:
                    score = float(out[4 * num_pred + i])
                    cx, cy = float(out[0 * num_pred + i]), float(out[1 * num_pred + i])
                    bw, bh = float(out[2 * num_pred + i]), float(out[3 * num_pred + i])
                else:
                    score = float(out[i * num_attr + 4])
                    cx, cy = float(out[i * num_attr + 0]), float(out[i * num_attr + 1])
                    bw, bh = float(out[i * num_attr + 2]), float(out[i * num_attr + 3])
                if score < conf_thresh:
                    continue
                cx = (cx - pad_x) / scale
                cy = (cy - pad_y) / scale
                bw = bw / scale
                bh = bh / scale
                x, y = max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2))
                iw, ih = int(bw), int(bh)
                if iw > 0 and ih > 0:
                    boxes.append([x, y, iw, ih])
                    scores.append(score)

        if not boxes:
            return []

        idx = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, NMS_THRESH)
        results = []
        for i in idx:
            i = int(i)
            x, y, w, h = boxes[i]
            results.append((x, y, w, h, scores[i]))
        return results

    def close(self):
        cudart.cudaFree(self.d_in)
        cudart.cudaFree(self.d_out)
        cudart.cudaStreamDestroy(self.stream)


def list_inputs(path):
    """输入既可以是视频文件，也可以是图片目录（按文件名排序）"""
    if os.path.isdir(path):
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        return sorted([os.path.join(path, n) for n in os.listdir(path)
                       if n.lower().endswith(exts)])
    return [path]


def write_yolo_label(txt_path, dets, img_w, img_h):
    """把检测框写成 YOLO 标注：每行 class cx cy w h（归一化 0~1）"""
    lines = []
    for (x, y, w, h, _score) in dets:
        cx = (x + w / 2.0) / img_w
        cy = (y + h / 2.0) / img_h
        bw = w / img_w
        bh = h / img_h
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        bw = min(max(bw, 0.0), 1.0)
        bh = min(max(bh, 0.0), 1.0)
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def main():
    ap = argparse.ArgumentParser(description="TensorRT 乘客识别（CUDA）")
    ap.add_argument("--input", required=True, help="输入视频文件或图片目录")
    ap.add_argument("--model", default="model/passenger_yolov8n_trt10_x86_20261013_180200.engine",
                    help="TensorRT 引擎路径")
    ap.add_argument("--output", default="./passenger_detected", help="数据集输出目录（images/ + labels/ + data.yaml）")
    ap.add_argument("--conf", type=float, default=CONF_THRESH, help="置信度阈值")
    ap.add_argument("--every", type=int, default=1, help="每 N 帧检测一次（默认 1）")
    ap.add_argument("--max-frames", type=int, default=0, help="最多处理帧数（0=不限）")
    ap.add_argument("--vis", action="store_true", help="额外输出带框预览图到 vis/ 子目录")
    args = ap.parse_args()

    model_path = args.model
    if not os.path.isabs(model_path):
        here = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.abspath(os.path.join(here, "..", model_path))
    if not os.path.exists(model_path):
        print(f"[错误] 模型文件不存在: {model_path}", file=sys.stderr)
        return 1

    os.makedirs(args.output, exist_ok=True)
    img_dir = os.path.join(args.output, "images")
    lbl_dir = os.path.join(args.output, "labels")
    vis_dir = os.path.join(args.output, "vis")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    if args.vis:
        os.makedirs(vis_dir, exist_ok=True)

    data_yaml = os.path.join(args.output, "data.yaml")
    with open(data_yaml, "w") as f:
        f.write(f"path: {os.path.abspath(args.output)}\n"
                "train: images\n"
                "val: images\n"
                "nc: 1\n"
                "names: ['person']\n")
    print(f"[输出目录] {args.output}  (images/ + labels/ + data.yaml)")

    model = TrtYolo(model_path)

    inputs = list_inputs(args.input)
    if not inputs:
        print(f"[错误] 输入不存在或为空: {args.input}", file=sys.stderr)
        return 1

    is_video = not os.path.isdir(args.input)
    cap = cv2.VideoCapture(inputs[0]) if is_video else None
    if is_video and not cap.isOpened():
        print(f"[错误] 无法打开视频: {inputs[0]}", file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) if is_video else 0.0
    if fps <= 0:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_video else len(inputs)

    frame_idx = 0
    saved_total = 0
    t0 = time.time()

    try:
        while True:
            if is_video:
                if args.max_frames and frame_idx >= args.max_frames:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
            else:
                if frame_idx >= len(inputs):
                    break
                frame = cv2.imread(inputs[frame_idx])
                if frame is None:
                    frame_idx += 1
                    continue

            if frame_idx % args.every != 0:
                frame_idx += 1
                continue

            dets = model.detect(frame, args.conf)
            if not dets:
                frame_idx += 1
                continue

            # 识别到乘客 → 保存干净全图 + YOLO 标注文本（可直接训练）
            h, w = frame.shape[:2]
            ts = time.strftime("%H%M%S", time.gmtime(frame_idx / fps))
            base = f"passenger_{frame_idx:06d}_{ts}"
            cv2.imwrite(os.path.join(img_dir, base + ".jpg"), frame)   # 干净原图，不带标注
            write_yolo_label(os.path.join(lbl_dir, base + ".txt"), dets, w, h)

            if args.vis:
                vis = frame.copy()
                for (x, y, bw, bh, score) in dets:
                    cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                    cv2.putText(vis, f"person {score:.2f}", (x, max(20, y - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imwrite(os.path.join(vis_dir, base + ".jpg"), vis)

            saved_total += 1

            if frame_idx % 100 == 0:
                fps_now = frame_idx / max(1e-3, time.time() - t0)
                print(f"[进度] 已处理 {frame_idx}/{total_frames if total_frames else '?'} 帧, "
                      f"保存乘客全图累计 {saved_total} 张, 速度 {fps_now:.1f} fps")
            frame_idx += 1
    finally:
        if cap is not None:
            cap.release()
        model.close()

    elapsed = time.time() - t0
    print(f"[完成] 共处理 {frame_idx} 帧, 保存乘客全图+标注 {saved_total} 份到: {args.output}")
    print(f"[完成] 耗时 {elapsed:.1f}s, 平均 {frame_idx / max(1e-3, elapsed):.1f} fps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
