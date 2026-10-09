#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
head_verify_gui.py —— 模型识别准确性可视化验证工具（PyQt5）

用途
    在“当前模型”（默认 head_yolo11n.onnx）上逐张验证检测准确性：
    - 打开程序 → 点击「开始」→ 自动识别第一张
    - 点击「下一张 / 上一张」→ 自动识别下一张 / 上一张
    - 同时叠加显示 Ground Truth（VOC XML，绿色）与模型预测（红色，
      命中真值的预测显示为蓝色），并在状态栏给出本张 P/R 与累计准确率。

数据集
    默认目录（PHD VOC2007，3717 张 head 标注图）：
      /run/user/1000/gvfs/sftp:host=192.168.200.8/share/vision/head数据集/数据集1/dataset/PHD/VOC2007/JPEGImages
    对应标注（可选）：同级的 ../Annotations/*.xml

依赖
    PyQt5, opencv-python(cv2), numpy, onnxruntime

运行
    python3 head_verify_gui.py
    # 或
    ./head_verify.sh
"""

import os
import sys
import glob
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import onnxruntime as ort


def _fix_qt_plugins():
    """
    修复 opencv-python 与 PyQt5 的 Qt 插件冲突。

    `import cv2` 会把环境变量 QT_QPA_PLATFORM_PLUGIN_PATH 指向 cv2 自带的
    插件目录（.../cv2/qt/plugins），导致 PyQt5 去那里找 xcb，从而报
    “Could not load the Qt platform plugin xcb”。这里在导入 PyQt5 之前，
    把插件路径改回真正带平台插件的 Qt 目录。
    """
    import importlib.util

    cands = []
    conda = os.environ.get("CONDA_PREFIX")
    if conda:
        cands.append(os.path.join(conda, "plugins"))
    spec = importlib.util.find_spec("PyQt5")
    if spec and spec.submodule_search_locations:
        pkg = list(spec.submodule_search_locations)[0]
        cands += [os.path.join(pkg, "Qt5", "plugins"), os.path.join(pkg, "Qt", "plugins")]
    cands.append(os.path.expanduser(
        "~/.local/lib/python3.11/site-packages/PyQt5/Qt5/plugins"))

    chosen = next((c for c in cands if os.path.isdir(os.path.join(c, "platforms"))), None)

    # 关键：无论 cv2 之前写了什么，这里强制覆盖
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    if chosen:
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = chosen
        os.environ["QT_PLUGIN_PATH"] = chosen
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "0")
    return chosen


_fix_qt_plugins()

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QComboBox, QSlider,
    QLineEdit, QTextEdit, QHBoxLayout, QVBoxLayout, QFileDialog, QGroupBox,
    QFormLayout, QSizePolicy,
)

# ============================ 默认参数 ============================
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "CameraViewer", "model", "head_yolo11n.onnx")
MODEL_CHOICES = [
    ("head_yolo11n.onnx (头部模型)", os.path.join(HERE, "CameraViewer", "model", "head_yolo11n.onnx")),
    ("yolov8n.onnx (通用COCO)",       os.path.join(HERE, "CameraViewer", "model", "yolov8n.onnx")),
]
DEFAULT_DIR = ("/run/user/1000/gvfs/sftp:host=192.168.200.8/share/vision/"
               "head数据集/数据集1/dataset/PHD/VOC2007/JPEGImages")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ============================ 核心：推理 / 标注 / 指标 ============================
class OnnxDetector:
    """轻量 ONNX 检测器：支持 letterbox / squash 两种预处理，解析 YOLO 输出。"""

    def __init__(self, model_path, input_size=640):
        self.input_size = input_size
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.input_shape = self.sess.get_inputs()[0].shape

    def _preprocess(self, img, mode):
        H = W = self.input_size
        h, w = img.shape[:2]
        if mode == "letterbox":
            s = min(H / h, W / w)
            nw, nh = int(round(w * s)), int(round(h * s))
            resized = cv2.resize(img, (nw, nh))
            canvas = np.full((H, W, 3), 114, np.uint8)
            dw, dh = (W - nw) // 2, (H - nh) // 2
            canvas[dh:dh + nh, dw:dw + nw] = resized
            sx = sy = s
            px, py = dw, dh
        else:  # squash
            canvas = cv2.resize(img, (W, H))
            sx, sy = W / w, H / h
            px = py = 0
        x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)[None]
        return x, sx, sy, px, py

    def detect(self, img, conf=0.25, iou=0.45, mode="letterbox", class_id=0,
               max_det=300):
        """返回 [(x1, y1, x2, y2, score), ...]（原图像素坐标）。"""
        x, sx, sy, px, py = self._preprocess(img, mode)
        out = self.sess.run(None, {self.input_name: x})[0]

        # 兼容 [1, N, 4+nc] 与 [1, 4+nc, N] 两种布局
        if out.ndim != 3:
            return []
        a, b = out.shape[1], out.shape[2]
        if b <= 512 and a > b:           # [1, N, attrs]
            pred = out[0]
        else:                            # [1, attrs, N]
            pred = out[0].transpose(1, 0)

        nc = pred.shape[1] - 4
        if nc <= 0:
            return []

        # 若为多类，取指定类分数；单类则用第 4 列
        if nc == 1:
            scores = pred[:, 4]
        else:
            scores = pred[:, 4 + class_id]

        idx = np.where(scores >= conf)[0]
        if idx.size == 0:
            return []
        boxes_raw = pred[idx, :4]
        sc = scores[idx]

        # cx,cy,w,h → 原图坐标 x1,y1,x2,y2
        cx, cy, bw, bh = boxes_raw[:, 0], boxes_raw[:, 1], boxes_raw[:, 2], boxes_raw[:, 3]
        x1 = (cx - bw / 2 - px) / sx
        y1 = (cy - bh / 2 - py) / sy
        x2 = (cx + bw / 2 - px) / sx
        y2 = (cy + bh / 2 - py) / sy
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        keep = _nms(boxes, sc, iou, max_det)
        h, w = img.shape[:2]
        res = []
        for k in keep:
            bx1, by1, bx2, by2 = boxes[k]
            bx1 = max(0, min(w - 1, bx1)); by1 = max(0, min(h - 1, by1))
            bx2 = max(0, min(w - 1, bx2)); by2 = max(0, min(h - 1, by2))
            if bx2 - bx1 < 1 or by2 - by1 < 1:
                continue
            res.append((float(bx1), float(by1), float(bx2), float(by2), float(sc[k])))
        return res


def _nms(boxes, scores, iou_thr, max_det):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0 and len(keep) < max_det:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = np.where(union > 0, inter / union, 0)
        order = order[1:][iou <= iou_thr]
    return keep


def load_voc_gt(xml_path):
    """读取 VOC XML，返回 [(x1, y1, x2, y2), ...]（仅取所有 object）。"""
    if not xml_path or not os.path.isfile(xml_path):
        return None
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return None
    gts = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            x1 = float(bb.findtext("xmin")); y1 = float(bb.findtext("ymin"))
            x2 = float(bb.findtext("xmax")); y2 = float(bb.findtext("ymax"))
        except (TypeError, ValueError):
            continue
        gts.append((x1, y1, x2, y2))
    return gts


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match_metrics(preds, gts, iou_thr=0.5):
    """
    返回 (tp, fp, fn, precision, recall, f1, matched_pred_idx_set)
    preds: [(x1,y1,x2,y2,score)...]; gts: [(x1,y1,x2,y2)...] 或 None
    """
    if gts is None:
        return None
    matched_gt = set()
    matched_pred = set()
    pbox = [p[:4] for p in preds]
    for pi, pb in enumerate(pbox):
        best_j, best_iou = -1, 0.0
        for gi, gb in enumerate(gts):
            if gi in matched_gt:
                continue
            v = _iou(pb, gb)
            if v > best_iou:
                best_iou, best_j = v, gi
        if best_j >= 0 and best_iou >= iou_thr:
            matched_gt.add(best_j)
            matched_pred.add(pi)
    tp = len(matched_pred)
    fp = len(preds) - tp
    fn = len(gts) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return tp, fp, fn, precision, recall, f1, matched_pred


def draw_result(img, preds, gts, matched_pred):
    """在原图上画：GT=绿色，预测=红色（命中=蓝色）。返回 BGR 图。"""
    vis = img.copy()
    if gts:
        for (x1, y1, x2, y2) in gts:
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
            cv2.putText(vis, "GT", (int(x1), max(12, int(y1) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
    for i, (x1, y1, x2, y2, s) in enumerate(preds):
        color = (255, 0, 0) if (matched_pred and i in matched_pred) else (0, 0, 255)
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(vis, f"{s:.2f}", (int(x1), max(12, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis


# ============================ 界面 ============================
class VerifyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("模型识别准确性验证工具 —— head_verify_gui")
        self.resize(1280, 860)

        self.detector = None
        self.model_path = None
        self.images = []
        self.index = -1
        self.cur_img = None
        self.acc = {"TP": 0, "FP": 0, "FN": 0, "imgs": 0}

        self._build_ui()

    # ---------- UI ----------
    def _build_ui(self):
        root = QVBoxLayout(self)

        # 顶部控制区
        top = QHBoxLayout()
        self.btn_start = QPushButton("开始")
        self.btn_prev = QPushButton("上一张")
        self.btn_next = QPushButton("下一张")
        self.btn_dir = QPushButton("选择目录…")
        self.btn_file = QPushButton("选择图像文件…")
        self.btn_start.setMinimumHeight(34)
        self.btn_prev.setMinimumHeight(34)
        self.btn_next.setMinimumHeight(34)
        self.btn_file.setMinimumHeight(34)
        top.addWidget(self.btn_start)
        top.addWidget(self.btn_prev)
        top.addWidget(self.btn_next)
        top.addWidget(self.btn_dir)
        top.addWidget(self.btn_file)
        top.addStretch(1)
        root.addLayout(top)

        # 参数区
        param_box = QGroupBox("参数")
        form = QFormLayout(param_box)

        self.cmb_model = QComboBox()
        self.cmb_model.addItem("-- 点击『开始』后加载 --", "")
        for name, path in MODEL_CHOICES:
            self.cmb_model.addItem(name, path)
        self.cmb_model.setCurrentIndex(1)  # 默认 head_yolo11n

        self.cmb_pre = QComboBox()
        self.cmb_pre.addItem("letterbox（保持比例+灰边）", "letterbox")
        self.cmb_pre.addItem("squash（等比拉伸）", "squash")
        self.cmb_pre.setCurrentIndex(1)    # head 模型上线用 squash

        self.sld_conf = QSlider(Qt.Horizontal)
        self.sld_conf.setRange(1, 95)
        self.sld_conf.setValue(25)         # 默认 0.25，便于观察低分
        self.lbl_conf = QLabel("0.25")

        self.sld_iou = QSlider(Qt.Horizontal)
        self.sld_iou.setRange(10, 90)
        self.sld_iou.setValue(45)
        self.lbl_iou = QLabel("0.45")

        self.edt_dir = QLineEdit(DEFAULT_DIR)

        form.addRow("模型：", self.cmb_model)
        form.addRow("预处理：", self.cmb_pre)
        form.addRow("置信度阈值：", self._h(self.sld_conf, self.lbl_conf))
        form.addRow("NMS IoU：", self._h(self.sld_iou, self.lbl_iou))
        form.addRow("图片目录：", self.edt_dir)
        root.addWidget(param_box)

        # 图像显示
        self.lbl_img = QLabel("点击「开始」加载并识别第一张图片")
        self.lbl_img.setAlignment(Qt.AlignCenter)
        self.lbl_img.setMinimumHeight(460)
        self.lbl_img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_img.setStyleSheet("background:#202020;color:#bbb;border:1px solid #444;")
        root.addWidget(self.lbl_img, 1)

        # 状态/日志
        self.txt = QTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setMaximumHeight(160)
        self.txt.setStyleSheet("font-family:monospace;")
        root.addWidget(self.txt)

        # 事件
        self.btn_start.clicked.connect(self.on_start)
        self.btn_prev.clicked.connect(lambda: self.step(-1))
        self.btn_next.clicked.connect(lambda: self.step(+1))
        self.btn_dir.clicked.connect(self.on_choose_dir)
        self.btn_file.clicked.connect(self.on_choose_file)
        self.sld_conf.valueChanged.connect(lambda v: self.lbl_conf.setText(f"{v/100:.2f}"))
        self.sld_iou.valueChanged.connect(lambda v: self.lbl_iou.setText(f"{v/100:.2f}"))

    @staticmethod
    def _h(slider, label):
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(slider, 1)
        lay.addWidget(label)
        return w

    # ---------- 逻辑 ----------
    def on_choose_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择图片目录", self.edt_dir.text())
        if d:
            self.edt_dir.setText(d)

    def on_choose_file(self):
        """选择一个图像文件并立即识别验证（其后可用上一张/下一张浏览同目录其他图片）。"""
        start = self.edt_dir.text().strip()
        if os.path.isdir(os.path.join(start, "JPEGImages")):
            start = os.path.join(start, "JPEGImages")
        if not os.path.isdir(start):
            start = os.path.dirname(start) if os.path.isfile(start) else os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图像文件", start,
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.webp);;所有文件 (*)")
        if not path:
            return
        if not self._load_model():
            return

        # 装载所选文件所在目录，便于继续用 上一张/下一张 浏览
        folder = os.path.dirname(path)
        files = []
        for ext in IMG_EXTS:
            files += glob.glob(os.path.join(folder, "*" + ext))
            files += glob.glob(os.path.join(folder, "*" + ext.upper()))
        self.images = sorted(set(files))
        if path not in self.images:
            self.images = [path]
        self.index = self.images.index(path)

        self.acc = {"TP": 0, "FP": 0, "FN": 0, "imgs": 0}
        self.txt.append(f">>> 已选择图像文件：{path}（同目录共 {len(self.images)} 张）")
        self._process_current()

    def on_start(self):
        if not self._load_model():
            return
        self._load_images()
        if not self.images:
            self.txt.append(">>> 目录中没有找到图片。")
            return
        self.acc = {"TP": 0, "FP": 0, "FN": 0, "imgs": 0}
        self.txt.clear()
        self.txt.append(f">>> 共 {len(self.images)} 张图片，开始自动识别…")
        self.step(+1)

    def _load_model(self):
        path = self.cmb_model.currentData()
        if not path:
            self.txt.append(">>> 请先在下拉框选择模型。")
            return False
        if not os.path.isfile(path):
            self.txt.append(f">>> 模型不存在：{path}")
            return False
        if self.model_path != path or self.detector is None:
            try:
                self.detector = OnnxDetector(path)
                self.model_path = path
                self.txt.append(f">>> 已加载模型：{path}")
            except Exception as e:
                self.txt.append(f">>> 模型加载失败：{e}")
                return False
        return True

    def _load_images(self):
        d = self.edt_dir.text().strip()
        if os.path.isdir(os.path.join(d, "JPEGImages")):
            d = os.path.join(d, "JPEGImages")
        if not os.path.isdir(d):
            self.images = []
            return
        files = []
        for ext in IMG_EXTS:
            files += glob.glob(os.path.join(d, "*" + ext))
            files += glob.glob(os.path.join(d, "*" + ext.upper()))
        self.images = sorted(set(files))
        self.index = -1

    def _gt_path_for(self, img_path):
        """JPEGImages/xxx.jpg → ../Annotations/xxx.xml"""
        ann = os.path.join(os.path.dirname(os.path.dirname(img_path)), "Annotations")
        base = os.path.splitext(os.path.basename(img_path))[0]
        p = os.path.join(ann, base + ".xml")
        return p if os.path.isfile(p) else None

    def step(self, d):
        if not self.images:
            self.txt.append(">>> 请先点击「开始」。")
            return
        if self.detector is None:
            if not self._load_model():
                return
        new_idx = self.index + d
        if new_idx < 0:
            return
        if new_idx >= len(self.images):
            self.txt.append(">>> 已是最后一张。")
            return
        self.index = new_idx
        self._process_current()

    def _process_current(self):
        img_path = self.images[self.index]
        img = cv2.imread(img_path)
        if img is None:
            self.txt.append(f">>> 无法读取：{img_path}")
            return
        self.cur_img = img

        conf = self.sld_conf.value() / 100.0
        iou = self.sld_iou.value() / 100.0
        pre = self.cmb_pre.currentData()

        preds = self.detector.detect(img, conf=conf, iou=iou, mode=pre, class_id=0)
        gts = load_voc_gt(self._gt_path_for(img_path))
        mt = match_metrics(preds, gts, iou_thr=0.5)
        matched_pred = mt[6] if mt else set()

        vis = draw_result(img, preds, gts, matched_pred)
        self._show(vis)

        head = (f"[{self.index + 1}/{len(self.images)}] {os.path.basename(img_path)}  "
                f"{img.shape[1]}x{img.shape[0]}  预测={len(preds)}  GT={len(gts) if gts is not None else '-'}")
        self.txt.append(head)
        if gts is None:
            self.txt.append("    (未找到对应标注 XML，仅显示预测)")
        else:
            tp, fp, fn, p, r, f1, _ = mt
            self.acc["TP"] += tp; self.acc["FP"] += fp; self.acc["FN"] += fn; self.acc["imgs"] += 1
            self.txt.append(f"    本张: TP={tp} FP={fp} FN={fn}  P={p:.3f} R={r:.3f} F1={f1:.3f}")
            A = self.acc
            P = A["TP"] / (A["TP"] + A["FP"]) if (A["TP"] + A["FP"]) else 0.0
            R = A["TP"] / (A["TP"] + A["FN"]) if (A["TP"] + A["FN"]) else 0.0
            F = 2 * P * R / (P + R) if (P + R) else 0.0
            self.txt.append(f"    累计({A['imgs']}张): P={P:.3f} R={R:.3f} F1={F:.3f} "
                            f"(TP={A['TP']} FP={A['FP']} FN={A['FN']})")
        self.txt.verticalScrollBar().setValue(self.txt.verticalScrollBar().maximum())

    def _show(self, bgr):
        h, w = bgr.shape[:2]
        max_w, max_h = 1180, 640
        s = min(max_w / w, max_h / h, 1.0)
        if s < 1.0:
            bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        hh, ww = rgb.shape[:2]
        qimg = QImage(rgb.data, ww, hh, 3 * ww, QImage.Format_RGB888).copy()
        self.lbl_img.setPixmap(QPixmap.fromImage(qimg))

    def keyPressEvent(self, e):  # 键盘快捷键
        if e.key() in (Qt.Key_Right, Qt.Key_Down):
            self.step(+1)
        elif e.key() in (Qt.Key_Left, Qt.Key_Up):
            self.step(-1)
        elif e.key() == Qt.Key_Space:
            self.on_start()
        else:
            super().keyPressEvent(e)


def main():
    app = QApplication(sys.argv)
    win = VerifyWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
