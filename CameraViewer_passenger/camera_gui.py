#!/usr/bin/env python3
"""
HX-CCD23 相机实时采集 GUI 界面
============================
功能:
  - 实时显示相机视频流
  - 保存/不保存图像的开关
  - 手动拍摄按钮
  - 信息显示（FPS、分辨率、时间等）

用法:
  python3 camera_gui.py
  python3 camera_gui.py --url rtsp://192.168.99.10:8554/live/camera
"""

import cv2
import os
import sys
import time
import argparse
import threading
from datetime import datetime

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    from PIL import Image, ImageTk
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False
    print("需要安装: pip install Pillow")

# ========== 配置 ==========
DEFAULT_RTSP_URL = "rtsp://192.168.99.10:8554/live/camera"
DEFAULT_SAVE_DIR = "captured_images"


class CameraApp:
    """相机实时采集 GUI 应用"""

    def __init__(self, root, rtsp_url, save_dir):
        self.root = root
        self.rtsp_url = rtsp_url
        self.save_dir = save_dir

        # 状态变量
        self.is_connected = False
        self.is_running = True
        self.auto_save = tk.BooleanVar(value=False)  # 是否自动保存
        self.fps = 0
        self.frame_count = 0
        self.fps_start = time.time()
        self.saved_count = 0
        self.last_frame = None
        self.current_frame = None

        # 视频捕获
        self.cap = None

        # 初始化 GUI
        self._setup_gui()

        # 启动视频线程
        self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.video_thread.start()

        # 启动 FPS 更新
        self._update_info()

        # 绑定关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_gui(self):
        """初始化 GUI 布局"""
        self.root.title("HX-CCD23 相机实时采集")
        self.root.configure(bg="#2b2b2b")

        # 设置窗口大小
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        win_w = min(1280, screen_w - 100)
        win_h = min(800, screen_h - 100)
        x = (screen_w - win_w) // 2
        y = (screen_h - win_h) // 2
        self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")

        # ---- 顶部工具栏 ----
        toolbar = tk.Frame(self.root, bg="#3c3f41", height=50)
        toolbar.pack(fill=tk.X, padx=0, pady=0)
        toolbar.pack_propagate(False)

        # 连接状态指示灯
        self.status_indicator = tk.Canvas(toolbar, width=16, height=16, bg="#3c3f41", highlightthickness=0)
        self.status_indicator.pack(side=tk.LEFT, padx=(15, 5), pady=15)
        self.status_dot = self.status_indicator.create_oval(2, 2, 14, 14, fill="red", outline="")

        self.status_label = tk.Label(toolbar, text="未连接", fg="white", bg="#3c3f41",
                                      font=("Arial", 10))
        self.status_label.pack(side=tk.LEFT, padx=5)

        # 分辨率标签
        self.resolution_label = tk.Label(toolbar, text="", fg="#aaa", bg="#3c3f41",
                                          font=("Arial", 10))
        self.resolution_label.pack(side=tk.LEFT, padx=20)

        # FPS 标签
        self.fps_label = tk.Label(toolbar, text="FPS: --", fg="#aaa", bg="#3c3f41",
                                   font=("Arial", 10))
        self.fps_label.pack(side=tk.LEFT, padx=20)

        # 时间标签
        self.time_label = tk.Label(toolbar, text="", fg="#aaa", bg="#3c3f41",
                                    font=("Arial", 10))
        self.time_label.pack(side=tk.LEFT, padx=20)

        # 保存计数
        self.save_count_label = tk.Label(toolbar, text="已保存: 0", fg="#4CAF50", bg="#3c3f41",
                                          font=("Arial", 10, "bold"))
        self.save_count_label.pack(side=tk.RIGHT, padx=15)

        # ---- 视频显示区域 ----
        video_frame = tk.Frame(self.root, bg="#1e1e1e")
        video_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.video_label = tk.Label(video_frame, bg="#1e1e1e", text="正在连接相机...",
                                     fg="white", font=("Arial", 16))
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # ---- 底部控制栏 ----
        control_frame = tk.Frame(self.root, bg="#3c3f41", height=70)
        control_frame.pack(fill=tk.X, padx=0, pady=0)
        control_frame.pack_propagate(False)

        # 左侧：自动保存开关
        left_frame = tk.Frame(control_frame, bg="#3c3f41")
        left_frame.pack(side=tk.LEFT, padx=20, pady=15)

        self.auto_save_check = tk.Checkbutton(
            left_frame, text="自动保存图像", variable=self.auto_save,
            fg="white", bg="#3c3f41", selectcolor="#555",
            activebackground="#3c3f41", activeforeground="white",
            font=("Arial", 11), cursor="hand2"
        )
        self.auto_save_check.pack(side=tk.LEFT)

        # 自动保存间隔
        tk.Label(left_frame, text="间隔(秒):", fg="#aaa", bg="#3c3f41",
                 font=("Arial", 10)).pack(side=tk.LEFT, padx=(20, 5))
        self.interval_var = tk.StringVar(value="1.0")
        interval_spin = tk.Spinbox(left_frame, from_=0.1, to=60, increment=0.5,
                                    textvariable=self.interval_var, width=5,
                                    font=("Arial", 10))
        interval_spin.pack(side=tk.LEFT)

        # 中间：拍摄按钮
        center_frame = tk.Frame(control_frame, bg="#3c3f41")
        center_frame.pack(side=tk.LEFT, expand=True, pady=15)

        self.capture_btn = tk.Button(
            center_frame, text="📷  拍  摄", font=("Arial", 14, "bold"),
            bg="#4CAF50", fg="white", activebackground="#45a049",
            activeforeground="white", relief=tk.FLAT, padx=30, pady=8,
            cursor="hand2", command=self._manual_capture
        )
        self.capture_btn.pack()

        # 右侧：设置
        right_frame = tk.Frame(control_frame, bg="#3c3f41")
        right_frame.pack(side=tk.RIGHT, padx=20, pady=15)

        # 保存目录
        dir_btn = tk.Button(right_frame, text="📁 保存目录", font=("Arial", 10),
                             bg="#555", fg="white", relief=tk.FLAT, padx=10, pady=5,
                             cursor="hand2", command=self._choose_dir)
        dir_btn.pack(side=tk.RIGHT, padx=5)

        self.dir_label = tk.Label(right_frame, text=f"📁 {self.save_dir}", fg="#aaa",
                                   bg="#3c3f41", font=("Arial", 9))
        self.dir_label.pack(side=tk.RIGHT, padx=5)

        # 自动保存的定时器
        self._auto_save_timer = None

    def _connect_camera(self):
        """连接相机"""
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
        self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)

        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.is_connected = True
                h, w = frame.shape[:2]
                self.root.after(0, lambda: self._update_status(True, f"{w}x{h}"))
                return True

        self.is_connected = False
        self.root.after(0, lambda: self._update_status(False, ""))
        return False

    def _update_status(self, connected, resolution=""):
        """更新连接状态显示"""
        if connected:
            self.status_indicator.itemconfig(self.status_dot, fill="#4CAF50")
            self.status_label.config(text="已连接", fg="#4CAF50")
            self.resolution_label.config(text=f"分辨率: {resolution}")
        else:
            self.status_indicator.itemconfig(self.status_dot, fill="red")
            self.status_label.config(text="未连接", fg="red")
            self.resolution_label.config(text="")

    def _video_loop(self):
        """视频循环（在后台线程中运行）"""
        while self.is_running:
            if not self.is_connected:
                self._connect_camera()
                if not self.is_connected:
                    time.sleep(2)
                    continue

            ret, frame = self.cap.read()
            if not ret:
                self.is_connected = False
                self.root.after(0, lambda: self._update_status(False, ""))
                continue

            self.current_frame = frame.copy()
            self.frame_count += 1

            # 计算 FPS
            elapsed = time.time() - self.fps_start
            if elapsed >= 1.0:
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self.fps_start = time.time()

            # 自动保存
            if self.auto_save.get():
                self._auto_save_frame(frame)

            # 显示帧（在主线程中）
            self.root.after(0, lambda f=frame: self._display_frame(f))

        # 释放资源
        if self.cap is not None:
            self.cap.release()

    def _display_frame(self, frame):
        """在 GUI 中显示帧"""
        try:
            # 获取标签大小
            label_w = self.video_label.winfo_width()
            label_h = self.video_label.winfo_height()

            if label_w <= 1 or label_h <= 1:
                return

            # 保持宽高比缩放
            h, w = frame.shape[:2]
            scale = min(label_w / w, label_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            frame_resized = cv2.resize(frame, (new_w, new_h))

            # BGR -> RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk  # 防止垃圾回收
            self.video_label.configure(image=imgtk, text="")
        except Exception:
            pass

    def _update_info(self):
        """定时更新信息显示"""
        if not self.is_running:
            return

        # 更新 FPS
        self.fps_label.config(text=f"FPS: {self.fps:.1f}")

        # 更新时间
        self.time_label.config(text=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # 更新保存计数
        self.save_count_label.config(text=f"已保存: {self.saved_count}")

        self.root.after(500, self._update_info)

    def _manual_capture(self):
        """手动拍摄"""
        if self.current_frame is None:
            messagebox.showwarning("提示", "没有可保存的帧，请等待相机连接")
            return

        filepath = self._save_frame(self.current_frame)
        if filepath:
            self.saved_count += 1
            # 按钮闪烁效果
            self.capture_btn.config(bg="#FFC107")
            self.root.after(200, lambda: self.capture_btn.config(bg="#4CAF50"))

    def _auto_save_frame(self, frame):
        """自动保存帧"""
        try:
            interval = float(self.interval_var.get())
        except ValueError:
            interval = 1.0

        now = time.time()
        if not hasattr(self, '_last_auto_save'):
            self._last_auto_save = 0

        if now - self._last_auto_save >= interval:
            self._last_auto_save = now
            filepath = self._save_frame(frame, prefix="auto")
            if filepath:
                self.saved_count += 1

    def _save_frame(self, frame, prefix="capture"):
        """保存帧到文件"""
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            filepath = os.path.join(self.save_dir, filename)
            cv2.imwrite(filepath, frame)
            return filepath
        except Exception as e:
            print(f"保存失败: {e}")
            return None

    def _choose_dir(self):
        """选择保存目录"""
        dir_path = filedialog.askdirectory(initialdir=self.save_dir, title="选择保存目录")
        if dir_path:
            self.save_dir = dir_path
            self.dir_label.config(text=f"📁 {self.save_dir}")

    def _on_close(self):
        """关闭窗口"""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="HX-CCD23 相机实时采集 GUI")
    parser.add_argument("--url", default=DEFAULT_RTSP_URL, help=f"RTSP 流地址 (默认: {DEFAULT_RTSP_URL})")
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR, help=f"保存目录 (默认: {DEFAULT_SAVE_DIR})")
    args = parser.parse_args()

    if not HAS_TKINTER:
        print("请安装 Pillow: pip install Pillow")
        return 1

    root = tk.Tk()
    app = CameraApp(root, args.url, args.save_dir)
    root.mainloop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
