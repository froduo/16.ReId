#!/usr/bin/env python3
"""
HX-CCD23 相机实时采集 Web GUI
============================
基于 Flask 的 Web 界面，通过浏览器访问实现实时视频显示和图像采集。

功能:
  - 实时视频流（MJPEG）
  - 手动拍摄/保存图像
  - 自动保存开关（可设间隔）
  - 查看已采集图像列表
  - 下载已采集图像

用法:
  python3 camera_webgui.py
  python3 camera_webgui.py --port 5000
  python3 camera_webgui.py --url rtsp://192.168.99.10:8554/live/camera

然后在浏览器打开: http://localhost:5000
"""

import cv2
import os
import sys
import time
import json
import argparse
import threading
from datetime import datetime
from io import BytesIO

from flask import Flask, Response, render_template_string, request, jsonify, send_file

# ========== 配置 ==========
DEFAULT_RTSP_URL = "rtsp://192.168.99.10:8554/live/camera"
DEFAULT_SAVE_DIR = "captured_images"
DEFAULT_PORT = 5000

app = Flask(__name__)

# ========== 全局状态 ==========
class CameraState:
    def __init__(self, rtsp_url, save_dir):
        self.rtsp_url = rtsp_url
        self.save_dir = save_dir
        self.cap = None
        self.current_frame = None
        self.is_connected = False
        self.fps = 0
        self.frame_count = 0
        self.fps_start = time.time()
        self.saved_count = 0
        self.auto_save = False
        self.auto_save_interval = 1.0  # 秒
        self.last_auto_save = 0
        self.lock = threading.Lock()
        self.resolution = ""
        self.running = True
        self.last_error = ""

        # 启动视频线程
        self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.video_thread.start()

    def _connect(self):
        """连接相机"""
        try:
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
            self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
            ret, frame = self.cap.read()
            if ret:
                h, w = frame.shape[:2]
                self.resolution = f"{w}x{h}"
                self.is_connected = True
                self.last_error = ""
                return True
            else:
                self.last_error = "无法读取帧"
        except Exception as e:
            self.last_error = str(e)
        self.is_connected = False
        return False

    def _video_loop(self):
        """视频读取循环"""
        while self.running:
            if not self.is_connected:
                self._connect()
                if not self.is_connected:
                    time.sleep(2)
                    continue

            ret, frame = self.cap.read()
            if not ret:
                self.is_connected = False
                continue

            with self.lock:
                self.current_frame = frame.copy()

            self.frame_count += 1
            elapsed = time.time() - self.fps_start
            if elapsed >= 1.0:
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self.fps_start = time.time()

            # 自动保存
            if self.auto_save:
                now = time.time()
                if now - self.last_auto_save >= self.auto_save_interval:
                    self.last_auto_save = now
                    self.save_frame(frame, "auto")

        if self.cap:
            self.cap.release()

    def get_frame_jpeg(self):
        """获取当前帧的 JPEG 字节"""
        with self.lock:
            if self.current_frame is None:
                return None
            frame = self.current_frame.copy()

        # 添加信息叠加
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        text = f"FPS:{self.fps:.1f} | {w}x{h} | {datetime.now().strftime('%H:%M:%S')} | Saved:{self.saved_count}"
        cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return jpeg.tobytes()

    def save_frame(self, frame=None, prefix="capture"):
        """保存帧"""
        if frame is None:
            with self.lock:
                if self.current_frame is None:
                    return None
                frame = self.current_frame.copy()

        try:
            os.makedirs(self.save_dir, exist_ok=True)
            filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            filepath = os.path.join(self.save_dir, filename)
            cv2.imwrite(filepath, frame)
            self.saved_count += 1
            return filepath
        except Exception as e:
            return None

    def get_status(self):
        """获取状态信息"""
        return {
            "connected": self.is_connected,
            "fps": round(self.fps, 1),
            "resolution": self.resolution,
            "saved_count": self.saved_count,
            "auto_save": self.auto_save,
            "auto_save_interval": self.auto_save_interval,
            "rtsp_url": self.rtsp_url,
            "save_dir": self.save_dir,
            "last_error": self.last_error,
        }

    def get_saved_images(self):
        """获取已保存图像列表"""
        if not os.path.exists(self.save_dir):
            return []
        files = []
        for f in sorted(os.listdir(self.save_dir), reverse=True):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(self.save_dir, f)
                stat = os.stat(filepath)
                files.append({
                    "name": f,
                    "size": stat.st_size,
                    "time": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                })
        return files


# 全局实例
camera = None


# ========== 路由 ==========

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HX-CCD23 相机采集</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background: #1a1a2e; color: #eee; min-height: 100vh;
        }
        .header {
            background: #16213e; padding: 12px 24px; display: flex; align-items: center;
            justify-content: space-between; border-bottom: 2px solid #0f3460;
        }
        .header h1 { font-size: 18px; color: #e94560; }
        .header .status { display: flex; align-items: center; gap: 8px; font-size: 13px; }
        .status-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
        .status-dot.ok { background: #4CAF50; box-shadow: 0 0 6px #4CAF50; }
        .status-dot.err { background: #f44336; box-shadow: 0 0 6px #f44336; }
        .main { display: flex; gap: 16px; padding: 16px; max-width: 1600px; margin: 0 auto; }
        .video-section { flex: 1; min-width: 0; }
        .video-container {
            background: #0a0a0a; border-radius: 8px; overflow: hidden;
            position: relative; border: 1px solid #333;
        }
        .video-container img {
            width: 100%; display: block; min-height: 400px; object-fit: contain;
            background: #000;
        }
        .controls {
            display: flex; align-items: center; gap: 16px; padding: 12px 0;
            flex-wrap: wrap;
        }
        .btn {
            padding: 10px 24px; border: none; border-radius: 6px; cursor: pointer;
            font-size: 14px; font-weight: 600; transition: all 0.2s;
        }
        .btn:active { transform: scale(0.96); }
        .btn-capture { background: #4CAF50; color: white; font-size: 16px; padding: 12px 32px; }
        .btn-capture:hover { background: #45a049; }
        .btn-capture.saving { background: #FFC107; color: #333; }
        .btn-small { background: #555; color: white; padding: 8px 16px; font-size: 12px; }
        .btn-small:hover { background: #666; }
        .toggle-container {
            display: flex; align-items: center; gap: 8px; background: #2a2a3e;
            padding: 8px 16px; border-radius: 6px;
        }
        .toggle {
            position: relative; width: 44px; height: 24px; cursor: pointer;
        }
        .toggle input { opacity: 0; width: 0; height: 0; }
        .toggle .slider {
            position: absolute; top: 0; left: 0; right: 0; bottom: 0;
            background: #555; border-radius: 12px; transition: 0.3s;
        }
        .toggle .slider:before {
            content: ""; position: absolute; height: 18px; width: 18px;
            left: 3px; bottom: 3px; background: white; border-radius: 50%;
            transition: 0.3s;
        }
        .toggle input:checked + .slider { background: #4CAF50; }
        .toggle input:checked + .slider:before { transform: translateX(20px); }
        .interval-input {
            width: 60px; padding: 4px 8px; background: #333; border: 1px solid #555;
            border-radius: 4px; color: white; text-align: center;
        }
        .side-panel {
            width: 300px; flex-shrink: 0;
        }
        .panel-section {
            background: #16213e; border-radius: 8px; padding: 16px; margin-bottom: 12px;
        }
        .panel-section h3 {
            font-size: 14px; color: #e94560; margin-bottom: 12px; padding-bottom: 8px;
            border-bottom: 1px solid #333;
        }
        .info-row {
            display: flex; justify-content: space-between; padding: 4px 0;
            font-size: 13px;
        }
        .info-row .label { color: #888; }
        .info-row .value { color: #4CAF50; font-weight: 600; }
        .image-list {
            max-height: 400px; overflow-y: auto;
        }
        .image-item {
            display: flex; justify-content: space-between; align-items: center;
            padding: 6px 0; border-bottom: 1px solid #1a1a2e; font-size: 12px;
        }
        .image-item .name { color: #aaa; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .image-item .time { color: #666; margin: 0 8px; white-space: nowrap; }
        .image-item .dl-btn {
            color: #4CAF50; text-decoration: none; font-size: 11px; padding: 2px 8px;
            border: 1px solid #4CAF50; border-radius: 3px; white-space: nowrap;
        }
        .image-item .dl-btn:hover { background: #4CAF50; color: white; }
        .flash { animation: flash 0.3s; }
        @keyframes flash {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        .toast {
            position: fixed; top: 20px; right: 20px; background: #4CAF50;
            color: white; padding: 12px 24px; border-radius: 6px; font-size: 14px;
            z-index: 1000; display: none; box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        .toast.show { display: block; animation: fadeIn 0.3s; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(-10px); } to { opacity: 1; } }
    </style>
</head>
<body>
    <div class="header">
        <h1>📷 HX-CCD23 实时采集</h1>
        <div class="status">
            <span class="status-dot" id="statusDot"></span>
            <span id="statusText">连接中...</span>
            <span style="margin-left:16px" id="fpsText">FPS: --</span>
            <span style="margin-left:16px" id="resText">--</span>
            <span style="margin-left:16px" id="savedText">已保存: 0</span>
        </div>
    </div>

    <div class="main">
        <div class="video-section">
            <div class="video-container">
                <img id="videoFeed" src="/video_feed" alt="视频流">
            </div>
            <div class="controls">
                <button class="btn btn-capture" id="captureBtn" onclick="doCapture()">📷 拍 摄</button>
                <div class="toggle-container">
                    <label class="toggle">
                        <input type="checkbox" id="autoSave" onchange="toggleAutoSave()">
                        <span class="slider"></span>
                    </label>
                    <span>自动保存</span>
                    <input type="number" class="interval-input" id="interval" value="1.0" 
                           min="0.1" max="60" step="0.5" onchange="updateInterval()">
                    <span style="font-size:12px;color:#888">秒/帧</span>
                </div>
            </div>
        </div>

        <div class="side-panel">
            <div class="panel-section">
                <h3>📊 状态信息</h3>
                <div class="info-row"><span class="label">RTSP地址</span></div>
                <div class="info-row"><span class="value" id="rtspUrl" style="font-size:11px;word-break:break-all"></span></div>
                <div class="info-row"><span class="label">分辨率</span><span class="value" id="infoRes">--</span></div>
                <div class="info-row"><span class="label">帧率</span><span class="value" id="infoFps">--</span></div>
                <div class="info-row"><span class="label">连接状态</span><span class="value" id="infoConn">--</span></div>
                <div class="info-row"><span class="label">自动保存</span><span class="value" id="infoAuto">关闭</span></div>
                <div class="info-row"><span class="label">已采集</span><span class="value" id="infoSaved">0 张</span></div>
            </div>

            <div class="panel-section">
                <h3>📁 已采集图像</h3>
                <div class="image-list" id="imageList">加载中...</div>
            </div>
        </div>
    </div>

    <div class="toast" id="toast"></div>

    <script>
        // 定时刷新状态
        setInterval(updateStatus, 1000);
        setInterval(updateImageList, 5000);
        updateStatus();
        updateImageList();

        function showToast(msg, duration=2000) {
            const t = document.getElementById('toast');
            t.textContent = msg;
            t.classList.add('show');
            setTimeout(() => t.classList.remove('show'), duration);
        }

        async function updateStatus() {
            try {
                const resp = await fetch('/api/status');
                const data = await resp.json();
                
                const dot = document.getElementById('statusDot');
                const txt = document.getElementById('statusText');
                dot.className = 'status-dot ' + (data.connected ? 'ok' : 'err');
                txt.textContent = data.connected ? '已连接' : '未连接';
                
                document.getElementById('fpsText').textContent = 'FPS: ' + data.fps;
                document.getElementById('resText').textContent = data.resolution || '--';
                document.getElementById('savedText').textContent = '已保存: ' + data.saved_count;
                
                document.getElementById('rtspUrl').textContent = data.rtsp_url;
                document.getElementById('infoRes').textContent = data.resolution || '--';
                document.getElementById('infoFps').textContent = data.fps + ' fps';
                document.getElementById('infoConn').textContent = data.connected ? '✅ 正常' : '❌ 断开';
                document.getElementById('infoAuto').textContent = data.auto_save ? '✅ 开启 (' + data.auto_save_interval + 's)' : '❌ 关闭';
                document.getElementById('infoSaved').textContent = data.saved_count + ' 张';
            } catch(e) {}
        }

        async function updateImageList() {
            try {
                const resp = await fetch('/api/images');
                const data = await resp.json();
                const list = document.getElementById('imageList');
                if (data.length === 0) {
                    list.innerHTML = '<div style="color:#666;font-size:12px">暂无图像</div>';
                    return;
                }
                list.innerHTML = data.slice(0, 50).map(img => 
                    '<div class="image-item">' +
                    '<span class="name" title="' + img.name + '">' + img.name + '</span>' +
                    '<span class="time">' + img.time + '</span>' +
                    '<a class="dl-btn" href="/api/download/' + img.name + '" target="_blank">下载</a>' +
                    '</div>'
                ).join('');
            } catch(e) {}
        }

        async function doCapture() {
            const btn = document.getElementById('captureBtn');
            btn.classList.add('saving');
            btn.textContent = '📷 保存中...';
            try {
                const resp = await fetch('/api/capture', {method: 'POST'});
                const data = await resp.json();
                if (data.success) {
                    showToast('✅ 已保存: ' + data.filename);
                } else {
                    showToast('❌ 保存失败', 3000);
                }
            } catch(e) {
                showToast('❌ 请求失败', 3000);
            }
            setTimeout(() => {
                btn.classList.remove('saving');
                btn.textContent = '📷 拍 摄';
            }, 500);
            updateImageList();
        }

        async function toggleAutoSave() {
            const on = document.getElementById('autoSave').checked;
            await fetch('/api/auto_save', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({enabled: on})
            });
            showToast(on ? '✅ 自动保存已开启' : '⏹️ 自动保存已关闭');
        }

        async function updateInterval() {
            const val = parseFloat(document.getElementById('interval').value);
            await fetch('/api/auto_save', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({interval: val})
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/video_feed')
def video_feed():
    """MJPEG 视频流"""
    def generate():
        while camera.running:
            frame = camera.get_frame_jpeg()
            if frame is None:
                # 返回占位图
                time.sleep(0.1)
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)  # ~30fps 上限

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/status')
def api_status():
    return jsonify(camera.get_status())


@app.route('/api/capture', methods=['POST'])
def api_capture():
    filepath = camera.save_frame()
    if filepath:
        return jsonify({"success": True, "filename": os.path.basename(filepath), "path": filepath})
    return jsonify({"success": False, "error": "无法保存帧"})


@app.route('/api/auto_save', methods=['POST'])
def api_auto_save():
    data = request.json
    if 'enabled' in data:
        camera.auto_save = data['enabled']
    if 'interval' in data:
        camera.auto_save_interval = max(0.1, float(data['interval']))
    return jsonify({"success": True, "auto_save": camera.auto_save, "interval": camera.auto_save_interval})


@app.route('/api/images')
def api_images():
    return jsonify(camera.get_saved_images())


@app.route('/api/download/<filename>')
def api_download(filename):
    filepath = os.path.join(camera.save_dir, filename)
    if os.path.exists(filepath):
        return send_file(filepath, mimetype='image/jpeg', as_attachment=True)
    return jsonify({"error": "文件不存在"}), 404


def main():
    parser = argparse.ArgumentParser(description="HX-CCD23 相机实时采集 Web GUI")
    parser.add_argument("--url", default=DEFAULT_RTSP_URL, help=f"RTSP 流地址")
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR, help="保存目录")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"HTTP端口 (默认: {DEFAULT_PORT})")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址 (默认: 0.0.0.0)")
    args = parser.parse_args()

    global camera
    camera = CameraState(args.url, args.save_dir)

    print("=" * 60)
    print("  HX-CCD23 相机实时采集 Web GUI")
    print(f"  RTSP: {args.url}")
    print(f"  保存: {args.save_dir}")
    print(f"  地址: http://localhost:{args.port}")
    print(f"  局域网: http://<本机IP>:{args.port}")
    print("=" * 60)
    print("\n  打开浏览器访问以上地址即可\n")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
