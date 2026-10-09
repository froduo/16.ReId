#!/usr/bin/env python3
"""
HX-CCD23 双目客流统计终端 - 拉流显示与图像采集脚本
=============================================
设备: HX-CCD23 (华芯智云 foorir)
IP: 192.168.99.10
RTSP端口: 8554 (ZLMediaKit)

已确认的流地址:
  RTSP: rtsp://192.168.99.10:8554/live/camera (H.264, 1920x1080, 25fps)
  WebRTC (浏览器): http://192.168.99.10/media/webrtc?app=live&stream=camera&type=play

RTMP推流配置 (用户提供的):
  服务器: passengerflowpush.humanplustech.com
  推流地址: live
  密钥: EXMdlOnbfhKmwfXW

用法:
  python3 camera_stream.py                          # 探测并显示视频（需要显示器/X11）
  python3 camera_stream.py --capture                # 采集单帧图像
  python3 camera_stream.py --headless --count 10    # 无头模式采集10帧
  python3 camera_stream.py --headless --interval 1  # 每秒采集1帧（持续）
  python3 camera_stream.py --rtmp                   # 使用RTMP拉流
  python3 camera_stream.py --url <URL>              # 指定流地址
  python3 camera_stream.py --ffmpeg                 # 使用ffplay直接播放

已验证信息 (2026-09-11):
  相机端口: 80(HTTP), 8554(RTSP), 8000
  RTSP流: rtsp://192.168.99.10:8554/live/camera
  分辨率: 1920x1080, 编码: H.264
  Web前端: Vue.js SPA (使用ZLMediaKit WebRTC)
  HTTP截图API: /ImageAndVideo/CapturePictureBase64 (POST)
  RTMP拉流API: /ImageAndVideo/RtmpPullGet
"""

import cv2
import os
import sys
import time
import argparse
import subprocess
import signal
from datetime import datetime

# ========== 配置 ==========
CAMERA_IP = "192.168.99.10"
CAMERA_RTSP_PORT = 8554
CAMERA_USER = "admin"
CAMERA_PASS = "hf123456"

# 已确认的 RTSP 流地址
RTSP_URL = f"rtsp://{CAMERA_IP}:{CAMERA_RTSP_PORT}/live/camera"

# RTMP 推流配置（用户提供的）
RTMP_SERVER = "passengerflowpush.humanplustech.com"
RTMP_PATH = "live"
RTMP_KEY = "EXMdlOnbfhKmwfXW"

# 保存目录
SAVE_DIR = "captured_images"

# 全局退出标志
running = True


def signal_handler(sig, frame):
    """Ctrl+C 信号处理"""
    global running
    print("\n[信号] 收到中断信号，正在退出...")
    running = False


signal.signal(signal.SIGINT, signal_handler)


def check_camera_reachable(ip, timeout=3):
    """检查相机是否可达"""
    print(f"[检查] 测试相机 {ip} 是否可达...")
    try:
        result = subprocess.run(
            ["ping", "-c", "2", "-W", str(timeout), ip],
            capture_output=True, text=True, timeout=timeout + 2
        )
        if result.returncode == 0:
            print(f"  ✓ 相机 {ip} 可达")
            return True
        else:
            print(f"  ✗ 相机 {ip} 不可达")
            return False
    except Exception as e:
        print(f"  ✗ ping 失败: {e}")
        return False


def check_port(ip, port, timeout=2):
    """检查端口是否开放"""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    result = s.connect_ex((ip, port))
    s.close()
    return result == 0


def probe_rtsp_url(url, timeout=8):
    """使用 ffprobe 探测 RTSP 流"""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-rtsp_transport", "tcp",
             "-analyzeduration", "3000000", "-probesize", "3000000",
             "-i", url,
             "-show_entries", "stream=codec_name,width,height,r_frame_rate",
             "-of", "json"],
            capture_output=True, text=True, timeout=timeout
        )
        if result.returncode == 0 and "codec_name" in result.stdout:
            return True, result.stdout
        return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "超时"
    except Exception as e:
        return False, str(e)


def find_working_stream():
    """探测可用的视频流地址"""
    print("\n[探测] 正在扫描可用的视频流地址...")

    all_urls = [
        RTSP_URL,
        f"rtsp://{CAMERA_USER}:{CAMERA_PASS}@{CAMERA_IP}:{CAMERA_RTSP_PORT}/live/camera",
        f"rtsp://{CAMERA_IP}:{CAMERA_RTSP_PORT}/live/main",
        f"rtsp://{CAMERA_IP}:{CAMERA_RTSP_PORT}/live/sub",
        f"rtsp://{CAMERA_IP}:{CAMERA_RTSP_PORT}/stream0",
    ]

    if check_port(CAMERA_IP, CAMERA_RTSP_PORT):
        print(f"  ✓ RTSP 端口 {CAMERA_RTSP_PORT} 已开放")
        for url in all_urls:
            print(f"  尝试: {url}")
            ok, info = probe_rtsp_url(url)
            if ok:
                print(f"  ✓ 成功!")
                return "rtsp", url
            else:
                print(f"    ✗")
    else:
        print(f"  ✗ RTSP 端口 {CAMERA_RTSP_PORT} 未开放")

    return None, None


def open_stream(url):
    """打开 RTSP 视频流"""
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
    return cap


def save_frame(frame, save_dir, prefix="capture"):
    """保存帧为图像文件"""
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    filepath = os.path.join(save_dir, filename)
    cv2.imwrite(filepath, frame)
    return filepath


def display_stream(stream_url, save_dir=None):
    """显示视频流（需要显示器/X11）"""
    global running

    if save_dir is None:
        save_dir = SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n[拉流] 正在打开流: {stream_url}")
    cap = open_stream(stream_url)

    if not cap.isOpened():
        print("  ✗ 无法打开视频流")
        return False

    ret, test_frame = cap.read()
    if not ret:
        print("  ✗ 无法从流中读取帧")
        cap.release()
        return False

    print(f"  ✓ 视频流已打开，分辨率: {test_frame.shape[1]}x{test_frame.shape[0]}")
    print("\n" + "=" * 60)
    print("  操作说明:")
    print("    按 's' 或 空格  - 保存当前帧（采集图像）")
    print("    按 'r'         - 重新连接流")
    print("    按 'f'         - 切换全屏")
    print("    按 'p'         - 暂停/恢复")
    print("    按 'q' 或 ESC   - 退出")
    print("=" * 60 + "\n")

    window_name = f"HX-CCD23 Camera ({CAMERA_IP})"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    frame_count = 0
    fps_start = time.time()
    fps = 0
    fullscreen = False
    paused = False
    current_frame = None
    saved_count = 0

    while running:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("  ⚠ 帧读取失败，尝试重新连接...")
                cap.release()
                time.sleep(2)
                cap = open_stream(stream_url)
                if not cap.isOpened():
                    print("  ✗ 重连失败")
                    break
                continue

            current_frame = frame.copy()
            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()
        else:
            frame = current_frame
            if frame is None:
                continue

        # 信息叠加
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        info1 = f"FPS: {fps:.1f} | {w}x{h} | RTSP | {CAMERA_IP}"
        info2 = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  Saved: {saved_count} {'[PAUSED]' if paused else ''}"
        cv2.putText(frame, info1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(frame, info2, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord('s') or key == 32:
            if current_frame is not None:
                filepath = save_frame(current_frame, save_dir)
                saved_count += 1
                print(f"  [采集 #{saved_count}] {filepath}")
        elif key == ord('r'):
            print("  [重连] 正在重新连接...")
            cap.release()
            time.sleep(1)
            cap = open_stream(stream_url)
        elif key == ord('f'):
            fullscreen = not fullscreen
            prop = cv2.WND_PROP_FULLSCREEN if fullscreen else cv2.WND_PROP_FULLSCREEN
            val = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(window_name, prop, val)
        elif key == ord('p'):
            paused = not paused
            print(f"  {'[暂停]' if paused else '[恢复]'}")

    cap.release()
    cv2.destroyAllWindows()
    return True


def headless_capture(stream_url, save_dir=None, count=1, interval=0, duration=0):
    """无头模式采集（不需要显示器）
    
    Args:
        stream_url: RTSP 流地址
        save_dir: 保存目录
        count: 采集帧数 (0=无限)
        interval: 采集间隔(秒)
        duration: 持续时间(秒), 0=持续到 count 达到或 Ctrl+C
    """
    global running

    if save_dir is None:
        save_dir = SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n[无头采集] 正在打开流: {stream_url}")
    cap = open_stream(stream_url)

    if not cap.isOpened():
        print("  ✗ 无法打开视频流")
        return False

    ret, test_frame = cap.read()
    if not ret:
        print("  ✗ 无法读取帧")
        cap.release()
        return False

    print(f"  ✓ 流已打开: {test_frame.shape[1]}x{test_frame.shape[0]}")
    print(f"  采集模式: {'无限' if count == 0 else count}帧, 间隔{interval}秒")
    print(f"  按 Ctrl+C 停止\n")

    saved_count = 0
    start_time = time.time()

    while running:
        # 检查持续时间限制
        if duration > 0 and (time.time() - start_time) >= duration:
            print(f"\n[完成] 已达到持续时间限制 ({duration}秒)")
            break

        # 检查数量限制
        if count > 0 and saved_count >= count:
            print(f"\n[完成] 已采集 {count} 帧")
            break

        ret, frame = cap.read()
        if not ret:
            print("  ⚠ 读取失败，等待重连...")
            cap.release()
            time.sleep(2)
            cap = open_stream(stream_url)
            if not cap.isOpened():
                print("  ✗ 重连失败")
                break
            continue

        filepath = save_frame(frame, save_dir)
        saved_count += 1
        print(f"  [{saved_count}] {filepath} ({frame.shape[1]}x{frame.shape[0]})")

        if interval > 0:
            # 等待间隔时间，但允许中断
            wait_end = time.time() + interval
            while running and time.time() < wait_end:
                time.sleep(0.1)

    cap.release()
    total_time = time.time() - start_time
    print(f"\n[统计] 采集 {saved_count} 帧, 耗时 {total_time:.1f}秒, 保存目录: {save_dir}")
    return True


def capture_single_frame(stream_url, save_dir=None):
    """采集单帧图像"""
    if save_dir is None:
        save_dir = SAVE_DIR

    print(f"\n[采集] 正在采集单帧...")
    cap = open_stream(stream_url)

    if not cap.isOpened():
        print("  ✗ 无法打开流")
        return None

    # 跳过前几帧（可能花屏）
    for _ in range(5):
        ret, frame = cap.read()
    cap.release()

    if not ret:
        print("  ✗ 无法读取帧")
        return None

    filepath = save_frame(frame, save_dir)
    print(f"  ✓ 已保存: {filepath} ({frame.shape[1]}x{frame.shape[0]})")
    return filepath


def play_with_ffplay(stream_url):
    """使用 ffplay 直接播放（低延迟）"""
    print(f"\n[ffplay] 正在播放: {stream_url}")
    print("  按 'q' 退出播放\n")
    try:
        subprocess.run([
            "ffplay", "-v", "error",
            "-rtsp_transport", "tcp",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-framedrop",
            "-i", stream_url
        ])
    except KeyboardInterrupt:
        pass


def play_with_ffmpeg_record(stream_url, output_file=None, duration=0):
    """使用 ffmpeg 录制流"""
    if output_file is None:
        output_file = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

    print(f"\n[录制] 正在录制: {stream_url}")
    print(f"  输出: {output_file}")
    print("  按 Ctrl+C 停止录制\n")

    cmd = [
        "ffmpeg", "-y",
        "-rtsp_transport", "tcp",
        "-i", stream_url,
        "-c:v", "copy",  # 不重新编码
        "-an",  # 无音频
    ]
    if duration > 0:
        cmd.extend(["-t", str(duration)])
    cmd.append(output_file)

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print(f"\n[录制] 已保存: {output_file}")


def main():
    global CAMERA_IP
    parser = argparse.ArgumentParser(
        description="HX-CCD23 相机拉流显示与图像采集工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 camera_stream.py                          # 显示视频流
  python3 camera_stream.py --capture                # 采集一帧
  python3 camera_stream.py --headless               # 无头采集一帧
  python3 camera_stream.py --headless --count 0 --interval 5  # 每5秒采集一帧
  python3 camera_stream.py --headless --count 100   # 连续采集100帧
  python3 camera_stream.py --ffmpeg                 # ffplay 播放
  python3 camera_stream.py --record                 # ffmpeg 录制
        """
    )
    parser.add_argument("--ip", default=CAMERA_IP, help=f"相机IP (默认: {CAMERA_IP})")
    parser.add_argument("--url", default=None, help="指定流URL (默认自动探测)")
    parser.add_argument("--capture", action="store_true", help="采集单帧")
    parser.add_argument("--headless", action="store_true", help="无头模式（不需要显示器）")
    parser.add_argument("--count", type=int, default=1, help="采集帧数 (0=无限, 默认: 1)")
    parser.add_argument("--interval", type=float, default=0, help="采集间隔秒数 (默认: 0)")
    parser.add_argument("--duration", type=float, default=0, help="持续采集时间秒 (默认: 0=不限)")
    parser.add_argument("--save-dir", default=SAVE_DIR, help=f"保存目录 (默认: {SAVE_DIR})")
    parser.add_argument("--ffmpeg", action="store_true", help="使用 ffplay 播放")
    parser.add_argument("--record", action="store_true", help="使用 ffmpeg 录制")
    parser.add_argument("--record-file", default=None, help="录制文件名")
    parser.add_argument("--record-duration", type=int, default=0, help="录制时长(秒)")
    parser.add_argument("--rtmp", action="store_true", help="使用RTMP拉流")
    parser.add_argument("--rtmp-url", default=None, help="自定义RTMP拉流地址")

    args = parser.parse_args()
    CAMERA_IP = args.ip

    print("=" * 60)
    print("  HX-CCD23 双目客流终端 - 视频流采集工具")
    print(f"  相机IP: {CAMERA_IP}")
    print(f"  保存目录: {args.save_dir}")
    print("=" * 60)

    # 确定流地址
    stream_url = args.url
    if args.rtmp or args.rtmp_url:
        if args.rtmp_url:
            stream_url = args.rtmp_url
        else:
            stream_url = f"rtmp://{RTMP_SERVER}/{RTMP_PATH}/{RTMP_KEY}"
        print(f"\n[RTMP] {stream_url}")
    elif stream_url is None:
        if not check_camera_reachable(CAMERA_IP):
            print(f"\n[警告] 相机 {CAMERA_IP} 不可达")
            print("  请检查网络连接或指定正确的IP")
            return 1

        stream_type, stream_url = find_working_stream()
        if stream_url is None:
            print("\n[失败] 未找到可用流，请手动指定:")
            print(f"  python3 camera_stream.py --url rtsp://{CAMERA_IP}:8554/live/camera")
            return 1

    print(f"\n[确定] 流地址: {stream_url}")

    # 执行模式
    if args.ffmpeg:
        play_with_ffplay(stream_url)
    elif args.record:
        play_with_ffmpeg_record(stream_url, args.record_file, args.record_duration)
    elif args.capture:
        capture_single_frame(stream_url, args.save_dir)
    elif args.headless:
        headless_capture(stream_url, args.save_dir, args.count, args.interval, args.duration)
    else:
        display_stream(stream_url, args.save_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
