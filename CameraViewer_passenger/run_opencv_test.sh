#!/bin/bash
cd /media/froduo/4ECEA802CEA7E107/AA_project/16.reId
mkdir -p captured_images

echo "=== Testing OpenCV RTSP capture ==="
python3 -c "
import cv2, os
os.makedirs('captured_images', exist_ok=True)
cap = cv2.VideoCapture('rtsp://192.168.99.10:8554/live/camera', cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
print('Opened:', cap.isOpened())
ret, frame = cap.read()
print('Read:', ret)
if ret:
    print('Shape:', frame.shape)
    cv2.imwrite('captured_images/test_capture.jpg', frame)
    print('Saved to captured_images/test_capture.jpg')
cap.release()
"
echo "=== Done ==="
ls -la captured_images/test_capture.jpg 2>/dev/null
