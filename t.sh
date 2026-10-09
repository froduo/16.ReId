#!/bin/bash
cd /media/froduo/4ECEA802CEA7E107/AA_project/16.reId
mkdir -p captured_images
python3 -c "import cv2; cap = cv2.VideoCapture('rtsp://192.168.99.10:8554/live/camera', cv2.CAP_FFMPEG); cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000); print('OK:', cap.isOpened()); ret, frame = cap.read(); print('ret:', ret); print('shape:', frame.shape if ret else 'N/A'); cv2.imwrite('captured_images/test.jpg', frame) if ret else None; cap.release()"
ls -la captured_images/test.jpg
