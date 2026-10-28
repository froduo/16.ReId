#!/bin/sh
# 编译命令行离线复核工具：复用工程内同一套「检测 → 跟踪 → 穿越判定」逻辑
# 用法: sh build_clip_check.sh
# 产物: ./offline_clip_check
set -e

QT=${QT_DIR:-/home/froduo/Qt/6.2.4/gcc_64}
INC="-I CameraViewer/src -I $QT/include -I $QT/include/QtCore -I $QT/include/QtGui"

echo "[1/3] moc..."
$QT/libexec/moc CameraViewer/src/zonemonitor.h  -o /tmp/moc_zonemonitor.cpp  $INC
$QT/libexec/moc CameraViewer/src/regionmanager.h -o /tmp/moc_regionmanager.cpp $INC

echo "[2/3] compile & link..."
g++ -std=c++17 -O2 -fPIC offline_clip_check.cpp \
  CameraViewer/src/persondetector.cpp CameraViewer/src/trt_yolo.cpp \
  CameraViewer/src/persontracker.cpp CameraViewer/src/reidextractor.cpp \
  CameraViewer/src/reiddeep.cpp \
  CameraViewer/src/backgroundmodel.cpp CameraViewer/src/regionmanager.cpp \
  CameraViewer/src/zonemonitor.cpp \
  /tmp/moc_zonemonitor.cpp /tmp/moc_regionmanager.cpp \
  $INC -I /usr/local/include/opencv4 \
  -I /usr/local/TensorRT-10.3/include -I /usr/local/cuda/include \
  -L /usr/local/TensorRT-10.3/targets/x86_64-linux-gnu/lib \
  -lnvinfer -lnvinfer_plugin -lnvonnxparser \
  -L /usr/local/cuda/lib64 -lcudart \
  -L $QT/lib -lQt6Core -lQt6Gui \
  -lopencv_core -lopencv_imgproc -lopencv_dnn -lopencv_videoio \
  -lopencv_highgui -lopencv_imgcodecs -lopencv_video \
  -lpthread -o offline_clip_check

echo "[3/3] done -> ./offline_clip_check"
echo
echo "用法:"
echo "  ./offline_clip_check <视频文件|目录> [step=2] [模型路径] [regions.json]"
echo "示例:"
echo "  ./offline_clip_check build-CameraViewer-main-Desktop-Debug/save_images/clips 3"
