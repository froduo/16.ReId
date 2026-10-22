# ============================================================================
# passenger_clip_tool.pro —— 离线批处理工具（乘客上下车片段裁剪）
#
# 复用主程序的检测/跟踪/ReID 源码（../src），生成独立控制台可执行文件，
# 不影响主程序 CameraViewer_orbbec 的构建。
#
# 构建：
#   mkdir -p build-passenger-tool && cd build-passenger-tool
#   /home/froduo/Qt/6.2.4/gcc_64/bin/qmake ../tools/passenger_clip_tool.pro
#   make -j$(nproc)
# ============================================================================

QT       += core
QT       -= gui
CONFIG   += console c++17
CONFIG   -= app_bundle
TEMPLATE  = app
TARGET    = passenger_clip_tool

# 与主程序一致：启用 OpenCV CUDA（深度 ReID 后端）
DEFINES  += USE_OPENCV_CUDA

INCLUDEPATH += ../src \
    /usr/local/include/opencv4 \
    /usr/local/cuda/include \
    /usr/local/TensorRT-10.3/include

SOURCES += \
    passenger_clip_tool.cpp \
    ../src/persondetector.cpp \
    ../src/trt_yolo.cpp \
    ../src/persontracker.cpp \
    ../src/reidextractor.cpp \
    ../src/reiddeep.cpp \
    ../src/backgroundmodel.cpp

HEADERS += \
    ../src/persondetector.h \
    ../src/trt_yolo.h \
    ../src/persontracker.h \
    ../src/reidextractor.h \
    ../src/reiddeep.h \
    ../src/backgroundmodel.h \
    ../src/logger.h \
    ../src/hungarian.h

# OpenCV
LIBS += -L/usr/local/lib \
    -lopencv_core \
    -lopencv_imgproc \
    -lopencv_highgui \
    -lopencv_imgcodecs \
    -lopencv_videoio \
    -lopencv_dnn \
    -lopencv_video \
    -lopencv_cudaarithm \
    -lopencv_cudaimgproc \
    -lopencv_cudafilters \
    -lopencv_cudawarping

# TensorRT & CUDA
LIBS += \
    -L/usr/local/cuda/lib64 \
    -L/usr/local/TensorRT-10.3/targets/x86_64-linux-gnu/lib \
    -lnvinfer \
    -lnvinfer_plugin \
    -lnvonnxparser \
    -lcudart
