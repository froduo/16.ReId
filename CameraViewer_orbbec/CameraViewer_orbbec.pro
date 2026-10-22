QT       += core gui network sql

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17
TARGET = CameraViewer_orbbec

# 启用 OpenCV CUDA 加速（背景分割等）；若本机 OpenCV 未编译 CUDA 可注释掉本行
#DEFINES += USE_OPENCV_CUDA  # 88deploy: OpenCV CUDA only sm_61, RTX2060 sm_75 unsupported

SOURCES += \
    src/main.cpp \
    src/cameracontroller.cpp \
    src/frameslot.cpp \
    src/grabthread.cpp \
    src/depthgrabthread.cpp \
    src/detectthread.cpp \
    src/imageview.cpp \
    src/overlaytext.cpp \
    src/mainwindow.cpp \
    src/saveimageworker.cpp \
    src/persondetector.cpp \
    src/persontracker.cpp \
    src/reidextractor.cpp \
    src/reiddeep.cpp \
    src/trt_reid.cpp \
    src/backgroundmodel.cpp \
    src/regionmanager.cpp \
    src/trt_yolo.cpp \
    src/offlinedetectform.cpp \
    src/settingform.cpp \
    src/visitorlog.cpp \
    src/passengerbook.cpp \
    src/identitymatchworker.cpp \
    src/zonemonitor.cpp \
    src/personfolder.cpp \
    src/foldermatchworker.cpp \
    src/statuswebserver.cpp \
    src/tripdatastore.cpp

HEADERS += \
    src/version.h \
    src/app_config.h \
    src/cameracontroller.h \
    src/frameslot.h \
    src/grabthread.h \
    src/depthgrabthread.h \
    src/detectthread.h \
    src/imageview.h \
    src/overlaytext.h \
    src/logger.h \
    src/mainwindow.h \
    src/saveimageworker.h \
    src/persondetector.h \
    src/persontracker.h \
    src/reidextractor.h \
    src/reiddeep.h \
    src/trt_reid.h \
    src/backgroundmodel.h \
    src/regionmanager.h \
    src/trt_yolo.h \
    src/offlinedetectform.h \
    src/settingform.h \
    src/visitorlog.h \
    src/passengerbook.h \
    src/identitymatchworker.h \
    src/zonemonitor.h \
    src/personfolder.h \
    src/foldermatchworker.h \
    src/statuswebserver.h \
    src/tripdatastore.h \
    src/hungarian.h

FORMS += \
    src/mainwindow.ui \
    src/settingform.ui

INCLUDEPATH += src

# =========================================================
# OpenCV
# =========================================================
INCLUDEPATH += /usr/local/include/opencv4
LIBS += \
    -lopencv_core \
    -lopencv_imgproc \
    -lopencv_highgui \
    -lopencv_imgcodecs \
    -lopencv_videoio \
    -lopencv_dnn \
    -lopencv_video

# =========================================================
# TensorRT & CUDA
# =========================================================
INCLUDEPATH += /usr/local/cuda/include \
    /usr/local/TensorRT-10.3/include

LIBS += \
    -L/usr/local/cuda/lib64 \
    -L/usr/local/TensorRT-10.3/targets/x86_64-linux-gnu/lib \
    -lnvinfer \
    -lnvinfer_plugin \
    -lnvonnxparser \
    -lcudart

# =========================================================
# 模型文件复制到输出目录
#  - 检测：passenger_yolov8n 人体检测（单类 person，与 CameraViewer 一致）
#  - ReID：现场数据微调的 OSNet x1.0（4D 输出，DeepReID 兼容）
# =========================================================
MODEL_DEST_DIR = $$OUT_PWD/model
MODEL_SRC_DIR  = $$PWD/model

# 用独立脚本复制模型（for() 循环内的 += 在 qmake 中不生效，故走脚本）
copy_models.target   = copy_models
copy_models.commands = $$shell_path($$PWD/copy_models.sh) $$shell_path($$MODEL_SRC_DIR) $$shell_path($$MODEL_DEST_DIR)
copy_models.depends  = FORCE
QMAKE_EXTRA_TARGETS += copy_models

# ===== 版本号自动生成（每次构建自增）=====
version_gen.target = version_gen
version_gen.commands = $$shell_path($$PWD/gen_version.sh)
version_gen.depends = FORCE
QMAKE_EXTRA_TARGETS += version_gen
first.depends = version_gen copy_models $(first)
export(first.depends)
export(version_gen.commands)
QMAKE_EXTRA_TARGETS += first
