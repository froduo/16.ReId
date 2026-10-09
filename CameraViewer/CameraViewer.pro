QT       += core gui network sql

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# 启用 OpenCV CUDA 加速（背景分割等）；若本机 OpenCV 未编译 CUDA 可注释掉本行
DEFINES += USE_OPENCV_CUDA

SOURCES += \
    src/main.cpp \
    src/cameracontroller.cpp \
    src/grabthread.cpp \
    src/imageview.cpp \
    src/overlaytext.cpp \
    src/mainwindow.cpp \
    src/saveimageworker.cpp \
    src/persondetector.cpp \
    src/persontracker.cpp \
    src/reidextractor.cpp \
    src/reiddeep.cpp \
    src/backgroundmodel.cpp \
    src/regionmanager.cpp \
    src/trt_yolo.cpp \
    src/offlinedetectform.cpp \
    src/settingform.cpp \
    src/visitorlog.cpp \
    src/passengerbook.cpp \
    src/identitymatchworker.cpp \
    src/zonemonitor.cpp \
    src/tripdatastore.cpp

HEADERS += \
    src/version.h \
    src/app_config.h \
    src/cameracontroller.h \
    src/grabthread.h \
    src/imageview.h \
    src/overlaytext.h \
    src/logger.h \
    src/mainwindow.h \
    src/saveimageworker.h \
    src/persondetector.h \
    src/persontracker.h \
    src/reidextractor.h \
    src/reiddeep.h \
    src/backgroundmodel.h \
    src/regionmanager.h \
    src/trt_yolo.h \
    src/offlinedetectform.h \
    src/settingform.h \
    src/visitorlog.h \
    src/passengerbook.h \
    src/identitymatchworker.h \
    src/zonemonitor.h \
    src/tripdatastore.h \
    src/hungarian.h

FORMS += \
    src/mainwindow.ui \
    src/settingform.ui

# 头文件搜索路径
INCLUDEPATH += src

# OpenCV
INCLUDEPATH += /usr/local/include/opencv4

LIBS += \
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
INCLUDEPATH += /usr/local/cuda/include \
    /usr/local/TensorRT-10.3/include

LIBS += \
    -L/usr/local/cuda/lib64 \
    -L/usr/local/TensorRT-10.3/targets/x86_64-linux-gnu/lib \
    -lnvinfer \
    -lnvinfer_plugin \
    -lnvonnxparser \
    -lcudart

# 模型文件复制（工程内路径须与系统配置(config.ini)的模型路径保持一致）：
#   - [Inference]/ModelPath    = ./model/yolo26n_trt10_x86.engine
#   - [ReID]/DeepModelPath     = ./model/osnet_x1_0_head_finetuned.onnx
ENGINE_SOURCE = $$PWD/model/yolo26n_trt10_x86.engine
ENGINE_DEST_DIR = $$OUT_PWD/model

QMAKE_POST_LINK += mkdir -p $$ENGINE_DEST_DIR && test "$$OUT_PWD" != "$$PWD" && cp -f $$ENGINE_SOURCE $$ENGINE_DEST_DIR || true

# 深度 ReID(OSNet) ONNX 模型一并复制到输出目录（与 [ReID]/DeepModelPath 一致）
OSNET_SOURCE = $$PWD/model/osnet_x1_0_head_finetuned.onnx
QMAKE_POST_LINK += && test "$$OUT_PWD" != "$$PWD" && test -f $$OSNET_SOURCE && cp -f $$OSNET_SOURCE $$ENGINE_DEST_DIR || true

# ===== 版本号自动生成（每次构建自增）=====
# 构建前先运行 gen_version.sh 重新生成 src/version.h，
# 这样每一次「编译/更新」都会自动产生新的版本号。
# 注意：version_gen 必须排在 $(first) 之前，保证先生成版本号、再编译。
version_gen.target = version_gen
version_gen.commands = $$shell_path($$PWD/gen_version.sh)
version_gen.depends = FORCE
QMAKE_EXTRA_TARGETS += version_gen
first.depends = version_gen $(first)
export(first.depends)
export(version_gen.commands)
QMAKE_EXTRA_TARGETS += first
