QT       += core gui network sql

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17
TARGET = CameraViewer_orbbec

# 启用 OpenCV CUDA 加速（背景分割、特征比对等）
DEFINES += USE_OPENCV_CUDA

# =========================================================
# OrbbecSDK v2（Gemini 335 深度相机）
# =========================================================
ORBBEC_SDK_DIR = $$PWD/third_party/OrbbecSDK_v2.10.6_202609291004_575347c_linux_x86_64
INCLUDEPATH += $$ORBBEC_SDK_DIR/include
LIBS += -L$$ORBBEC_SDK_DIR/lib -lOrbbecSDK
# 运行时库搜索路径（免设 LD_LIBRARY_PATH）
QMAKE_RPATHDIR += $$ORBBEC_SDK_DIR/lib
QMAKE_RPATHDIR += $$ORBBEC_SDK_DIR/lib/extensions/filters
QMAKE_RPATHDIR += $$ORBBEC_SDK_DIR/lib/extensions/depthengine

SOURCES += \
    src/main.cpp \
    src/cameracontroller.cpp \
    src/orbbecsource.cpp \
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
    src/orbbecsource.h \
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
    -lopencv_video \
    -lopencv_cudaarithm \
    -lopencv_cudaimgproc \
    -lopencv_cudafilters \
    -lopencv_cudawarping

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
#  - 检测：head 数据集训练的 YOLO11n 头部检测（单类 head）
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
