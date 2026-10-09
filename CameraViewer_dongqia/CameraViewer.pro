QT       += core gui network

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

SOURCES += \
    src/main.cpp \
    src/cameracontroller.cpp \
    src/grabthread.cpp \
    src/imageview.cpp \
    src/mainwindow.cpp \
    src/saveimageworker.cpp \
    src/persondetector.cpp \
    src/persontracker.cpp \
    src/reidextractor.cpp \
    src/regionmanager.cpp \
    src/trt_yolo.cpp \
    src/offlinedetectform.cpp \
    src/settingform.cpp \
    src/visitorlog.cpp \
    src/zonemonitor.cpp \
    src/httpuplink.cpp \
    src/flowaggregator.cpp \
    src/visitorregistry.cpp

HEADERS += \
    src/app_config.h \
    src/cameracontroller.h \
    src/grabthread.h \
    src/imageview.h \
    src/logger.h \
    src/mainwindow.h \
    src/saveimageworker.h \
    src/persondetector.h \
    src/persontracker.h \
    src/reidextractor.h \
    src/regionmanager.h \
    src/trt_yolo.h \
    src/offlinedetectform.h \
    src/settingform.h \
    src/visitorlog.h \
    src/zonemonitor.h \
    src/dongqia_protocol.h \
    src/httpuplink.h \
    src/flowaggregator.h \
    src/visitorregistry.h

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
    -lopencv_video

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

# 模型文件复制（引擎文件和类别文件）
ENGINE_SOURCE = $$PWD/model/yolo26n_trt10_x86.engine
ENGINE_DEST_DIR = $$OUT_PWD/model

QMAKE_POST_LINK += mkdir -p $$ENGINE_DEST_DIR && test "$$OUT_PWD" != "$$PWD" && cp -f $$ENGINE_SOURCE $$ENGINE_DEST_DIR || true

# 版本号
unix {
    BUILD_DATE = $$system(date +%Y%m%d%H%M%S)
}

MAJOR_VERSION = 1
MINOR_VERSION = 0

DEFINES += APP_VERSION_STR=\\\"v$${MAJOR_VERSION}.$${MINOR_VERSION}.$${BUILD_DATE}\\\"
