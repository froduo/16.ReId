#!/bin/bash
# 构建「动恰客流上行」副本工程 CameraViewer_dongqia
set -e
cd "$(dirname "$0")"

QMAKE=${QMAKE:-/home/froduo/Qt/6.2.4/gcc_64/bin/qmake}
BUILD_DIR=${BUILD_DIR:-build-dongqia}

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

"$QMAKE" ../CameraViewer_dongqia/CameraViewer.pro -spec linux-g++ CONFIG+=debug CONFIG+=qml_debug
make -j4
echo "BUILD_EXIT_CODE=$?"
echo "可执行文件: $(pwd)/CameraViewer"
